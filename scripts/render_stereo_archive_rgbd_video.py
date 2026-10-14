#!/usr/bin/env python3
"""Render synchronized stereo RGB and depth from RoboLab archives or HDF5 files."""
from __future__ import annotations
import argparse, io, shutil, subprocess, tarfile, tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


HDF5_CAMERA_PAIRS = {
    "camera00": ("stereo_left_camera", "stereo_right_camera"),
    "camera03": ("stereo_front_left_camera", "stereo_front_right_camera"),
    "stereo": ("stereo_left_camera", "stereo_right_camera"),
    "global": ("stereo_left_camera", "stereo_right_camera"),
}
ARCHIVE_CAMERAS = tuple(f"camera{index:02d}" for index in range(5))


def args():
    p = argparse.ArgumentParser()
    p.add_argument("archives", nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--camera", default="camera00")
    p.add_argument("--fps", type=float, default=7.5)
    p.add_argument("--frame-step", type=int, default=1,
                   help="Use every Nth HDF5 frame (reduce fps by the same factor to preserve time).")
    p.add_argument("--eye-width", type=int, default=640)
    p.add_argument("--crf", type=int, default=20)
    return p.parse_args()


def colorize_depth(value):
    depth = np.asarray(value, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    unit = np.zeros(depth.shape, np.float32)
    if valid.any():
        near, far = np.percentile(depth[valid], [2, 98])
        unit[valid] = np.clip((depth[valid] - near) / max(float(far - near), 1e-6), 0, 1)
    stops = [0, .33, .66, 1]
    colors = np.array([[30, 40, 180], [0, 220, 255], [255, 235, 40], [210, 25, 20]])
    rgb = np.stack([np.interp(unit, stops, colors[:, c]) for c in range(3)], -1).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _output_dirs(root):
    dirs = {name: root / name for name in ("rgb_left", "rgb_right", "depth_left", "depth_right")}
    for path in dirs.values(): path.mkdir(parents=True)
    return dirs


def extract_tar(archive_path, camera, root):
    dirs = _output_dirs(root)
    markers = {
        f"/{camera}/rgb/left/": dirs["rgb_left"], f"/{camera}/rgb/right/": dirs["rgb_right"],
        f"/{camera}/depth/left/": dirs["depth_left"], f"/{camera}/depth/right/": dirs["depth_right"],
    }
    counts = {key: 0 for key in dirs}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile(): continue
            found = next(((marker, path) for marker, path in markers.items() if marker in member.name), None)
            if found is None: continue
            source = archive.extractfile(member)
            if source is None: continue
            path = found[1]
            output = path / (Path(member.name).stem + ".png")
            if path.name.startswith("rgb_") and member.name.endswith(".png"):
                with output.open("wb") as target: shutil.copyfileobj(source, target, 1024 * 1024)
            elif path.name.startswith("depth_") and member.name.endswith(".npy"):
                Image.fromarray(colorize_depth(np.load(io.BytesIO(source.read()), allow_pickle=False))).save(output)
            else: continue
            counts[path.name] += 1
    if min(counts.values()) == 0 or len(set(counts.values())) != 1:
        raise ValueError(f"Invalid RGB-D frame counts: {counts}")
    return counts["rgb_left"]


def extract_tar_cameras(archive_path, cameras, root):
    """Extract several directory-format cameras in one sequential gzip pass."""
    camera_dirs = {camera: _output_dirs(root / camera) for camera in cameras}
    markers = {}
    for camera, dirs in camera_dirs.items():
        markers.update({
            f"/{camera}/rgb/left/": (camera, dirs["rgb_left"]),
            f"/{camera}/rgb/right/": (camera, dirs["rgb_right"]),
            f"/{camera}/depth/left/": (camera, dirs["depth_left"]),
            f"/{camera}/depth/right/": (camera, dirs["depth_right"]),
        })
    counts = {
        camera: {name: 0 for name in dirs}
        for camera, dirs in camera_dirs.items()
    }
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            found = next(
                ((marker, destination) for marker, destination in markers.items()
                 if marker in member.name),
                None,
            )
            if found is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            camera, path = found[1]
            output = path / (Path(member.name).stem + ".png")
            if path.name.startswith("rgb_") and member.name.endswith(".png"):
                with output.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            elif path.name.startswith("depth_") and member.name.endswith(".npy"):
                Image.fromarray(
                    colorize_depth(np.load(io.BytesIO(source.read()), allow_pickle=False))
                ).save(output, compress_level=1)
            else:
                continue
            counts[camera][path.name] += 1
    for camera, camera_counts in counts.items():
        if min(camera_counts.values()) == 0 or len(set(camera_counts.values())) != 1:
            raise ValueError(f"Invalid RGB-D frame counts for {camera}: {camera_counts}")
    return {camera: camera_counts["rgb_left"] for camera, camera_counts in counts.items()}


def _save_preview_png(image, output, width):
    if width and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    # These PNGs are short-lived ffmpeg inputs. Low compression avoids spending
    # most of the preview time compressing thousands of full-resolution frames.
    image.save(output, compress_level=1)


def extract_hdf5(hdf5_path, camera, root, preview_width=None):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 input requires h5py") from exc

    if camera not in HDF5_CAMERA_PAIRS:
        choices = ", ".join(HDF5_CAMERA_PAIRS)
        raise ValueError(f"Unknown HDF5 camera {camera!r}; choose one of: {choices}")

    dirs = _output_dirs(root)
    left_name, right_name = HDF5_CAMERA_PAIRS[camera]
    with h5py.File(hdf5_path, "r") as archive:
        data = archive.get("data")
        demo_names = sorted(data.keys()) if data is not None else []
        if not demo_names:
            raise ValueError(f"No data/demo_* group found in {hdf5_path}")
        if len(demo_names) > 1:
            raise ValueError(
                f"HDF5 contains multiple demos {demo_names}; split the file or add demo selection support"
            )
        obs = archive[f"data/{demo_names[0]}/obs"]
        missing = [name for name in (left_name, right_name) if name not in obs]
        if missing:
            raise ValueError(f"HDF5 camera datasets are missing: {missing}; available={list(obs.keys())}")

        arrays = {
            "rgb_left": obs[left_name]["rgb"],
            "rgb_right": obs[right_name]["rgb"],
            "depth_left": obs[left_name]["depth"],
            "depth_right": obs[right_name]["depth"],
        }
        counts = {name: len(value) for name, value in arrays.items()}
        if min(counts.values()) == 0 or len(set(counts.values())) != 1:
            raise ValueError(f"Invalid RGB-D frame counts: {counts}")

        for name in ("rgb_left", "rgb_right"):
            for frame_id in range(counts[name]):
                filename = f"{frame_id:06d}.png"
                rgb = np.asarray(arrays[name][frame_id], dtype=np.uint8).squeeze()
                if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
                    raise ValueError(f"Invalid {name} frame shape at {frame_id}: {rgb.shape}")
                _save_preview_png(
                    Image.fromarray(rgb[..., :3], mode="RGB"),
                    dirs[name] / filename,
                    preview_width,
                )
        for name in ("depth_left", "depth_right"):
            for frame_id in range(counts[name]):
                filename = f"{frame_id:06d}.png"
                _save_preview_png(
                    Image.fromarray(colorize_depth(arrays[name][frame_id])),
                    dirs[name] / filename,
                    preview_width,
                )
    return counts["rgb_left"]


def extract(input_path, camera, root, preview_width=None):
    name = input_path.name.lower()
    if name.endswith((".hdf5", ".h5")):
        return extract_hdf5(input_path, camera, root, preview_width)
    if name.endswith((".tar.gz", ".tgz")):
        return extract_tar(input_path, camera, root)
    raise ValueError(f"Unsupported input format: {input_path}; expected .hdf5, .h5, .tar.gz, or .tgz")


def render(root, output, fps, width, crf):
    names = ("rgb_left", "rgb_right", "depth_left", "depth_right")
    labels = ("LEFT RGB", "RIGHT RGB", "LEFT DEPTH", "RIGHT DEPTH")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for name in names: cmd += ["-framerate", str(fps), "-i", str(root / name / "%06d.png")]
    filters = [f"[{i}:v]scale={width}:-2,drawtext=text='{label}':x=18:y=18:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55[v{i}]" for i, label in enumerate(labels)]
    filters += ["[v0][v1]hstack=inputs=2[top]", "[v2][v3]hstack=inputs=2[bottom]", "[top][bottom]vstack=inputs=2[v]"]
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True)


def render_hdf5_direct(input_path, camera, output, fps, width, crf, frame_step=1):
    """Stream HDF5 to ffmpeg in small batches without temporary PNG files."""
    import h5py

    if camera not in HDF5_CAMERA_PAIRS:
        choices = ", ".join(HDF5_CAMERA_PAIRS)
        raise ValueError(f"Unknown HDF5 camera {camera!r}; choose one of: {choices}")
    left_name, right_name = HDF5_CAMERA_PAIRS[camera]
    with h5py.File(input_path, "r") as archive:
        data = archive.get("data")
        demo_names = sorted(data.keys()) if data is not None else []
        if len(demo_names) != 1:
            raise ValueError(f"Expected one HDF5 demo, found {demo_names}")
        obs = archive[f"data/{demo_names[0]}/obs"]
        missing = [name for name in (left_name, right_name) if name not in obs]
        if missing:
            raise ValueError(f"HDF5 camera datasets are missing: {missing}; available={list(obs.keys())}")
        datasets = {
            "rgb_left": obs[left_name]["rgb"],
            "rgb_right": obs[right_name]["rgb"],
            "depth_left": obs[left_name]["depth"],
            "depth_right": obs[right_name]["depth"],
        }
        counts = {name: len(value) for name, value in datasets.items()}
        if min(counts.values()) == 0 or len(set(counts.values())) != 1:
            raise ValueError(f"Invalid RGB-D frame counts: {counts}")

        source_height, source_width = datasets["rgb_left"].shape[1:3]
        height = max(1, round(source_height * width / source_width))
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{2 * width}x{2 * height}",
            "-framerate", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        labels = ("LEFT RGB", "RIGHT RGB", "LEFT DEPTH", "RIGHT DEPTH")
        if frame_step < 1:
            raise ValueError("--frame-step must be >= 1")
        frame_ids = list(range(0, counts["rgb_left"], frame_step))
        batch_size = 4
        encoded_count = 0
        try:
            for batch_start in range(0, len(frame_ids), batch_size):
                ids = frame_ids[batch_start:batch_start + batch_size]
                # h5py fancy indexing is pathologically slow for these
                # per-frame, spatially chunked datasets. Read the few selected
                # frames individually, then stack them in memory.
                batch = {
                    name: np.stack([np.asarray(value[frame_id]) for frame_id in ids])
                    for name, value in datasets.items()
                }
                for offset in range(len(ids)):
                    frames = (
                        np.asarray(batch["rgb_left"][offset], dtype=np.uint8).squeeze()[..., :3],
                        np.asarray(batch["rgb_right"][offset], dtype=np.uint8).squeeze()[..., :3],
                        colorize_depth(batch["depth_left"][offset]),
                        colorize_depth(batch["depth_right"][offset]),
                    )
                    panels = []
                    for frame, label in zip(frames, labels):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        cv2.rectangle(frame, (0, 0), (180, 32), (0, 0, 0), -1)
                        cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
                        panels.append(frame)
                    canvas = np.concatenate(
                        (np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)),
                        axis=0,
                    )
                    process.stdin.write(np.ascontiguousarray(canvas).tobytes())
                    encoded_count += 1
                if encoded_count % 20 < batch_size or encoded_count == len(frame_ids):
                    print(f"[INFO] Encoded {encoded_count}/{len(frame_ids)} preview frames", flush=True)
        finally:
            if process.stdin is not None:
                process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return len(frame_ids)


def main():
    a = args()
    for archive in a.archives:
        archive = archive.resolve()
        stem = archive.name.removesuffix(".tar.gz").removesuffix(".tgz")
        if archive.suffix.lower() in (".hdf5", ".h5"):
            stem = archive.stem
        if a.camera == "all":
            if archive.suffix.lower() in (".hdf5", ".h5"):
                raise ValueError("--camera all currently requires a directory-format tar.gz archive")
            with tempfile.TemporaryDirectory(prefix="robolab_rgbd_all_") as temp:
                root = Path(temp)
                counts = extract_tar_cameras(archive, ARCHIVE_CAMERAS, root)
                for camera in ARCHIVE_CAMERAS:
                    output = a.output_dir / f"{stem}_{camera}_stereo_rgbd.mp4"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    render(root / camera, output, a.fps, a.eye_width, a.crf)
                    print(f"[OK] frames={counts[camera]} fps={a.fps:g} path={output}")
            continue
        output = a.output_dir / f"{stem}_{a.camera}_stereo_rgbd.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        if archive.suffix.lower() in (".hdf5", ".h5"):
            count = render_hdf5_direct(
                archive, a.camera, output, a.fps, a.eye_width, a.crf, a.frame_step
            )
        else:
            with tempfile.TemporaryDirectory(prefix="robolab_rgbd_") as temp:
                count = extract(archive, a.camera, Path(temp), a.eye_width)
                render(Path(temp), output, a.fps, a.eye_width, a.crf)
        print(f"[OK] frames={count} fps={a.fps:g} path={output}")


if __name__ == "__main__": main()
