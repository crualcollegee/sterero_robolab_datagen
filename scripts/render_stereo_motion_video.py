#!/usr/bin/env python3
"""Visualize stereo RGB-D, raw flow, ego-compensated flow, and fixed-view motion."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

from robolab.core.scene_flow import remove_camera_motion


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera", default=None, help="One camera directory to render.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Render multiple camera directories from each input archive in one extraction pass.",
    )
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--eye-width", type=int, default=480)
    parser.add_argument("--flow2d-max-px", type=float, default=20.0)
    parser.add_argument(
        "--motion-threshold-px",
        type=float,
        default=1.5,
        help="Residual-flow magnitude threshold in original-image pixels.",
    )
    parser.add_argument(
        "--fixed-view-frame",
        type=int,
        default=0,
        help="Frame whose left/right poses define the fixed observer.",
    )
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--ffmpeg-preset", default="fast")
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    return parser.parse_args()


def extract_cameras(
    archive_path: Path, cameras: tuple[str, ...], root: Path
) -> dict[str, Path]:
    markers = {camera: f"/{camera}/" for camera in cameras}
    camera_roots = {camera: root / camera for camera in cameras}
    counts = {camera: 0 for camera in cameras}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            wrapped_name = f"/{member.name}"
            if not member.isfile():
                continue
            selected = next(
                (camera for camera, marker in markers.items() if marker in wrapped_name),
                None,
            )
            if selected is None:
                continue
            relative = wrapped_name.split(markers[selected], 1)[1]
            if not (
                relative.startswith(("rgb/", "depth/", "flow/", "flow_masks/", "flow_valid/", "camera/"))
                or relative == "metadata.json"
            ):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target = camera_roots[selected] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            counts[selected] += 1
    missing = [camera for camera, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Cameras not found in {archive_path}: {missing}")
    return camera_roots


def extract_camera(archive_path: Path, camera: str, root: Path) -> Path:
    return extract_cameras(archive_path, (camera,), root)[camera]


def camera_root(input_path: Path, camera: str, temp_root: Path) -> Path:
    if input_path.is_dir():
        path = input_path if input_path.name == camera else input_path / camera
        if not path.is_dir():
            raise ValueError(f"{camera} was not found below {input_path}")
        return path
    if input_path.name.lower().endswith((".tar.gz", ".tgz")):
        return extract_camera(input_path, camera, temp_root)
    raise ValueError("Expected a tar.gz archive or an extracted scene directory")


def camera_has_flow(camera_dir: Path) -> bool:
    """Return whether this camera sequence stores the optional flow pair."""
    flow = camera_dir / "flow"
    masks = camera_dir / "flow_masks"
    if flow.is_dir() != masks.is_dir():
        raise ValueError(
            f"flow and flow_masks must both exist or both be absent in {camera_dir}"
        )
    return flow.is_dir()


def depth_color(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros_like(depth)
    if valid.any():
        near, far = np.percentile(depth[valid], (2, 98))
        scaled[valid] = np.clip(
            (depth[valid] - near) / max(float(far - near), 1e-6), 0, 1
        )
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def flow2d_color(flow: np.ndarray, valid: np.ndarray, maximum: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0], flow[..., 1], angleInDegrees=True
    )
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = np.mod(angle / 2, 180).astype(np.uint8)
    # Encode magnitude as saturation instead of brightness.  This keeps every
    # valid vector bright (small flow is pale, large flow is vivid), while
    # reserving black exclusively for invalid flow.  A brightness floor still
    # looked black for saturated red/blue after resizing and H.264 encoding.
    hsv[..., 1] = np.clip(
        magnitude / max(maximum, 1e-6) * 255, 0, 255
    ).astype(np.uint8)
    hsv[..., 2] = 255
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image[~valid] = 0
    return image


def motion_mask_color(motion_mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Render derived object motion in white and invalid flow in dark red."""
    image = np.zeros(motion_mask.shape + (3,), dtype=np.uint8)
    image[valid] = (32, 32, 32)
    image[motion_mask] = (255, 255, 255)
    image[~valid] = (0, 0, 96)
    return image


def flow_valid_color(valid: np.ndarray) -> np.ndarray:
    """Render flow validity explicitly: valid white, invalid bright red."""
    image = np.full(valid.shape + (3,), 255, dtype=np.uint8)
    image[~valid] = (0, 0, 255)
    return image


def panel(image: np.ndarray, label: str, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def resize_rgbd_flow(
    rgb: np.ndarray,
    depth: np.ndarray,
    flow: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    source_height, source_width = depth.shape
    height = max(1, int(round(source_height * width / source_width)))
    scale_x = width / source_width
    scale_y = height / source_height
    size = (width, height)
    rgb_small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    # Flow is defined on the source pixel grid.  Bilinear resizing blends
    # vectors from different surfaces at depth discontinuities while depth and
    # validity use nearest-neighbour sampling below.  That mismatch creates
    # thin false-positive "motion" outlines after camera-motion removal.  Keep
    # all three source-grid quantities sampled from the same pixel instead.
    flow_small = cv2.resize(flow, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    flow_small[..., 0] *= scale_x
    flow_small[..., 1] *= scale_y
    valid_small = cv2.resize(
        valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    k_small = np.asarray(intrinsics, dtype=np.float32).copy()
    k_small[0, 0] *= scale_x
    k_small[0, 2] *= scale_x
    k_small[1, 1] *= scale_y
    k_small[1, 2] *= scale_y
    return rgb_small, depth_small, flow_small, valid_small, k_small, scale_x


def project_to_fixed_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    motion_mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    reference_camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = depth.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    reference = np.asarray(reference_camera_to_world, dtype=np.float64)
    points_camera = np.stack(
        (
            (x - k[0, 2]) * depth / k[0, 0],
            (y - k[1, 2]) * depth / k[1, 1],
            depth,
        ),
        axis=-1,
    )
    points_world = points_camera @ c2w[:3, :3].T + c2w[:3, 3]
    points_reference = (
        points_world - reference[:3, 3]
    ) @ reference[:3, :3]
    z = points_reference[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_x = k[0, 0] * points_reference[..., 0] / z + k[0, 2]
        projected_y = k[1, 1] * points_reference[..., 1] / z + k[1, 2]

    safe_x = np.nan_to_num(projected_x, nan=-1.0, posinf=-1.0, neginf=-1.0)
    safe_y = np.nan_to_num(projected_y, nan=-1.0, posinf=-1.0, neginf=-1.0)
    target_x = np.rint(safe_x).astype(np.int32)
    target_y = np.rint(safe_y).astype(np.int32)
    source_valid = np.isfinite(depth) & (depth > 0.0)
    source_valid &= np.isfinite(projected_x) & np.isfinite(projected_y) & (z > 0.0)
    source_valid &= (target_x >= 0) & (target_x < width)
    source_valid &= (target_y >= 0) & (target_y < height)

    source_indices = np.flatnonzero(source_valid)
    target_indices = (target_y[source_valid] * width + target_x[source_valid]).astype(np.int64)
    source_z = z[source_valid].astype(np.float32)
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, target_indices, source_z)
    visible = source_z <= z_buffer[target_indices] + 1e-4
    source_indices = source_indices[visible]
    target_indices = target_indices[visible]

    fixed_rgb = np.zeros_like(rgb)
    fixed_motion = np.zeros(height * width, dtype=bool)
    fixed_rgb.reshape(-1, 3)[target_indices] = rgb.reshape(-1, 3)[source_indices]
    moving_sources = motion_mask.reshape(-1)[source_indices]
    fixed_motion[target_indices[moving_sources]] = True
    fixed_motion = fixed_motion.reshape(height, width)

    overlay = fixed_rgb.copy()
    if fixed_motion.any():
        red = np.zeros_like(overlay)
        red[..., 2] = 255
        overlay[fixed_motion] = cv2.addWeighted(
            overlay[fixed_motion], 0.35, red[fixed_motion], 0.65, 0
        )
    coverage = float(np.mean(np.isfinite(z_buffer)))
    return overlay, fixed_motion, coverage


def render(camera_dir: Path, output: Path, options) -> int:
    has_flow = camera_has_flow(camera_dir)
    frame_ids = sorted(
        path.stem
        for path in (
            camera_dir / ("flow" if has_flow else "rgb") / "left"
        ).glob("*.npy" if has_flow else "*.png")
    )
    if not frame_ids:
        raise ValueError(f"No renderable frames in {camera_dir}")

    cameras = {}
    for side in ("left", "right"):
        with np.load(camera_dir / "camera" / f"{side}.npz") as data:
            cameras[side] = (
                np.asarray(data["intrinsics"], dtype=np.float32),
                np.asarray(data["camera_to_world"], dtype=np.float32),
            )
    pose_count = min(poses.shape[0] for _, poses in cameras.values())
    if has_flow and not 0 <= options.fixed_view_frame < pose_count:
        raise ValueError(
            f"--fixed-view-frame must be in [0, {pose_count - 1}], "
            f"got {options.fixed_view_frame}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robolab_motion_frames_") as frame_temp:
        frames = Path(frame_temp)
        for output_index, frame_id in enumerate(frame_ids):
            index = int(frame_id)
            rows = []
            for side in ("left", "right"):
                rgb = cv2.imread(str(camera_dir / "rgb" / side / f"{frame_id}.png"))
                depth = np.load(
                    camera_dir / "depth" / side / f"{frame_id}.npy",
                    allow_pickle=False,
                ).astype(np.float32)
                name = side.upper()
                if not has_flow:
                    source_height, source_width = depth.shape
                    height = max(
                        1,
                        int(round(source_height * options.eye_width / source_width)),
                    )
                    size = (options.eye_width, height)
                    rgb_small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                    depth_small = cv2.resize(
                        depth, size, interpolation=cv2.INTER_NEAREST
                    )
                    rows.append(
                        [
                            panel(
                                rgb_small,
                                f"{name} RGB frame={frame_id}",
                                options.eye_width,
                            ),
                            panel(
                                depth_color(depth_small),
                                f"{name} DEPTH meters; FLOW NOT STORED",
                                options.eye_width,
                            ),
                        ]
                    )
                    continue
                flow = np.load(
                    camera_dir / "flow" / side / f"{frame_id}.npy",
                    allow_pickle=False,
                ).astype(np.float32)
                mask_path = camera_dir / "flow_masks" / side / f"{frame_id}.npz"
                if mask_path.is_file():
                    with np.load(mask_path, allow_pickle=False) as masks:
                        if "flow_invalid" not in masks:
                            raise ValueError(f"Missing flow_invalid in {mask_path}")
                        valid = ~np.asarray(masks["flow_invalid"], dtype=bool)
                else:
                    valid = np.load(
                        camera_dir / "flow_valid" / side / f"{frame_id}.npy",
                        allow_pickle=False,
                    ).astype(bool)
                intrinsics, poses = cameras[side]
                if index + 1 >= len(poses):
                    raise ValueError(f"Missing next pose for flow frame {frame_id}/{side}")

                rgb_small, depth_small, flow_small, valid_small, k_small, scale = (
                    resize_rgbd_flow(
                        rgb, depth, flow, valid, intrinsics, options.eye_width
                    )
                )
                residual, _, residual_valid = remove_camera_motion(
                    depth_small,
                    flow_small,
                    k_small,
                    poses[index],
                    poses[index + 1],
                    valid_small,
                )
                threshold = options.motion_threshold_px * scale
                motion_mask = residual_valid & (
                    np.linalg.norm(residual, axis=-1) > threshold
                )
                rows.append(
                    [
                        panel(
                            rgb_small,
                            f"{name} RGB frame={frame_id}",
                            options.eye_width,
                        ),
                        panel(
                            depth_color(depth_small),
                            f"{name} DEPTH meters",
                            options.eye_width,
                        ),
                        panel(
                            motion_mask_color(motion_mask, residual_valid),
                            f"{name} MOTION MASK white=moving ratio={np.mean(motion_mask):.3f}",
                            options.eye_width,
                        ),
                        panel(
                            flow2d_color(
                                flow_small,
                                valid_small,
                                options.flow2d_max_px * scale,
                            ),
                            f"{name} FORWARD FLOW pale=small black=invalid",
                            options.eye_width,
                        ),
                        panel(
                            flow_valid_color(valid_small),
                            f"{name} FLOW VALID white=valid red=invalid "
                            f"ratio={np.mean(valid_small):.4f}",
                            options.eye_width,
                        ),
                    ]
                )
            image = np.vstack([np.hstack(row) for row in rows])
            cv2.imwrite(str(frames / f"{output_index:06d}.png"), image)

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(options.fps),
                "-i",
                str(frames / "%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                options.ffmpeg_preset,
                "-threads",
                str(options.ffmpeg_threads),
                "-crf",
                str(options.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    return len(frame_ids)


def main():
    options = arguments()
    if options.camera is not None and options.cameras is not None:
        raise ValueError("Use either --camera or --cameras, not both")
    camera_names = tuple(options.cameras or [options.camera or "camera00"])
    if len(set(camera_names)) != len(camera_names):
        raise ValueError(f"Duplicate camera names: {camera_names}")
    for source in options.inputs:
        source = source.resolve()
        stem = source.name.removesuffix(".tar.gz").removesuffix(".tgz")
        with tempfile.TemporaryDirectory(prefix="robolab_motion_extract_") as temp:
            if source.name.lower().endswith((".tar.gz", ".tgz")):
                camera_dirs = extract_cameras(source, camera_names, Path(temp))
            else:
                camera_dirs = {
                    camera: camera_root(source, camera, Path(temp))
                    for camera in camera_names
                }
            for camera in camera_names:
                suffix = (
                    "motion_verify"
                    if camera_has_flow(camera_dirs[camera])
                    else "rgbd_verify"
                )
                output = options.output_dir / f"{stem}_{camera}_{suffix}.mp4"
                count = render(camera_dirs[camera], output, options)
                print(
                    f"[OK] camera={camera} frames={count} "
                    f"fps={options.fps:g} path={output}"
                )


if __name__ == "__main__":
    main()
