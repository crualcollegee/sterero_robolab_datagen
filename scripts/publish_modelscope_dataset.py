#!/usr/bin/env python3
"""Prepare a small valid stereo sample and publish it to ModelScope Hub.

The default invocation prepares ``SOURCE_DIR/samples`` and uploads:

* full tar archives + env_cfg.json -> data/<source-directory-name>
* all existing preview videos      -> previews/<source-directory-name>
* the inspectable 8-frame sample   -> samples/<source-directory-name>

Authentication is read from MODELSCOPE_API_TOKEN and is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


DEFAULT_SOURCE = Path(
    "/root/autodl-fs/robolab/output/stereo_dataset/global_stereo_2dflow_v6"
)
DEFAULT_REPO = "bairuide/robolab_stereo_2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument(
        "--sample-archive",
        type=Path,
        help="Archive used for the sample (default: first sorted tar.gz in source-dir).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare/check locally and print the remote plan without network access.",
    )
    return parser.parse_args()


def _archive_stem(path: Path) -> str:
    name = path.name
    if not name.endswith(".tar.gz"):
        raise ValueError(f"Expected a .tar.gz archive: {path}")
    return name[: -len(".tar.gz")]


def _selected_member(relative: PurePosixPath, frame_count: int) -> bool:
    parts = relative.parts
    if not parts or parts[0] != "camera00":
        return False
    if len(parts) == 2 and parts[1] in {"metadata.json", ".complete"}:
        return parts[1] == "metadata.json"
    if len(parts) == 3 and parts[1] == "camera":
        return parts[2] in {"left.npz", "right.npz", "intrinsics.json"}
    if len(parts) != 4 or parts[2] not in {"left", "right"}:
        return False
    try:
        frame_id = int(PurePosixPath(parts[3]).stem)
    except ValueError:
        return False
    if parts[1] == "rgb":
        return parts[3].endswith(".png") and frame_id < frame_count
    if parts[1] == "depth":
        return parts[3].endswith(".npy") and frame_id < frame_count
    if parts[1] == "flow":
        return parts[3].endswith(".npy") and frame_id < frame_count - 1
    if parts[1] == "flow_masks":
        return parts[3].endswith(".npz") and frame_id < frame_count - 1
    return False


def _extract_sample(archive: Path, destination: Path, frame_count: int) -> Path:
    scene_name = _archive_stem(archive)
    sequence_dir = destination / scene_name / "camera00"
    seen: set[str] = set()

    # Streaming mode makes one sequential pass through the large gzip archive.
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            if path.parts[0] != scene_name:
                continue
            relative = PurePosixPath(*path.parts[1:])
            if not _selected_member(relative, frame_count):
                continue
            output = destination / scene_name / Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            fileobj = source.extractfile(member)
            if fileobj is None:
                raise IOError(f"Could not read archive member: {member.name}")
            with fileobj, output.open("wb") as target:
                shutil.copyfileobj(fileobj, target, length=8 * 1024 * 1024)
            seen.add(str(relative))

    required = {
        "camera00/metadata.json",
        "camera00/camera/left.npz",
        "camera00/camera/right.npz",
        "camera00/camera/intrinsics.json",
    }
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Archive is missing required sample members: {missing}")
    return sequence_dir


def _trim_camera_npz(path: Path, frame_count: int) -> None:
    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        expected = {"frame_ids", "intrinsics", "camera_to_world"}
        if fields != expected:
            raise ValueError(f"Unexpected camera fields in {path}: {sorted(fields)}")
        frame_ids = np.asarray(data["frame_ids"][:frame_count], dtype=np.int64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        camera_to_world = np.asarray(
            data["camera_to_world"][:frame_count], dtype=np.float32
        )
    np.savez_compressed(
        path,
        frame_ids=frame_ids,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )


def _update_metadata(path: Path, archive: Path, frame_count: int) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json is not a JSON object: {path}")
    metadata["num_frames"] = frame_count
    metadata["frame_ids"] = list(range(frame_count))
    metadata["flow_frame_count"] = frame_count - 1
    if isinstance(metadata.get("env_steps"), list):
        metadata["env_steps"] = metadata["env_steps"][:frame_count]
    metadata["sample_subset"] = {
        "source_archive": archive.name,
        "selection": f"first {frame_count} consecutive frames",
        "rgb_depth_pose_frames": frame_count,
        "forward_flow_frames": frame_count - 1,
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _expected_names(count: int, suffix: str) -> list[str]:
    return [f"{index:06d}{suffix}" for index in range(count)]


def _validate_sample(sequence_dir: Path, frame_count: int) -> tuple[int, int]:
    intrinsics_info = json.loads(
        (sequence_dir / "camera" / "intrinsics.json").read_text(encoding="utf-8")
    )
    width = int(intrinsics_info["left"]["width"])
    height = int(intrinsics_info["left"]["height"])
    if (width, height) != (
        int(intrinsics_info["right"]["width"]),
        int(intrinsics_info["right"]["height"]),
    ):
        raise ValueError("Left/right intrinsics resolutions differ")

    for side in ("left", "right"):
        rgb_dir = sequence_dir / "rgb" / side
        depth_dir = sequence_dir / "depth" / side
        flow_dir = sequence_dir / "flow" / side
        mask_dir = sequence_dir / "flow_masks" / side
        actual = sorted(path.name for path in rgb_dir.glob("*.png"))
        if actual != _expected_names(frame_count, ".png"):
            raise ValueError(f"Bad RGB frame list for {side}: {actual}")
        if sorted(path.name for path in depth_dir.glob("*.npy")) != _expected_names(
            frame_count, ".npy"
        ):
            raise ValueError(f"Bad depth frame list for {side}")
        if sorted(path.name for path in flow_dir.glob("*.npy")) != _expected_names(
            frame_count - 1, ".npy"
        ):
            raise ValueError(f"Bad flow frame list for {side}")
        if sorted(path.name for path in mask_dir.glob("*.npz")) != _expected_names(
            frame_count - 1, ".npz"
        ):
            raise ValueError(f"Bad flow-mask frame list for {side}")

        for index in range(frame_count):
            with Image.open(rgb_dir / f"{index:06d}.png") as image:
                if image.mode != "RGB" or image.size != (width, height):
                    raise ValueError(
                        f"Invalid RGB {side}/{index:06d}: mode={image.mode} size={image.size}"
                    )
            depth = np.load(depth_dir / f"{index:06d}.npy", allow_pickle=False)
            if depth.dtype != np.float32 or depth.shape != (height, width):
                raise ValueError(f"Invalid depth shape/dtype for {side}/{index:06d}")
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError(f"Invalid depth values for {side}/{index:06d}")

        for index in range(frame_count - 1):
            flow = np.load(flow_dir / f"{index:06d}.npy", allow_pickle=False)
            if flow.dtype != np.float32 or flow.shape != (height, width, 2):
                raise ValueError(f"Invalid flow shape/dtype for {side}/{index:06d}")
            with np.load(mask_dir / f"{index:06d}.npz", allow_pickle=False) as mask:
                if set(mask.files) != {"flow_invalid"}:
                    raise ValueError(f"Invalid flow-mask fields for {side}/{index:06d}")
                invalid = mask["flow_invalid"]
                if invalid.dtype != np.bool_ or invalid.shape != (height, width):
                    raise ValueError(f"Invalid flow-mask shape/dtype for {side}/{index:06d}")
            if not np.isfinite(flow[~invalid]).all():
                raise ValueError(f"Non-finite valid flow for {side}/{index:06d}")
            native_sentinel = np.all(flow == np.float32(-1.0), axis=-1)
            if np.any(native_sentinel & (~invalid)):
                raise ValueError(
                    f"Unmasked native (-1,-1) flow sentinel for {side}/{index:06d}"
                )

        with np.load(sequence_dir / "camera" / f"{side}.npz", allow_pickle=False) as camera:
            if set(camera.files) != {"frame_ids", "intrinsics", "camera_to_world"}:
                raise ValueError(f"Invalid camera fields for {side}")
            if not np.array_equal(camera["frame_ids"], np.arange(frame_count)):
                raise ValueError(f"Invalid camera frame_ids for {side}")
            if camera["intrinsics"].dtype != np.float32 or camera["intrinsics"].shape != (3, 3):
                raise ValueError(f"Invalid intrinsics for {side}")
            poses = camera["camera_to_world"]
            if poses.dtype != np.float32 or poses.shape != (frame_count, 4, 4):
                raise ValueError(f"Invalid camera_to_world for {side}")
            if not np.isfinite(poses).all():
                raise ValueError(f"Non-finite camera_to_world for {side}")
    return width, height


def _write_sample_readme(samples_dir: Path, archive: Path, frame_count: int) -> None:
    text = f"""# RoboLab stereo sample

This folder contains a loader-compatible subset of `{archive.name}`.

- RGB/depth/pose frames: {frame_count}
- Forward-flow/mask frames: {frame_count - 1}
- Selection: first {frame_count} consecutive frames
- The `.complete` marker was written only after the subset passed validation.
- The MP4 under `previews/` is for visual inspection and is not training data.

The full source archive is uploaded under `data/{archive.parent.name}/`.
"""
    (samples_dir / "README.md").write_text(text, encoding="utf-8")


def _write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == manifest:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing sample checksum manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, separator, relative_name = line.partition("  ")
        relative = PurePosixPath(relative_name)
        if (
            not separator
            or len(expected) != 64
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(f"Malformed checksum line: {line!r}")
        path = root / Path(*relative.parts)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Sample checksum mismatch: {relative_name}")


def prepare_sample(
    source_dir: Path, archive: Path, frame_count: int, force: bool
) -> Path:
    if frame_count < 2:
        raise ValueError("sample-frames must be at least 2")
    samples_dir = source_dir / "samples"
    if samples_dir.exists() and not force:
        sequence_dir = samples_dir / _archive_stem(archive) / "camera00"
        _validate_sample(sequence_dir, frame_count)
        if not (sequence_dir / ".complete").is_file():
            raise ValueError(f"Prepared sample has no .complete marker: {sequence_dir}")
        _verify_manifest(samples_dir)
        print(
            f"[OK] Reusing validated sample: {samples_dir} "
            f"({frame_count} frames)"
        )
        return samples_dir
    staging = source_dir / f".samples.staging.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sequence_dir = _extract_sample(archive, staging, frame_count)
        for side in ("left", "right"):
            _trim_camera_npz(sequence_dir / "camera" / f"{side}.npz", frame_count)
        _update_metadata(sequence_dir / "metadata.json", archive, frame_count)
        width, height = _validate_sample(sequence_dir, frame_count)
        (sequence_dir / ".complete").write_text("ok\n", encoding="utf-8")

        preview_dir = staging / "previews"
        preview_dir.mkdir()
        source_preview = source_dir / "preview"
        for preview in sorted(source_preview.glob(f"{_archive_stem(archive)}*.mp4")):
            shutil.copy2(preview, preview_dir / preview.name)
        _write_sample_readme(staging, archive, frame_count)
        _write_manifest(staging)

        if samples_dir.exists():
            shutil.rmtree(samples_dir)
        staging.replace(samples_dir)
        print(
            f"[OK] Prepared validated sample: {samples_dir} "
            f"({frame_count} frames, {width}x{height})"
        )
        return samples_dir
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _size_summary(paths: list[Path]) -> str:
    total = sum(path.stat().st_size for path in paths)
    return f"{len(paths)} files, {total / (1024**3):.2f} GiB"


def upload(source_dir: Path, samples_dir: Path, args: argparse.Namespace) -> None:
    archives = sorted(source_dir.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz archives in {source_dir}")
    previews = sorted((source_dir / "preview").glob("*.mp4"))
    dataset_name = source_dir.name
    data_path = f"data/{dataset_name}"
    preview_path = f"previews/{dataset_name}"
    sample_path = f"samples/{dataset_name}"
    print(f"[PLAN] Full data: {_size_summary(archives)} -> {data_path}/")
    print(f"[PLAN] Previews: {_size_summary(previews)} -> {preview_path}/")
    print(f"[PLAN] Sample: {samples_dir} -> {sample_path}/")
    if args.dry_run:
        print("[OK] Dry run finished; no network requests were made.")
        return

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "MODELSCOPE_API_TOKEN is not set. Export it before running the upload."
        )
    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise RuntimeError(
            "modelscope-hub is required: uv pip install modelscope-hub"
        ) from exc

    api = HubApi(token=token, endpoint="https://modelscope.cn")
    identity = api.whoami()
    api.get_repo(args.repo_id, repo_type="dataset")
    username = getattr(identity, "username", "authenticated user")
    print(f"[OK] Authenticated as {username}; repository is accessible.")

    common = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "max_workers": args.max_workers,
        "use_cache": True,
    }
    api.upload_folder(
        folder_path=source_dir,
        path_in_repo=data_path,
        allow_patterns=["*.tar.gz", "env_cfg.json"],
        commit_message=f"Upload RoboLab stereo {dataset_name} archives",
        **common,
    )
    if previews:
        api.upload_folder(
            folder_path=source_dir / "preview",
            path_in_repo=preview_path,
            commit_message=f"Upload RoboLab stereo {dataset_name} previews",
            **common,
        )
    api.upload_folder(
        folder_path=samples_dir,
        path_in_repo=sample_path,
        commit_message="Upload validated RoboLab stereo sample",
        **common,
    )
    print(f"[DONE] Published to https://modelscope.cn/datasets/{args.repo_id}")


def main() -> int:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    archives = sorted(source_dir.glob("*.tar.gz"))
    archive = (args.sample_archive or (archives[0] if archives else None))
    if archive is None:
        raise FileNotFoundError(f"No .tar.gz archives in {source_dir}")
    archive = archive.resolve()
    if archive.parent != source_dir or not archive.is_file():
        raise ValueError(f"sample-archive must be a file directly inside {source_dir}")

    samples_dir = source_dir / "samples"
    if not args.skip_prepare:
        samples_dir = prepare_sample(
            source_dir, archive, args.sample_frames, args.force_sample
        )
    elif not samples_dir.is_dir():
        raise FileNotFoundError(f"Prepared sample directory does not exist: {samples_dir}")

    if args.prepare_only:
        return 0
    upload(source_dir, samples_dir, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[ERROR] Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
