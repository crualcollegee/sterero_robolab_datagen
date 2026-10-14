#!/usr/bin/env python3
"""Render a side-by-side stereo RGB preview directly from dataset archives."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="camera00")
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--eye-width", type=int, default=960)
    parser.add_argument("--crf", type=int, default=20)
    return parser.parse_args()


def extract_stereo_rgb(archive_path: Path, camera: str, destination: Path) -> int:
    prefixes = {
        f"/{camera}/rgb/left/": destination / "left",
        f"/{camera}/rgb/right/": destination / "right",
    }
    for directory in prefixes.values():
        directory.mkdir(parents=True, exist_ok=True)

    counts = {"left": 0, "right": 0}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            target_dir = next((path for prefix, path in prefixes.items() if prefix in member.name), None)
            if target_dir is None:
                continue
            frame_name = Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                continue
            with (target_dir / frame_name).open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            counts[target_dir.name] += 1

    if counts["left"] == 0 or counts["left"] != counts["right"]:
        raise ValueError(f"Invalid stereo frame counts in {archive_path}: {counts}")
    return counts["left"]


def render_video(temp_root: Path, output_path: Path, fps: float, eye_width: int, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"[0:v]scale={eye_width}:-2,drawtext=text=LEFT:x=18:y=18:fontsize=28:"
        "fontcolor=white:box=1:boxcolor=black@0.55[l];"
        f"[1:v]scale={eye_width}:-2,drawtext=text=RIGHT:x=18:y=18:fontsize=28:"
        "fontcolor=white:box=1:boxcolor=black@0.55[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", str(temp_root / "left" / "%06d.png"),
        "-framerate", str(fps), "-i", str(temp_root / "right" / "%06d.png"),
        "-filter_complex", filter_graph, "-map", "[v]",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    for archive_path in args.archives:
        archive_path = archive_path.resolve()
        output_path = args.output_dir / f"{archive_path.name.removesuffix('.tar.gz')}_{args.camera}_stereo_rgb.mp4"
        print(f"[VIDEO] {archive_path.name} -> {output_path}", flush=True)
        with tempfile.TemporaryDirectory(prefix="robolab_video_", dir=archive_path.parent) as temp_dir:
            temp_root = Path(temp_dir)
            frame_count = extract_stereo_rgb(archive_path, args.camera, temp_root)
            render_video(temp_root, output_path, args.fps, args.eye_width, args.crf)
        print(f"[OK] frames={frame_count} fps={args.fps:g} path={output_path}", flush=True)


if __name__ == "__main__":
    main()
