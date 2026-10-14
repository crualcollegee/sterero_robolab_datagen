#!/usr/bin/env python3
"""Copy evenly spaced RGB frames from a generated scene for human review."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera", default="camera03")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--count", type=int, default=6)
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length < 1:
        return []
    count = min(length, max(1, count))
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def prepare(scene_dir: Path, output_dir: Path, camera: str, side: str, count: int) -> list[Path]:
    if count < 1:
        raise ValueError("--count must be positive")
    source_dir = scene_dir / camera / "rgb" / side
    frames = sorted(source_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not frames:
        raise FileNotFoundError(f"No RGB preview frames found in {source_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        written = []
        for index in evenly_spaced_indices(len(frames), count):
            source = frames[index]
            destination = staging / f"{camera}_{side}_{source.name}"
            shutil.copy2(source, destination)
            written.append(destination)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        return [output_dir / path.name for path in written]
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = arguments()
    written = prepare(
        args.scene_dir.resolve(),
        args.output_dir.resolve(),
        args.camera,
        args.side,
        args.count,
    )
    print(
        f"[OK] Prepared {len(written)} RGB previews in {args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
