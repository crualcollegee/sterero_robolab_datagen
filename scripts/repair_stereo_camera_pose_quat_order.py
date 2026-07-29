#!/usr/bin/env python3
"""Repair stereo directory/tar camera_to_world matrices written with wxyz as xyzw.

Old bug: IsaacLab camera.data.quat_w_ros is ordered (w, x, y, z), but the
writer interpreted it as (x, y, z, w). The resulting camera_to_world rotation
is wrong while translation, intrinsics, RGB, and depth are still usable.

This script recovers the wrongly-interpreted quaternion from each saved
rotation matrix, reorders it back to xyzw, rebuilds the rotation, and rewrites
left.npz/right.npz.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def mat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def repair_c2w(c2w: np.ndarray) -> np.ndarray:
    fixed = np.asarray(c2w, dtype=np.float32).copy()
    for i in range(fixed.shape[0]):
        # Old R = matrix_from_xyzw([w, x, y, z]). Recover those four stored
        # components, then rebuild true matrix_from_xyzw([x, y, z, w]).
        old_xyzw = mat_to_quat_xyzw(fixed[i, :3, :3])
        true_xyzw = np.asarray([old_xyzw[1], old_xyzw[2], old_xyzw[3], old_xyzw[0]], dtype=np.float64)
        fixed[i, :3, :3] = quat_xyzw_to_matrix(true_xyzw)
    return fixed


def repair_npz(npz_path: Path) -> None:
    data = np.load(npz_path)
    required = {"frame_ids", "intrinsics", "camera_to_world"}
    if set(data.files) != required:
        raise ValueError(f"Unexpected NPZ fields in {npz_path}: {data.files}")
    c2w = repair_c2w(data["camera_to_world"])
    tmp = npz_path.with_name(npz_path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, frame_ids=data["frame_ids"], intrinsics=data["intrinsics"], camera_to_world=c2w)
    tmp.replace(npz_path)


def repair_sequence_dir(scene_dir: Path) -> int:
    count = 0
    for npz_path in sorted(scene_dir.glob("camera*/camera/*.npz")):
        if npz_path.name not in {"left.npz", "right.npz"}:
            continue
        repair_npz(npz_path)
        count += 1
    for meta_path in scene_dir.glob("camera*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["pose_repaired_from_quat_order_bug"] = True
        meta["camera_axes"] = "+x right, +y down, +z forward"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return count


def repair_tar(tar_path: Path, backup: bool) -> int:
    parent = tar_path.parent
    with tempfile.TemporaryDirectory(prefix="pose_repair_", dir=str(parent)) as tmp_name:
        tmp_dir = Path(tmp_name)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_dir)
        roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise ValueError(f"Expected one scene directory in {tar_path}, got {roots}")
        count = repair_sequence_dir(roots[0])
        new_tar = tar_path.with_suffix(tar_path.suffix + ".tmp")
        with tarfile.open(new_tar, "w:gz") as tar:
            tar.add(roots[0], arcname=roots[0].name)
        if backup:
            backup_path = tar_path.with_suffix(tar_path.suffix + ".bak")
            shutil.move(tar_path, backup_path)
        new_tar.replace(tar_path)
        return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Scene directories or .tar.gz archives to repair.")
    parser.add_argument("--backup", action="store_true", help="Keep .tar.gz.bak before replacing archives.")
    args = parser.parse_args()
    total = 0
    for raw in args.paths:
        path = Path(raw)
        if path.suffixes[-2:] == [".tar", ".gz"]:
            n = repair_tar(path, backup=args.backup)
        elif path.is_dir():
            n = repair_sequence_dir(path)
        else:
            raise ValueError(f"Unsupported path: {path}")
        total += n
        print(f"[OK] repaired {n} camera NPZ files: {path}")
    print(f"[DONE] repaired total camera NPZ files: {total}")


if __name__ == "__main__":
    main()
