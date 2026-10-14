#!/usr/bin/env python3
"""Upload one RoboLab archive with an ordered remote name and verify it."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
from pathlib import Path


PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--path-prefix", default="data")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--preview-prefix", default="preview")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--index-width", type=int, default=6)
    parser.add_argument("--buffer-size-mb", type=int, default=16)
    parser.add_argument("--verify-attempts", type=int, default=12)
    parser.add_argument("--verify-delay", type=float, default=5.0)
    parser.add_argument("--direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delete-after-verify", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_prefix(value: str) -> str:
    value = value.strip("/")
    if not value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Invalid remote path prefix: {value!r}")
    return value


def archive_label(path: Path) -> str:
    name = path.name
    if not name.endswith(".tar.gz"):
        raise ValueError(f"Expected a .tar.gz archive: {path}")
    stem = name[: -len(".tar.gz")]
    # The generator's per-process index is always 00000 when the streaming
    # wrapper invokes it once.  Remove that prefix so the remote index is the
    # only ordering key.
    stem = re.sub(r"^\d{8}_\d{6}_\d{5}_", "", stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")
    return stem or "trajectory"


def next_index(paths: list[str], prefix: str, width: int) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}/(\d{{{width}}})_")
    indices = []
    for path in paths:
        match = pattern.match(path)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(info) -> str | None:
    candidates = [getattr(info, "blob_id", None)]
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        for key in ("sha256", "Sha256", "oid", "Oid"):
            candidates.append(lfs.get(key))
    for value in candidates:
        if not isinstance(value, str):
            continue
        value = value.removeprefix("sha256:").lower()
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    return None


def list_files(api, repo_id: str, revision: str):
    # HubApi.list_repo_files uses the legacy dataset-tree endpoint without
    # pagination and is currently capped at 100 entries.  Use the SDK's
    # paginated dataset method so ordered numbering and preview verification
    # remain correct as the repository grows.
    paginated = getattr(api.legacy, "list_dataset_files_paginated", None)
    if paginated is None:
        return api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
        )

    from modelscope_hub.types import FileInfo

    raw_files = paginated(
        repo_id=repo_id,
        revision=revision,
        page_size=200,
        root_path="/",
    )
    files = []
    for item in raw_files:
        normalised = {
            "path": item.get("Path") or item.get("path") or item.get("Name") or "",
            "size": int(item.get("Size") or item.get("size") or 0),
            "blob_id": item.get("BlobId") or item.get("blob_id") or item.get("Sha256"),
            "type": item.get("Type") or item.get("type") or "blob",
            "last_modified": item.get("CommittedDate") or item.get("last_modified"),
            "lfs": item.get("Lfs") or item.get("lfs"),
        }
        files.append(FileInfo.from_dict(normalised))
    return files


def preview_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if not path.is_dir():
        raise FileNotFoundError(f"Preview directory does not exist: {path}")
    files = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"Preview directory contains no images: {path}")
    return files


def main() -> int:
    args = arguments()
    archive = args.archive.resolve()
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise FileNotFoundError(f"Archive is missing or empty: {archive}")
    if args.index is not None and args.index < 0:
        raise ValueError("--index must be non-negative")
    if args.index_width < 1:
        raise ValueError("--index-width must be positive")
    prefix = clean_prefix(args.path_prefix)
    preview_prefix = clean_prefix(args.preview_prefix)
    previews = preview_files(args.preview_dir.resolve() if args.preview_dir else None)

    if args.direct:
        for name in PROXY_VARIABLES:
            os.environ.pop(name, None)

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token and not args.dry_run:
        raise RuntimeError("MODELSCOPE_API_TOKEN is not set")

    from modelscope_hub import HubApi

    api = HubApi(token=token, endpoint="https://modelscope.cn")
    api.get_repo(args.repo_id, repo_type="dataset")
    files = list_files(api, args.repo_id, args.revision)
    label = archive_label(archive)
    ordered_pattern = re.compile(
        rf"^{re.escape(prefix)}/(\d{{{args.index_width}}})_{re.escape(label)}\.tar\.gz$"
    )
    prior_matches = [
        (entry, ordered_pattern.match(entry.path))
        for entry in files
        if ordered_pattern.match(entry.path)
    ]
    if len(prior_matches) > 1:
        raise RuntimeError(
            f"Multiple remote copies already match this archive label: "
            f"{[entry.path for entry, _ in prior_matches]}"
        )

    remote_entry = None
    if prior_matches:
        remote_entry, match = prior_matches[0]
        assert match is not None
        index = int(match.group(1))
        if args.index is not None and args.index != index:
            raise FileExistsError(
                f"Archive already exists remotely at index {index}, not requested index {args.index}"
            )
        remote_path = remote_entry.path
        print(f"[RESUME] Found matching remote object: {remote_path}", flush=True)
    else:
        index = args.index
        if index is None:
            index = next_index([entry.path for entry in files], prefix, args.index_width)
        remote_name = f"{index:0{args.index_width}d}_{label}.tar.gz"
        remote_path = f"{prefix}/{remote_name}"
        remote_entry = next((entry for entry in files if entry.path == remote_path), None)
        if remote_entry is not None:
            raise FileExistsError(f"Remote index is already occupied: {remote_path}")

    preview_remote_dir = f"{preview_prefix}/{index:0{args.index_width}d}_{label}"
    preview_remote_paths = {
        path: f"{preview_remote_dir}/{path.name}" for path in previews
    }

    local_size = archive.stat().st_size
    print(f"[PLAN] {archive} ({local_size / 1024**3:.2f} GiB)", flush=True)
    print(f"[PLAN] -> {args.repo_id}/{remote_path}", flush=True)
    if previews:
        print(
            f"[PLAN] {len(previews)} preview images -> "
            f"{args.repo_id}/{preview_remote_dir}/",
            flush=True,
        )
    if args.dry_run:
        print("[OK] Dry run only; nothing uploaded or deleted.", flush=True)
        return 0

    local_digest = None
    if not args.skip_sha256:
        print("[VERIFY] Computing local SHA-256 before upload...", flush=True)
        local_digest = sha256_file(archive)
        print(f"[VERIFY] Local SHA-256: {local_digest}", flush=True)

    if remote_entry is None:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=archive,
            path_in_repo=remote_path,
            revision=args.revision,
            buffer_size_mb=max(1, args.buffer_size_mb),
            commit_message=f"Upload RoboLab trajectory {Path(remote_path).name}",
        )
    else:
        print("[RESUME] Skipping upload and verifying the existing remote object.", flush=True)

    for preview, preview_remote_path in preview_remote_paths.items():
        print(f"[UPLOAD] Preview -> {preview_remote_path}", flush=True)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=preview,
            path_in_repo=preview_remote_path,
            revision=args.revision,
            buffer_size_mb=max(1, args.buffer_size_mb),
            commit_message=f"Upload preview for {Path(remote_path).name}",
        )

    verified = None
    last_error = None
    for attempt in range(1, max(1, args.verify_attempts) + 1):
        try:
            entries = list_files(api, args.repo_id, args.revision)
            verified = next((entry for entry in entries if entry.path == remote_path), None)
            if verified is not None and int(verified.size) == local_size:
                digest = remote_sha256(verified)
                if local_digest is None or digest is None or digest == local_digest:
                    break
                last_error = RuntimeError(
                    f"Remote SHA-256 mismatch: local={local_digest} remote={digest}"
                )
            elif verified is not None:
                last_error = RuntimeError(
                    f"Remote size mismatch: local={local_size} remote={verified.size}"
                )
        except Exception as exc:  # eventual consistency or a transient API error
            last_error = exc
        verified = None
        if attempt < args.verify_attempts:
            print(
                f"[VERIFY] Remote object not yet verified ({attempt}/{args.verify_attempts}); "
                f"retrying in {args.verify_delay:g}s...",
                flush=True,
            )
            time.sleep(max(0.0, args.verify_delay))
    if verified is None:
        raise RuntimeError(
            f"Upload returned but remote verification failed for {remote_path}: {last_error}"
        )

    if preview_remote_paths:
        preview_errors = []
        previews_verified = False
        for attempt in range(1, max(1, args.verify_attempts) + 1):
            preview_errors = []
            try:
                entries = list_files(api, args.repo_id, args.revision)
                entries_by_path = {entry.path: entry for entry in entries}
                for local_path, remote_preview_path in preview_remote_paths.items():
                    entry = entries_by_path.get(remote_preview_path)
                    if entry is None:
                        preview_errors.append(f"missing {remote_preview_path}")
                    elif int(entry.size) != local_path.stat().st_size:
                        preview_errors.append(
                            f"size mismatch {remote_preview_path}: "
                            f"local={local_path.stat().st_size} remote={entry.size}"
                        )
            except Exception as exc:
                preview_errors = [str(exc)]
            if not preview_errors:
                previews_verified = True
                break
            if attempt < args.verify_attempts:
                print(
                    f"[VERIFY] Remote previews not yet verified "
                    f"({attempt}/{args.verify_attempts}); retrying in "
                    f"{args.verify_delay:g}s...",
                    flush=True,
                )
                time.sleep(max(0.0, args.verify_delay))
        if not previews_verified:
            raise RuntimeError("Preview verification failed: " + "; ".join(preview_errors))
        print(
            f"[OK] Remote previews verified: {preview_remote_dir}/ "
            f"({len(previews)} images)",
            flush=True,
        )

    remote_digest = remote_sha256(verified)
    print(
        f"[OK] Remote verified: {remote_path}, size={verified.size}"
        + (f", sha256={remote_digest}" if remote_digest else ", SHA-256 unavailable from API"),
        flush=True,
    )
    if args.delete_after_verify:
        archive.unlink()
        print(f"[CLEANUP] Deleted verified local archive: {archive}", flush=True)
        if args.preview_dir is not None:
            shutil.rmtree(args.preview_dir.resolve())
            print(
                f"[CLEANUP] Deleted verified local previews: {args.preview_dir.resolve()}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[ERROR] Interrupted; local archive was retained.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise SystemExit(1)
