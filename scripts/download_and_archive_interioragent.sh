#!/usr/bin/env bash
set -uo pipefail

# Complete each tracked InteriorAgent room, archive it, then remove its expanded
# files and the transient Git LFS object cache. This keeps long-term inode usage
# close to one file per room while preserving on-demand RoboLab loading.

REPO_DIR="${INTERIORAGENT_ROOT:-/autodl-fs/data/spatialverse/InteriorAgent}"
ARCHIVE_DIR="${INTERIORAGENT_ARCHIVE_ROOT:-/autodl-fs/data/spatialverse/InteriorAgent_archives}"
LOG_DIR="${INTERIORAGENT_LOG_DIR:-/autodl-fs/data/spatialverse/logs}"
MIN_FREE_INODES="${MIN_FREE_INODES:-1000}"
LFS_CONCURRENCY="${LFS_CONCURRENCY:-1}"

mkdir -p "$ARCHIVE_DIR" "$LOG_DIR"
cd "$REPO_DIR" || exit 1

git remote set-url origin https://huggingface.co/datasets/spatialverse/InteriorAgent
git config --local --unset-all lfs.concurrenttransfers 2>/dev/null || true
git config --local lfs.concurrenttransfers "$LFS_CONCURRENCY"
git config --local lfs.basictransfersonly true

free_inodes() {
  df -Pi "$REPO_DIR" | awk 'NR==2 {print $4}'
}

pointer_count() {
  local scene_dir="$1"
  find "$scene_dir" -type f -size -200c \
    -exec grep -Il '^version https://git-lfs.github.com/spec/v1' {} + 2>/dev/null \
    | wc -l
}

clear_lfs_cache() {
  rm -rf -- "$REPO_DIR/.git/lfs/objects" "$REPO_DIR/.git/lfs/tmp"
  mkdir -p "$REPO_DIR/.git/lfs/objects" "$REPO_DIR/.git/lfs/tmp"
}

mapfile -t scenes < <(
  git ls-files | awk -F/ '/^kujiale_[0-9]+\// {print $1}' | sort -u
)

echo "[INFO] repo=$REPO_DIR archives=$ARCHIVE_DIR scenes=${#scenes[@]} free_inodes=$(free_inodes)"

success=0
failed=0
for scene in "${scenes[@]}"; do
  archive="$ARCHIVE_DIR/${scene}.tar.gz"
  # A completed archive is atomically renamed from ``.partial`` only after its
  # main USD has been verified. Trust that marker on resume instead of reading
  # hundreds of megabytes again for every already-finished scene.
  if [[ -s "$archive" ]]; then
    echo "[SKIP] $scene already archived"
    rm -rf -- "$REPO_DIR/$scene"
    continue
  fi

  available="$(free_inodes)"
  if (( available <= MIN_FREE_INODES )); then
    echo "[STOP] only $available free inodes remain (reserve=$MIN_FREE_INODES)"
    break
  fi

  echo "========== START $scene $(date -u '+%F %T UTC') free_inodes=$available =========="
  main_usd="$REPO_DIR/$scene/$scene.usda"
  remaining="$(pointer_count "$REPO_DIR/$scene")"
  if [[ ! -s "$main_usd" ]] || (( remaining != 0 )); then
    echo "[INFO] downloading $scene: remaining_pointers=$remaining"
    if ! git lfs pull origin --include="$scene/**" --exclude=""; then
      echo "[WARN] git lfs pull failed for $scene"
      clear_lfs_cache
      ((failed += 1))
      continue
    fi
  fi

  remaining="$(pointer_count "$REPO_DIR/$scene")"
  if [[ ! -s "$main_usd" ]] || head -c 80 "$main_usd" | grep -q 'version https://git-lfs.github.com/spec' || (( remaining != 0 )); then
    echo "[WARN] $scene incomplete after download: remaining_pointers=$remaining"
    clear_lfs_cache
    ((failed += 1))
    continue
  fi

  tmp_archive="$archive.partial"
  rm -f -- "$tmp_archive"
  if ! tar -C "$REPO_DIR" -I 'gzip -1' -cf "$tmp_archive" "$scene"; then
    echo "[WARN] archive creation failed for $scene"
    rm -f -- "$tmp_archive"
    clear_lfs_cache
    ((failed += 1))
    continue
  fi
  if ! tar -tzf "$tmp_archive" "$scene/$scene.usda" >/dev/null 2>&1; then
    echo "[WARN] archive verification failed for $scene"
    rm -f -- "$tmp_archive"
    clear_lfs_cache
    ((failed += 1))
    continue
  fi

  mv -- "$tmp_archive" "$archive"
  rm -rf -- "$REPO_DIR/$scene"
  clear_lfs_cache
  ((success += 1))
  echo "[OK] $scene -> $archive size=$(du -h "$archive" | awk '{print $1}') free_inodes=$(free_inodes)"
done

clear_lfs_cache
echo "[DONE] success=$success failed=$failed archives=$(find "$ARCHIVE_DIR" -maxdepth 1 -name 'kujiale_*.tar.gz' | wc -l) free_inodes=$(free_inodes)"
exit "$(( failed > 0 ? 2 : 0 ))"
