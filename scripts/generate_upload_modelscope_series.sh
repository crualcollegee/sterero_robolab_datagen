#!/usr/bin/env bash
set -euo pipefail

# Generate one validated five-camera archive at a time, upload it under an
# ordered ModelScope path, verify the remote object, then delete the local copy.

COUNT="${1:-}"
if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/generate_upload_modelscope_series.sh COUNT" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/autodl-fs/data/robolab}"
OUT_DIR="${OUT_DIR:-output/stereo_dataset/modelscope_spool_formal_v1}"
MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-bairuide/robolab_stereo_formal_v1}"
MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-master}"
MODELSCOPE_PATH_PREFIX="${MODELSCOPE_PATH_PREFIX:-data}"
MODELSCOPE_PREVIEW_PREFIX="${MODELSCOPE_PREVIEW_PREFIX:-preview}"
MODELSCOPE_DIRECT="${MODELSCOPE_DIRECT:-1}"
CLEAN_RUNTIME_CACHE_AFTER_EACH="${CLEAN_RUNTIME_CACHE_AFTER_EACH:-1}"
MAX_GENERATION_ATTEMPTS_PER_ITEM="${MAX_GENERATION_ATTEMPTS_PER_ITEM:-5}"
ARCHIVE_PREVIEW_CAMERA="${ARCHIVE_PREVIEW_CAMERA:-camera03}"
ARCHIVE_PREVIEW_SIDE="${ARCHIVE_PREVIEW_SIDE:-left}"
ARCHIVE_PREVIEW_COUNT="${ARCHIVE_PREVIEW_COUNT:-6}"

if [[ "$OUT_DIR" = /* || "$OUT_DIR" == *".."* ]]; then
  echo "[ERROR] OUT_DIR must be a repository-relative path without '..': $OUT_DIR" >&2
  exit 2
fi
if [[ -z "${MODELSCOPE_API_TOKEN:-}" ]]; then
  echo "[ERROR] MODELSCOPE_API_TOKEN is not set" >&2
  exit 2
fi
if ! [[ "$MAX_GENERATION_ATTEMPTS_PER_ITEM" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_GENERATION_ATTEMPTS_PER_ITEM must be a positive integer" >&2
  exit 2
fi
if ! [[ "$ARCHIVE_PREVIEW_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] ARCHIVE_PREVIEW_COUNT must be a positive integer" >&2
  exit 2
fi

cd "$REPO_DIR"
source .venv/bin/activate

mkdir -p \
  /root/autodl-tmp/robolab_interioragent_cache \
  /root/autodl-tmp/optix_cache \
  /root/autodl-tmp/robolab_tmp \
  "$OUT_DIR"

export INTERIORAGENT_RUNTIME_CACHE="${INTERIORAGENT_RUNTIME_CACHE:-/root/autodl-tmp/robolab_interioragent_cache}"
export OPTIX_CACHE_PATH="${OPTIX_CACHE_PATH:-/root/autodl-tmp/optix_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/robolab_tmp}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMNI_USER_CACHE_DIR="${OMNI_USER_CACHE_DIR:-/autodl-fs/data/ov_cache}"

# Preserve the user's current five-camera production conditions. Values remain
# overrideable by exporting the corresponding variable before invoking this
# script.
export FIVE_CAMERA_TRAJECTORY="${FIVE_CAMERA_TRAJECTORY:-1}"
export CAMERA03_ONLY="${CAMERA03_ONLY:-0}"
export TASK_PAIR_MODE="${TASK_PAIR_MODE:-0}"
export GEOMETRY_BACKGROUND_SOURCE="${GEOMETRY_BACKGROUND_SOURCE:-interioragent}"
export GEOMETRY_BACKGROUND_STYLE="${GEOMETRY_BACKGROUND_STYLE:-random}"
export GEOMETRY_BACKGROUND_PLACEMENT="${GEOMETRY_BACKGROUND_PLACEMENT:-surround}"
export GEOMETRY_BACKGROUND_SHELL="${GEOMETRY_BACKGROUND_SHELL:-off}"
export GEOMETRY_BACKGROUND_RANDOMIZE_POSITION="${GEOMETRY_BACKGROUND_RANDOMIZE_POSITION:-1}"
export RANDOMIZE_HDR_BACKGROUND="${RANDOMIZE_HDR_BACKGROUND:-0}"
export ADD_GEOMETRY_BACKGROUND="${ADD_GEOMETRY_BACKGROUND:-1}"
export CAMERA_VISUAL_QUALITY_GATE="${CAMERA_VISUAL_QUALITY_GATE:-0}"
export CAMERA_DEPTH_COVERAGE_GATE="${CAMERA_DEPTH_COVERAGE_GATE:-1}"
export POLICY_INTERACTION_GATE="${POLICY_INTERACTION_GATE:-0}"
export INJECTED_SCENE_STABILITY_GATE="${INJECTED_SCENE_STABILITY_GATE:-1}"
export STEPS="${STEPS:-600}"
export RECORD_EVERY="${RECORD_EVERY:-2}"
export WIDTH="${WIDTH:-1280}"
export HEIGHT="${HEIGHT:-720}"
export BASELINE="${BASELINE:-0.16}"
export WRIST_BASELINE="${WRIST_BASELINE:-0.05}"
export DISTRACTORS="${DISTRACTORS:-12}"
export DISTRACTOR_MODE="${DISTRACTOR_MODE:-real}"
export OUTPUT_FORMAT=directory
export ARCHIVE_AFTER_EACH=1
export ARCHIVE_COMPRESSION_LEVEL="${ARCHIVE_COMPRESSION_LEVEL:-1}"
export RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-18000}"
export CONTINUE_ON_ERROR=0

absolute_out="$REPO_DIR/$OUT_DIR"
absolute_preview_root="$absolute_out/.preview_staging"
mkdir -p "$absolute_preview_root"
export ARCHIVE_PREVIEW_ROOT="$absolute_preview_root"
export ARCHIVE_PREVIEW_CAMERA
export ARCHIVE_PREVIEW_SIDE
export ARCHIVE_PREVIEW_COUNT

cleanup_runtime_cache_since() {
  local marker="$1"
  [[ "$CLEAN_RUNTIME_CACHE_AFTER_EACH" == "1" ]] || return 0
  while IFS= read -r -d '' path; do
    echo "[CLEANUP] Removing per-run InteriorAgent cache: $path"
    rm -rf -- "$path"
  done < <(
    find "$INTERIORAGENT_RUNTIME_CACHE" -mindepth 1 -maxdepth 1 -type d \
      -newer "$marker" -print0 2>/dev/null
  )
}

publish_archive() {
  local archive="$1"
  local scene_name
  local preview_dir
  scene_name="$(basename "$archive" .tar.gz)"
  preview_dir="$absolute_preview_root/$scene_name"
  if [[ ! -d "$preview_dir" ]]; then
    echo "[ERROR] Preview staging directory is missing: $preview_dir" >&2
    return 1
  fi
  direct_args=()
  if [[ "$MODELSCOPE_DIRECT" == "1" ]]; then
    direct_args=(--direct)
  else
    direct_args=(--no-direct)
  fi
  PYTHONUNBUFFERED=1 uv run --no-sync python scripts/upload_modelscope_archive.py \
    "$archive" \
    --repo-id "$MODELSCOPE_REPO_ID" \
    --revision "$MODELSCOPE_REVISION" \
    --path-prefix "$MODELSCOPE_PATH_PREFIX" \
    --preview-dir "$preview_dir" \
    --preview-prefix "$MODELSCOPE_PREVIEW_PREFIX" \
    "${direct_args[@]}" \
    --delete-after-verify
}

# A previous interruption may have happened after archiving but before remote
# verification. Upload that archive first instead of producing another one.
mapfile -d '' pending < <(
  find "$absolute_out" -maxdepth 1 -type f -name '*.tar.gz' -print0 | sort -z
)
completed_this_run=0
if (( ${#pending[@]} > 1 )); then
  echo "[ERROR] More than one pending archive exists in spool: $absolute_out" >&2
  exit 1
elif (( ${#pending[@]} == 1 )); then
  echo "[RESUME] Publishing retained archive before generating new data: ${pending[0]}"
  publish_archive "${pending[0]}"
  completed_this_run=1
fi

for ((series_index = completed_this_run; series_index < COUNT; series_index++)); do
  generation_attempt=0
  while true; do
    generation_attempt=$((generation_attempt + 1))
    marker="$absolute_out/.generation_attempt.$$"
    touch "$marker"
    echo "[SERIES $((series_index + 1))/$COUNT] Generating one trajectory (attempt $generation_attempt/$MAX_GENERATION_ATTEMPTS_PER_ITEM)..."
    if OUT_DIR="$OUT_DIR" bash scripts/generate_random_complex_series.sh 1; then
      break
    fi
    echo "[WARN] Generation or a quality gate rejected this attempt; removing only incomplete files." >&2
    while IFS= read -r -d '' path; do
      echo "[CLEANUP] Removing incomplete output: $path"
      rm -rf -- "$path"
    done < <(
      find "$absolute_out" -mindepth 1 -maxdepth 1 -newer "$marker" \
        \( -type f -name '*.hdf5' -o -type d -name '20*' \) -print0
    )
    while IFS= read -r -d '' path; do
      echo "[CLEANUP] Removing preview staging from rejected attempt: $path"
      rm -rf -- "$path"
    done < <(
      find "$absolute_preview_root" -mindepth 1 -maxdepth 1 -type d \
        -newer "$marker" -print0 2>/dev/null
    )
    cleanup_runtime_cache_since "$marker"
    rm -f -- "$marker"
    if (( generation_attempt >= MAX_GENERATION_ATTEMPTS_PER_ITEM )); then
      echo "[ERROR] Exhausted $MAX_GENERATION_ATTEMPTS_PER_ITEM generation attempts for series item $((series_index + 1))." >&2
      exit 1
    fi
  done

  mapfile -d '' archives < <(
    find "$absolute_out" -maxdepth 1 -type f -name '*.tar.gz' -newer "$marker" -print0
  )
  if (( ${#archives[@]} != 1 )); then
    echo "[ERROR] Expected exactly one new archive, found ${#archives[@]}; local files retained." >&2
    cleanup_runtime_cache_since "$marker"
    rm -f -- "$marker"
    exit 1
  fi

  echo "[SERIES $((series_index + 1))/$COUNT] Uploading and verifying ${archives[0]}..."
  if ! publish_archive "${archives[0]}"; then
    echo "[ERROR] Upload or remote verification failed; local archive retained for resume." >&2
    cleanup_runtime_cache_since "$marker"
    rm -f -- "$marker"
    exit 1
  fi
  cleanup_runtime_cache_since "$marker"
  rm -f -- "$marker"
  echo "[SERIES $((series_index + 1))/$COUNT] Complete; local archive removed."
done

echo "[DONE] Generated, uploaded, verified, and cleaned $COUNT trajectories."
echo "[DONE] Dataset: https://modelscope.cn/datasets/$MODELSCOPE_REPO_ID"
