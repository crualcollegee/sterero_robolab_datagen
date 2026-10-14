#!/usr/bin/env bash
set -uo pipefail

TARGET="${1:-3}"
REPO_DIR="${REPO_DIR:-/autodl-fs/data/robolab}"
ARCHIVE_ROOT="${INTERIORAGENT_ARCHIVE_ROOT:-/autodl-fs/data/spatialverse/InteriorAgent_archives}"
OUT_DIR="${OUT_DIR:-output/stereo_dataset/interioragent_pool_$(date +%Y%m%d_%H%M%S)}"
VIDEO_DIR="$OUT_DIR/videos"

if ! [[ "$TARGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] target must be a positive integer" >&2
  exit 2
fi

cd "$REPO_DIR"
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export OMP_NUM_THREADS=8
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMNI_USER_CACHE_DIR="${OMNI_USER_CACHE_DIR:-/autodl-fs/data/ov_cache}"

mapfile -t backgrounds < <(
  find "$ARCHIVE_ROOT" -maxdepth 1 -type f -name 'kujiale_*.tar.gz' -printf '%f\n' \
    | sed 's/\.tar\.gz$//' \
    | shuf
)
if (( ${#backgrounds[@]} < TARGET )); then
  echo "[ERROR] only ${#backgrounds[@]} InteriorAgent archives available for target=$TARGET" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "[POOL] available=${#backgrounds[@]} target=$TARGET output=$REPO_DIR/$OUT_DIR"
printf '[POOL] order=%s\n' "${backgrounds[*]}"

success=0
attempt=0
for scene in "${backgrounds[@]}"; do
  ((attempt += 1))
  echo
  echo "========== attempt=$attempt success=$success/$TARGET background=$scene =========="

  OUT_DIR="$OUT_DIR" \
  GEOMETRY_BACKGROUND_SOURCE=interioragent \
  GEOMETRY_BACKGROUND_STYLE="interioragent_${scene}" \
  GEOMETRY_BACKGROUND_PLACEMENT=surround \
  GEOMETRY_BACKGROUND_SHELL=off \
  GEOMETRY_BACKGROUND_SAFETY=1 \
  GEOMETRY_BACKGROUND_RANDOMIZE_POSITION=1 \
  RANDOMIZE_HDR_BACKGROUND=0 \
  ADD_GEOMETRY_BACKGROUND=1 \
  STEPS="${STEPS:-300}" \
  RECORD_EVERY="${RECORD_EVERY:-2}" \
  SKIP_INITIAL_SECONDS="${SKIP_INITIAL_SECONDS:-1.0}" \
  WIDTH="${WIDTH:-1280}" \
  HEIGHT="${HEIGHT:-720}" \
  DISTRACTORS="${DISTRACTORS:-12}" \
  CAMERA_CANDIDATES="${CAMERA_CANDIDATES:-24}" \
  CAMERA_COMPOSITION_PROFILE="${CAMERA_COMPOSITION_PROFILE:-balanced}" \
  LIGHTING_POOL="${LIGHTING_POOL:-warm_daylight cool_daylight warm_indoor neutral_studio}" \
  RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-5400}" \
  ARCHIVE_AFTER_EACH=1 \
  CONTINUE_ON_ERROR=0 \
  bash scripts/generate_random_complex_series.sh 1
  status=$?

  if [[ "$status" -eq 0 ]]; then
    ((success += 1))
    echo "[POOL OK] background=$scene success=$success/$TARGET"
  else
    echo "[POOL WARN] background=$scene status=$status; trying another background" >&2
  fi
  (( success >= TARGET )) && break
done

echo "[POOL DONE] requested=$TARGET success=$success attempts=$attempt output=$REPO_DIR/$OUT_DIR"
if (( success != TARGET )); then
  exit 1
fi

mapfile -t archives < <(find "$OUT_DIR" -maxdepth 1 -type f -name '*.tar.gz' | sort)
python scripts/render_stereo_archive_video.py "${archives[@]}" \
  --output-dir "$VIDEO_DIR" \
  --camera camera00 \
  --fps "${VIDEO_FPS:-7.5}"

echo "[ALL DONE] datasets=$success videos=${#archives[@]} output=$REPO_DIR/$OUT_DIR"
