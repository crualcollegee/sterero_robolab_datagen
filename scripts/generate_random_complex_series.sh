#!/usr/bin/env bash
set -euo pipefail

# Generate N randomized complex RoboLab stereo trajectories.
# Usage:
#   bash scripts/generate_random_complex_series.sh 10
# Optional overrides:
#   STEPS=300 RECORD_EVERY=2 WIDTH=1280 HEIGHT=720 DISTRACTORS=12 bash scripts/generate_random_complex_series.sh 10

N="${1:-}"
if [[ -z "$N" ]]; then
  read -r -p "How many trajectories? n = " N
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -le 0 ]]; then
  echo "[ERROR] n must be a positive integer." >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-/autodl-fs/data/robolab}"
OUT_DIR="${OUT_DIR:-output/stereo_dataset/random_complex_series}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-directory}"
ARCHIVE_AFTER_EACH="${ARCHIVE_AFTER_EACH:-1}"
ARCHIVE_COMPRESSION_LEVEL="${ARCHIVE_COMPRESSION_LEVEL:-1}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
STEPS="${STEPS:-300}"
RECORD_EVERY="${RECORD_EVERY:-2}"
SKIP_INITIAL_SECONDS="${SKIP_INITIAL_SECONDS:-1.0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
BASELINE="${BASELINE:-0.16}"
WRIST_BASELINE="${WRIST_BASELINE:-0.04}"
POLICY_BACKEND="${POLICY_BACKEND:-pi05}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
DISTRACTORS="${DISTRACTORS:-12}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-real}"
DISTRACTOR_COLLIDER="${DISTRACTOR_COLLIDER:-boundingCube}"
ADD_GEOMETRY_BACKGROUND="${ADD_GEOMETRY_BACKGROUND:-1}"
GEOMETRY_BACKGROUND_STYLE="${GEOMETRY_BACKGROUND_STYLE:-random}"
GEOMETRY_BACKGROUND_SCALE="${GEOMETRY_BACKGROUND_SCALE:-1.0}"
OBJECT_XY_RANGE="${OBJECT_XY_RANGE:-0.04}"
OBJECT_YAW_RANGE="${OBJECT_YAW_RANGE:-0.8}"
TABLE_Z="${TABLE_Z:-0.08}"

TASKS=(
  ClutterPlasticTask
  ClearOrganicObjectsTask
  ElectronicsInBinTask
  BlackItemsInBinTask
  CondimentsInBinTask
  DishesInBinTask
  RubiksCubesInBinTask
  FruitsOnPlateTask
  GrabABagelTask
  GrabAFruitTask
  MoveBananaToBagelPlateTask
  UtensilsInMugTask
  YogurtInBowlTask
  FruitsMovingTask
  FruitsMovingOrangeOrLimeTask
  FruitsOnionToPlateTask
  FruitsOrangesOnPlateTask
  CookingClearPlateTask
  CookingPickPastaToolTask
  MarkerInMugTask
  MouseOnKeyboardTask
  PickGlassesTask
  ThrowAwaySnacksTask
  JugsOnShelfTask
  OneBottleOnShelfTask
  PlasticBottlesInSquarePailTask
  PutTwoMugsOnShelfTask
  TakeSpatulaOffShelfTask
  PutMugsOnShelfTask
  TakeMugsOffOfShelfTask
  ToolOrganizationTask
  ToolOrganizationBothTask
  ToolsPickingAllHammersTask
  ToolsPickingDrillTask
  FoodPackingByColorTask
  FoodPacking3BoxesTask
  FoodPacking3CansTask
  RecycleCartonTask
  RecycleCartonsVerticalCrateTask
  AnimalsInBinTask
  BlocksInBinTask
  CleanUpToysTask
  CubesAndBlocksInBinTask
  StackYellowOnRedTask
)


LIGHTS=(top left right front behind sphere dim)
GEOMETRY_BACKGROUND_STYLES=($(printf "bg%03d " {0..99}))

declare -A OBJECTS
OBJECTS[ClutterPlasticTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall red_onion whitepackerbottle_a01 avocado01 crabbypenholder milkjug_a01 serving_bowl utilityjug_a03"
OBJECTS[ClearOrganicObjectsTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall red_onion whitepackerbottle_a01 avocado01 crabbypenholder milkjug_a01 serving_bowl utilityjug_a03"
OBJECTS[ElectronicsInBinTask]="ceramic_mug glasses keyboard lizard_figurine marker remote_control smartphone wooden_bowl spoon_big computer_mouse yogurt_cup granola_bars"
OBJECTS[BlackItemsInBinTask]="ceramic_mug glasses keyboard lizard_figurine marker remote_control smartphone wooden_bowl spoon_big computer_mouse yogurt_cup granola_bars"
OBJECTS[CondimentsInBinTask]="mug mustard bowl ranch_dressing bbq_sauce_bottle oatmeal_raisin_cookies canned_tuna soft_scrub wood_block coffee_pot bbq_sauce_bottle_01"
OBJECTS[DishesInBinTask]="mug mug_01 bowl banana_near banana_far rubiks_cube_top rubiks_cube_middle rubiks_cube_bottom ketchup_bottle"
OBJECTS[RubiksCubesInBinTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"
OBJECTS[FruitsOnPlateTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_spoons spatula"
OBJECTS[GrabABagelTask]="bowl banana bagel_07 coffee_can banana_01 yogurt_cup coffee_pot ceramic_mug pitcher fork_big spoon_big apple_01 orange2 milk_carton orange_juice_carton bagel_01 bagel_02 plate_small"
OBJECTS[GrabAFruitTask]="bowl banana bagel_07 coffee_can banana_01 yogurt_cup coffee_pot ceramic_mug pitcher fork_big spoon_big apple_01 orange2 milk_carton orange_juice_carton bagel_01 bagel_02 plate_small"
OBJECTS[MoveBananaToBagelPlateTask]="bowl banana bagel_07 coffee_can banana_01 yogurt_cup coffee_pot ceramic_mug pitcher fork_big spoon_big apple_01 orange2 milk_carton orange_juice_carton bagel_01 bagel_02 plate_small"
OBJECTS[UtensilsInMugTask]="bowl banana bagel_07 coffee_can banana_01 yogurt_cup coffee_pot ceramic_mug pitcher fork_big spoon_big apple_01 orange2 milk_carton orange_juice_carton bagel_01 bagel_02 plate_small"
OBJECTS[YogurtInBowlTask]="bowl banana bagel_07 coffee_can banana_01 yogurt_cup coffee_pot ceramic_mug pitcher fork_big spoon_big apple_01 orange2 milk_carton orange_juice_carton bagel_01 bagel_02 plate_small"
OBJECTS[FruitsMovingTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_bowl wooden_spoons spatula"
OBJECTS[FruitsMovingOrangeOrLimeTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_bowl wooden_spoons spatula"
OBJECTS[FruitsOnionToPlateTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_bowl wooden_spoons spatula"
OBJECTS[FruitsGreenLimesOnPlateTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_spoons spatula"
OBJECTS[FruitsOrangesOnPlateTask]="lemon_01 lemon_02 lime01 lime01_01 orange_01 orange_02 pomegranate01 pumpkinlarge pumpkinsmall redonion serving_bowl clay_plates wooden_spoons spatula"
OBJECTS[CookingClearPlateTask]="redonion serving_bowl clay_plates wooden_spoons spatula tomato_sauce_can measuring_cups_1 pink_spaghetti_spoon spoon_1 green_serving_spoon storage_box_01 ladle wooden_bowl potato_masher"
OBJECTS[CookingPickPastaToolTask]="redonion serving_bowl clay_plates wooden_spoons spatula tomato_sauce_can measuring_cups_1 pink_spaghetti_spoon spoon_1 green_serving_spoon storage_box_01 ladle wooden_bowl potato_masher"
OBJECTS[MarkerInMugTask]="ceramic_mug glasses keyboard lizard_figurine marker remote_control rubiks_cube smartphone wooden_bowl spoon_big computer_mouse yogurt_cup oatmeal_raisin_cookies granola_bars"
OBJECTS[MouseOnKeyboardTask]="ceramic_mug glasses keyboard lizard_figurine marker remote_control rubiks_cube smartphone wooden_bowl spoon_big computer_mouse yogurt_cup oatmeal_raisin_cookies granola_bars"
OBJECTS[PickGlassesTask]="ceramic_mug glasses keyboard lizard_figurine marker remote_control rubiks_cube smartphone wooden_bowl spoon_big computer_mouse yogurt_cup oatmeal_raisin_cookies granola_bars"
OBJECTS[ThrowAwaySnacksTask]="ceramic_mug glasses keyboard marker remote_control smartphone wooden_bowl spoon_big computer_mouse yogurt_cup pitcher plasticpail_a02 apple_01"
OBJECTS[JugsOnShelfTask]="whitepackerbottle_a01 whitepackerbottle_a02 utilityjug_a01 utilityjug_a02 squarepail_a01 plasticpail_a02 whitepackerbottle_a03"
OBJECTS[OneBottleOnShelfTask]="whitepackerbottle_a01 whitepackerbottle_a02 utilityjug_a01 utilityjug_a02 squarepail_a01 plasticpail_a02 whitepackerbottle_a03"
OBJECTS[PlasticBottlesInSquarePailTask]="whitepackerbottle_a01 whitepackerbottle_a02 utilityjug_a01 utilityjug_a02 squarepail_a01 plasticpail_a02 whitepackerbottle_a03"
OBJECTS[PutTwoMugsOnShelfTask]="spatula_01 plate_small fork_big fork_small ceramic_mug mug mug_01"
OBJECTS[TakeSpatulaOffShelfTask]="spatula_01 plate_small fork_big fork_small ceramic_mug mug mug_01"
OBJECTS[PutMugsOnShelfTask]="ceramic_mug mug serving_bowl utilityjug_a01"
OBJECTS[TakeMugsOffOfShelfTask]="ceramic_mug mug serving_bowl utilityjug_a01"
OBJECTS[ToolOrganizationTask]="red_hammer husky_hammer cordless_drill spring_clamp"
OBJECTS[ToolOrganizationBothTask]="red_hammer husky_hammer cordless_drill spring_clamp"
OBJECTS[ToolsPickingAllHammersTask]="clamp cordless_drill spring_clamp husky_hammer blue_hammer red_hammer wood_hammer clamp_01"
OBJECTS[ToolsPickingDrillTask]="clamp cordless_drill spring_clamp husky_hammer blue_hammer red_hammer wood_hammer clamp_01"
OBJECTS[FoodPackingByColorTask]="cheez_it chocolate_pudding coffee_can mustard spam_can sugar_box tomato_soup_can"
OBJECTS[FoodPacking3BoxesTask]="cheez_it chocolate_pudding mustard spam_can sugar_box tomato_soup_can tuna_can"
OBJECTS[FoodPacking3CansTask]="cheez_it chocolate_pudding mustard spam_can sugar_box tomato_soup_can tuna_can"
OBJECTS[RecycleCartonTask]="milk_carton orange_juice_carton alphabet_soup_can smartphone mayonnaise_bottle ketchup_bottle mug"
OBJECTS[RecycleCartonsVerticalCrateTask]="milk_carton orange_juice_carton alphabet_soup_can smartphone mayonnaise_bottle ketchup_bottle mug"
OBJECTS[AnimalsInBinTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"
OBJECTS[BlocksInBinTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"
OBJECTS[CleanUpToysTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"
OBJECTS[CubesAndBlocksInBinTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"
OBJECTS[StackYellowOnRedTask]="rubiks_cube rubiks_cube_1 rubiks_cube_2 lizard_figurine birdhouse yellow_block red_block green_block blue_block lizard_figurine_01"


rand_int() {
  local min="$1"
  local max="$2"
  echo $(( min + RANDOM % (max - min + 1) ))
}

seed_for() {
  local i="$1"
  echo $(( ($(date +%s) + RANDOM * 13 + i * 9973) % 2147483647 ))
}

cd "$REPO_DIR"
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
if ! [[ "${OMP_NUM_THREADS:-8}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
else
  export OMP_NUM_THREADS
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$OUT_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[INFO] Generating $N randomized complex trajectories"
echo "[INFO] Output dir: $REPO_DIR/$OUT_DIR"
echo "[INFO] Output format: $OUTPUT_FORMAT"
echo "[INFO] Archive after each directory result: $ARCHIVE_AFTER_EACH"
echo "[INFO] Steps=$STEPS, record_every=$RECORD_EVERY, skip_initial_seconds=$SKIP_INITIAL_SECONDS, size=${WIDTH}x${HEIGHT}, distractors=$DISTRACTORS"
echo "[INFO] Policy backend: $POLICY_BACKEND at ${REMOTE_HOST}:${REMOTE_PORT}"
echo "[INFO] Continue on per-run error: ${CONTINUE_ON_ERROR:-1}"
echo "[INFO] Per-run timeout seconds: $RUN_TIMEOUT_SECONDS"

success_count=0
fail_count=0

for ((idx = 0; idx < N; idx++)); do
  task="${TASKS[$(rand_int 0 $((${#TASKS[@]} - 1)))]}"
  lighting="${LIGHTS[$(rand_int 0 $((${#LIGHTS[@]} - 1)))]}"
  if [[ "$GEOMETRY_BACKGROUND_STYLE" == "random" ]]; then
    geometry_background_style="${GEOMETRY_BACKGROUND_STYLES[$(rand_int 0 $((${#GEOMETRY_BACKGROUND_STYLES[@]} - 1)))]}"
  else
    geometry_background_style="$GEOMETRY_BACKGROUND_STYLE"
  fi
  base_seed="$(seed_for "$idx")"
  camera_seed="$base_seed"
  background_seed="$((base_seed + 101))"
  distractor_seed="$((base_seed + 202))"
  geometry_background_seed="$((base_seed + 303))"
  file_name="${RUN_STAMP}_$(printf '%05d' "$idx")_${task}_seed${base_seed}.hdf5"

  echo
  echo "[RUN $((idx + 1))/$N] task=$task lighting=$lighting geometry_background=$geometry_background_style seed=$base_seed file=$file_name"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] task=$task lighting=$lighting geometry_background=$geometry_background_style camera_seed=$camera_seed background_seed=$background_seed distractor_seed=$distractor_seed"
    echo "[DRY_RUN] objects=${OBJECTS[$task]}"
    success_count=$((success_count + 1))
    continue
  fi

  geometry_background_args=()
  if [[ "$ADD_GEOMETRY_BACKGROUND" == "1" ]]; then
    geometry_background_args=(
      --add-geometry-background
      --geometry-background-seed "$geometry_background_seed"
      --geometry-background-style "$geometry_background_style"
      --geometry-background-scale "$GEOMETRY_BACKGROUND_SCALE"
    )
  fi

  set +e
  timeout --kill-after=60s "$RUN_TIMEOUT_SECONDS" uv run --no-sync python examples/generate_stereo_dataset.py \
    --task "$task" \
    --headless \
    --num-steps "$STEPS" \
    --record-every "$RECORD_EVERY" \
    --skip-initial-seconds "$SKIP_INITIAL_SECONDS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --baseline "$BASELINE" \
    --include-wrist-stereo-camera \
    --wrist-baseline "$WRIST_BASELINE" \
    --exclude-policy-wrist-camera-from-output \
    --policy-backend "$POLICY_BACKEND" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    --moving-camera-pose \
    --camera-motion-mode table_full_random \
    --camera-orbit-jitter-seed "$camera_seed" \
    --camera-random-keyframes 5 \
    --camera-random-start-table \
    --camera-random-start-pos 0.70 0.0 0.95 \
    --camera-random-start-look-at 0.45 0.0 0.72 \
    --camera-table-bounds-min 0.05 -0.55 0.50 \
    --camera-table-bounds-max 0.90 0.55 0.95 \
    --camera-table-full-margin 0.04 \
    --camera-table-full-vertical-fov-deg 75 \
    --camera-table-full-azimuth-range -75 75 \
    --camera-table-full-elevation-range 28 62 \
    --camera-table-full-distance-range 0.55 0.95 \
    --camera-table-full-look-at-jitter 0.10 \
    --camera-table-full-max-retreat 1.50 \
    --camera-table-full-distance-padding 0.04 \
    --camera-table-full-distance-change-per-step 0.008 \
    --randomize-background \
    --background-seed "$background_seed" \
    "${geometry_background_args[@]}" \
    --lighting "$lighting" \
    --randomize-object-pose \
    --randomize-objects ${OBJECTS[$task]} \
    --object-xy-range "$OBJECT_XY_RANGE" \
    --object-yaw-range "$OBJECT_YAW_RANGE" \
    --add-random-distractors \
    --distractor-mode "$DISTRACTOR_MODE" \
    --num-distractors "$DISTRACTORS" \
    --distractor-seed "$distractor_seed" \
    --distractor-table-z "$TABLE_Z" \
    --distractor-x-range 0.18 0.82 \
    --distractor-y-range -0.42 0.42 \
    --distractor-scale-range 0.85 1.15 \
    --distractor-mass-range 0.05 0.35 \
    --distractor-collider-approx "$DISTRACTOR_COLLIDER" \
    --output-dir "$OUT_DIR" \
    --file-name "$file_name" \
    --output-format "$OUTPUT_FORMAT" \
    --overwrite
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
      echo "[ERROR] Run $((idx + 1))/$N timed out after ${RUN_TIMEOUT_SECONDS}s: task=$task file=$file_name" >&2
    else
      echo "[ERROR] Run $((idx + 1))/$N failed with status $status: task=$task file=$file_name" >&2
    fi
    if [[ "${CONTINUE_ON_ERROR:-1}" == "1" ]]; then
      echo "[INFO] Continuing to next trajectory because CONTINUE_ON_ERROR=1" >&2
      continue
    fi
    exit "$status"
  fi

  scene_name="${file_name%.hdf5}"
  if [[ "$OUTPUT_FORMAT" == "directory" ]]; then
    success_path="$OUT_DIR/$scene_name/camera00/.complete"
  elif [[ "$OUTPUT_FORMAT" == "both" ]]; then
    success_path="$OUT_DIR/$scene_name/camera00/.complete"
  else
    success_path="$OUT_DIR/$file_name"
  fi

  if [[ -e "$success_path" ]]; then
    if [[ "$ARCHIVE_AFTER_EACH" == "1" && ( "$OUTPUT_FORMAT" == "directory" || "$OUTPUT_FORMAT" == "both" ) ]]; then
      scene_dir="$OUT_DIR/$scene_name"
      archive_path="$OUT_DIR/$scene_name.tar.gz"
      echo "[INFO] Archiving $scene_dir -> $archive_path"
      if command -v pigz >/dev/null 2>&1; then
        tar -C "$OUT_DIR" -I "pigz -$ARCHIVE_COMPRESSION_LEVEL" -cf "$archive_path" "$scene_name"
      else
        GZIP="-$ARCHIVE_COMPRESSION_LEVEL" tar -C "$OUT_DIR" -czf "$archive_path" "$scene_name"
      fi
      if [[ ! -s "$archive_path" ]]; then
        fail_count=$((fail_count + 1))
        echo "[ERROR] Archive failed or empty: $archive_path" >&2
        if [[ "${CONTINUE_ON_ERROR:-1}" != "1" ]]; then
          exit 1
        fi
        continue
      fi
      if ! tar -tzf "$archive_path" "$scene_name/camera00/.complete" >/dev/null 2>&1; then
        fail_count=$((fail_count + 1))
        echo "[ERROR] Archive validation failed, missing camera00/.complete: $archive_path" >&2
        if [[ "${CONTINUE_ON_ERROR:-1}" != "1" ]]; then
          exit 1
        fi
        continue
      fi
      rm -rf "$scene_dir"
      success_count=$((success_count + 1))
      echo "[OK] Wrote archive $archive_path and removed $scene_dir"
    else
      success_count=$((success_count + 1))
      echo "[OK] Wrote $success_path"
    fi
  else
    fail_count=$((fail_count + 1))
    echo "[ERROR] Run $((idx + 1))/$N exited 0 but output is missing: $success_path" >&2
    if [[ "${CONTINUE_ON_ERROR:-1}" != "1" ]]; then
      exit 1
    fi
  fi

done

echo
echo "[DONE] Requested=$N success=$success_count failed=$fail_count output=$REPO_DIR/$OUT_DIR"
if [[ "$success_count" -eq 0 && "${DRY_RUN:-0}" != "1" ]]; then
  exit 1
fi
