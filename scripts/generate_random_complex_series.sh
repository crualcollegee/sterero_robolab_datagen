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
if [[ "${OUTPUT_FORMAT:-directory}" != "directory" || "${ARCHIVE_AFTER_EACH:-1}" != "1" ]]; then
  echo "[WARN] Production output is fixed to validated tar.gz; OUTPUT_FORMAT and ARCHIVE_AFTER_EACH overrides are ignored." >&2
fi
OUTPUT_FORMAT="directory"
ARCHIVE_AFTER_EACH="1"
ARCHIVE_COMPRESSION_LEVEL="${ARCHIVE_COMPRESSION_LEVEL:-1}"
FIVE_CAMERA_TRAJECTORY="${FIVE_CAMERA_TRAJECTORY:-0}"
CAMERA03_ONLY="${CAMERA03_ONLY:-0}"
if [[ "$FIVE_CAMERA_TRAJECTORY" == "1" && "$CAMERA03_ONLY" == "1" ]]; then
  echo "[ERROR] FIVE_CAMERA_TRAJECTORY and CAMERA03_ONLY are mutually exclusive" >&2
  exit 1
fi
if [[ "$FIVE_CAMERA_TRAJECTORY" == "1" || "$CAMERA03_ONLY" == "1" ]]; then
  RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-18000}"
else
  RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-5400}"
fi
STEPS="${STEPS:-300}"
RECORD_EVERY="${RECORD_EVERY:-2}"
SKIP_INITIAL_SECONDS="${SKIP_INITIAL_SECONDS:-1.0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
BASELINE="${BASELINE:-0.16}"
WRIST_BASELINE="${WRIST_BASELINE:-0.05}"
MULTI_CAMERA_SEED_STRIDE="${MULTI_CAMERA_SEED_STRIDE:-1000003}"
FRONT_CAMERA_POSITION="${FRONT_CAMERA_POSITION:-${REAR_CAMERA_POSITION:-0.82 0.0 0.75}}"
FRONT_CAMERA_LOOK_AT="${FRONT_CAMERA_LOOK_AT:-${REAR_CAMERA_LOOK_AT:-0.55 0.0 0.40}}"
FRONT_CAMERA_MOTION="${FRONT_CAMERA_MOTION:-1}"
FRONT_CAMERA_POSITION_JITTER="${FRONT_CAMERA_POSITION_JITTER:-0.080 0.140 0.100}"
FRONT_CAMERA_LOOK_AT_JITTER="${FRONT_CAMERA_LOOK_AT_JITTER:-0.035 0.065 0.040}"
FRONT_CAMERA_ACTION_FOLLOW_GAIN="${FRONT_CAMERA_ACTION_FOLLOW_GAIN:-0.12}"
FRONT_CAMERA_VERTICAL_PITCH_COUPLING="${FRONT_CAMERA_VERTICAL_PITCH_COUPLING:-0.35}"
FRONT_CAMERA_DISTANCE_RANGE="${FRONT_CAMERA_DISTANCE_RANGE:-0.32 0.72}"
FRONT_CAMERA_ELEVATION_RANGE_DEG="${FRONT_CAMERA_ELEVATION_RANGE_DEG:-8 78}"
FRONT_CAMERA_AZIMUTH_RANGE_DEG="${FRONT_CAMERA_AZIMUTH_RANGE_DEG:--38 38}"
FRONT_CAMERA_TARGET_HEIGHT="${FRONT_CAMERA_TARGET_HEIGHT:-0.08}"
FRONT_CAMERA_MIN_ARM_DISTANCE="${FRONT_CAMERA_MIN_ARM_DISTANCE:-0.45}"
FRONT_CAMERA_SMOOTHING_WINDOW="${FRONT_CAMERA_SMOOTHING_WINDOW:-31}"
FRONT_CAMERA_OVERHEAD_ELEVATION_THRESHOLD_DEG="${FRONT_CAMERA_OVERHEAD_ELEVATION_THRESHOLD_DEG:-68}"
FRONT_CAMERA_OVERHEAD_TRANSITION_DEG="${FRONT_CAMERA_OVERHEAD_TRANSITION_DEG:-20}"
FRONT_CAMERA_OVERHEAD_MIN_DISTANCE="${FRONT_CAMERA_OVERHEAD_MIN_DISTANCE:-0.55}"
FRONT_CAMERA_RANDOM_KEYFRAMES="${FRONT_CAMERA_RANDOM_KEYFRAMES:-6}"
TASK_OVERRIDE="${TASK_OVERRIDE:-}"
TASK_PAIR_MODE="${TASK_PAIR_MODE:-0}"
PAIR_SCENE_SEED="${PAIR_SCENE_SEED:-20260827}"
POLICY_BACKEND="${POLICY_BACKEND:-pi05}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
DISTRACTORS="${DISTRACTORS:-12}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-real}"
DISTRACTOR_COLLIDER="${DISTRACTOR_COLLIDER:-boundingCube}"
DISTRACTOR_PLACEMENT_CLEARANCE="${DISTRACTOR_PLACEMENT_CLEARANCE:-0.03}"
DISTRACTOR_PLACEMENT_ATTEMPTS="${DISTRACTOR_PLACEMENT_ATTEMPTS:-64}"
DISTRACTOR_SETTLE_STEPS="${DISTRACTOR_SETTLE_STEPS:-15}"
DISTRACTOR_MAX_SETTLE_MOTION="${DISTRACTOR_MAX_SETTLE_MOTION:-0.12}"
INJECTED_SCENE_STABILITY_GATE="${INJECTED_SCENE_STABILITY_GATE:-1}"
POLICY_INPUT_PREVIEW="${POLICY_INPUT_PREVIEW:-0}"
POLICY_PILOT_ONLY="${POLICY_PILOT_ONLY:-0}"
ADD_GEOMETRY_BACKGROUND="${ADD_GEOMETRY_BACKGROUND:-1}"
GEOMETRY_BACKGROUND_STYLE="${GEOMETRY_BACKGROUND_STYLE:-random}"
# Geometry-depth production uses the local InteriorAgent room archive pool by
# default. Override this explicitly when testing RoboLab or Isaac presets.
GEOMETRY_BACKGROUND_SOURCE="${GEOMETRY_BACKGROUND_SOURCE:-interioragent}"
GEOMETRY_BACKGROUND_SCALE="${GEOMETRY_BACKGROUND_SCALE:-1.0}"
GEOMETRY_BACKGROUND_SHELL="${GEOMETRY_BACKGROUND_SHELL:-on}"
GEOMETRY_BACKGROUND_PLACEMENT="${GEOMETRY_BACKGROUND_PLACEMENT:-surround}"
GEOMETRY_BACKGROUND_SAFETY="${GEOMETRY_BACKGROUND_SAFETY:-1}"
GEOMETRY_BACKGROUND_LOS_CLEARANCE="${GEOMETRY_BACKGROUND_LOS_CLEARANCE:-0.08}"
GEOMETRY_BACKGROUND_LOS_SAMPLES="${GEOMETRY_BACKGROUND_LOS_SAMPLES:-300}"
GEOMETRY_BACKGROUND_RANDOMIZE_POSITION="${GEOMETRY_BACKGROUND_RANDOMIZE_POSITION:-1}"
GEOMETRY_BACKGROUND_POSITION_CANDIDATES="${GEOMETRY_BACKGROUND_POSITION_CANDIDATES:-64}"
GEOMETRY_BACKGROUND_MAX_POSITION_SHIFT="${GEOMETRY_BACKGROUND_MAX_POSITION_SHIFT:-2.0}"
# RoboLab's HDR/EXR DomeLight backgrounds are RGB-only and do not provide depth.
# Keep them off by default for geometry-depth datasets.
RANDOMIZE_HDR_BACKGROUND="${RANDOMIZE_HDR_BACKGROUND:-0}"
OBJECT_XY_RANGE="${OBJECT_XY_RANGE:-0.04}"
OBJECT_YAW_RANGE="${OBJECT_YAW_RANGE:-0.8}"
TABLE_Z="${TABLE_Z:-0.08}"
CAMERA_CANDIDATES="${CAMERA_CANDIDATES:-16}"
CAMERA_COMPOSITION_PROFILE="${CAMERA_COMPOSITION_PROFILE:-balanced}"
CAMERA_MAX_TRANSLATION_STEP="${CAMERA_MAX_TRANSLATION_STEP:-0.03}"
CAMERA_MAX_TRANSLATION_ACCEL="${CAMERA_MAX_TRANSLATION_ACCEL:-0.01}"
CAMERA_MAX_ROTATION_STEP_DEG="${CAMERA_MAX_ROTATION_STEP_DEG:-2.5}"
CAMERA_MAX_ROTATION_ACCEL_DEG="${CAMERA_MAX_ROTATION_ACCEL_DEG:-0.8}"
CAMERA_MAX_FOCUS_STEP="${CAMERA_MAX_FOCUS_STEP:-0.025}"
CAMERA_MAX_FOCUS_ACCEL="${CAMERA_MAX_FOCUS_ACCEL:-0.01}"
CAMERA_DISTANCE_CHANGE_PER_STEP="${CAMERA_DISTANCE_CHANGE_PER_STEP:-0.012}"
MIN_POLICY_EE_MOTION="${MIN_POLICY_EE_MOTION:-0.05}"
POLICY_INTERACTION_GATE="${POLICY_INTERACTION_GATE:-0}"
MIN_POLICY_CLOSED_CONTACT_STEPS="${MIN_POLICY_CLOSED_CONTACT_STEPS:-1}"
CAMERA_GEOMETRY_GATE="${CAMERA_GEOMETRY_GATE:-1}"
CAMERA_SCENE_CLEARANCE="${CAMERA_SCENE_CLEARANCE:-0.06}"
CAMERA_ROBOT_CLEARANCE="${CAMERA_ROBOT_CLEARANCE:-0.02}"
CAMERA_VISUAL_QUALITY_GATE="${CAMERA_VISUAL_QUALITY_GATE:-1}"
CAMERA_MAX_WEAK_BACKGROUND_RATIO="${CAMERA_MAX_WEAK_BACKGROUND_RATIO:-0.70}"
CAMERA_MIN_TEXTURE_SCORE="${CAMERA_MIN_TEXTURE_SCORE:-0.07}"
CAMERA_MAX_BAD_FRAME_RATIO="${CAMERA_MAX_BAD_FRAME_RATIO:-0.10}"
CAMERA_DEPTH_COVERAGE_GATE="${CAMERA_DEPTH_COVERAGE_GATE:-1}"
CAMERA_MIN_DEPTH_VALID_RATIO="${CAMERA_MIN_DEPTH_VALID_RATIO:-0.90}"
CAMERA_MAX_DEPTH_BAD_FRAME_RATIO="${CAMERA_MAX_DEPTH_BAD_FRAME_RATIO:-0.10}"

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


# Natural two-light setups retain material color while still varying color
# temperature. Override with LIGHTING_POOL="top left ..." when needed.
read -r -a LIGHTS <<< "${LIGHTING_POOL:-warm_daylight cool_daylight warm_indoor neutral_studio sphere}"
ROBOLAB_GEOMETRY_BACKGROUND_STYLES=(
  asset_kitchen_single_wall
  asset_kitchen_l_shaped
  asset_kitchen_island
  asset_kitchen_peninsula
  asset_kitchen_u_shaped
  asset_workdesk
  asset_workdesk_bin
  asset_workdesk_snacks
  asset_shelf_cleaning
  asset_shelf_cleaning2
  asset_shelf_mugs
  asset_wire_shelf
  asset_tools_container
  asset_tools_picking
  asset_clutter_shelf
  asset_front_shelf
  asset_cutlery_shelf
  asset_food_packing_dense
)

# Default Isaac background list excludes grid rooms, simple rooms, and bare terrains.
# Those are official Isaac USD assets, but they are too simple for reconstruction data.
ISAAC_GEOMETRY_BACKGROUND_STYLES=(
  isaac_warehouse
  isaac_warehouse_shelves
  isaac_warehouse_forklifts
  isaac_warehouse_extras
  isaac_hospital
  isaac_digital_twin_warehouse
  isaac_simple_room
  isaac_gridroom_black
  isaac_gridroom_curved
  isaac_jetracer_track
  isaac_rivermark
  isaac_terrain_flat
  isaac_terrain_rough
  isaac_terrain_slope
  isaac_terrain_stairs
)

ISAAC_ALL_GEOMETRY_BACKGROUND_STYLES=(
  isaac_grid_default
  isaac_gridroom_black
  isaac_gridroom_curved
  isaac_simple_room
  isaac_warehouse
  isaac_warehouse_shelves
  isaac_warehouse_forklifts
  isaac_warehouse_extras
  isaac_full_warehouse
  isaac_office
  isaac_hospital
  isaac_digital_twin_warehouse
  isaac_rivermark
  isaac_jetracer_track
  isaac_terrain_flat
  isaac_terrain_rough
  isaac_terrain_slope
  isaac_terrain_stairs
)

INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES=()
INTERIORAGENT_ROOT="${INTERIORAGENT_ROOT:-/autodl-fs/data/spatialverse/InteriorAgent}"
INTERIORAGENT_ARCHIVE_ROOT="${INTERIORAGENT_ARCHIVE_ROOT:-/autodl-fs/data/spatialverse/InteriorAgent_archives}"
if [[ -d "$INTERIORAGENT_ROOT" ]]; then
  while IFS= read -r scene_dir; do
    scene_name="$(basename "$scene_dir")"
    if [[ -f "$scene_dir/${scene_name}.usda" ]] && ! head -c 80 "$scene_dir/${scene_name}.usda" | grep -q 'version https://git-lfs.github.com/spec'; then
      INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES+=("interioragent_${scene_name}")
    fi
  done < <(find "$INTERIORAGENT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'kujiale_*' | sort)
fi
if [[ -d "$INTERIORAGENT_ARCHIVE_ROOT" ]]; then
  while IFS= read -r archive_path; do
    scene_name="$(basename "$archive_path" .tar.gz)"
    style="interioragent_${scene_name}"
    if [[ ! " ${INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES[*]} " =~ " ${style} " ]]; then
      INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES+=("$style")
    fi
  done < <(find "$INTERIORAGENT_ARCHIVE_ROOT" -mindepth 1 -maxdepth 1 -type f -name 'kujiale_*.tar.gz' | sort)
fi

case "$GEOMETRY_BACKGROUND_SOURCE" in
  all)
    GEOMETRY_BACKGROUND_STYLES=("${ROBOLAB_GEOMETRY_BACKGROUND_STYLES[@]}" "${ISAAC_GEOMETRY_BACKGROUND_STYLES[@]}" "${INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES[@]}")
    ;;
  robolab)
    GEOMETRY_BACKGROUND_STYLES=("${ROBOLAB_GEOMETRY_BACKGROUND_STYLES[@]}")
    ;;
  isaac)
    GEOMETRY_BACKGROUND_STYLES=("${ISAAC_GEOMETRY_BACKGROUND_STYLES[@]}")
    ;;
  isaac_all)
    GEOMETRY_BACKGROUND_STYLES=("${ISAAC_ALL_GEOMETRY_BACKGROUND_STYLES[@]}")
    ;;
  interioragent)
    GEOMETRY_BACKGROUND_STYLES=("${INTERIORAGENT_GEOMETRY_BACKGROUND_STYLES[@]}")
    ;;
  *)
    echo "[ERROR] GEOMETRY_BACKGROUND_SOURCE must be one of: all, robolab, isaac, isaac_all, interioragent (got '$GEOMETRY_BACKGROUND_SOURCE')" >&2
    exit 1
    ;;
esac
if [[ "$GEOMETRY_BACKGROUND_STYLE" == "random" && "${#GEOMETRY_BACKGROUND_STYLES[@]}" -eq 0 ]]; then
  echo "[ERROR] No geometry backgrounds found for GEOMETRY_BACKGROUND_SOURCE=$GEOMETRY_BACKGROUND_SOURCE" >&2
  exit 1
fi

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
echo "[INFO] Geometry background source: $GEOMETRY_BACKGROUND_SOURCE"
echo "[INFO] Geometry background pool size: ${#GEOMETRY_BACKGROUND_STYLES[@]}"
echo "[INFO] Continue on per-run error: ${CONTINUE_ON_ERROR:-1}"
echo "[INFO] Per-run timeout seconds: $RUN_TIMEOUT_SECONDS"
echo "[INFO] Five-camera shared trajectory: $FIVE_CAMERA_TRAJECTORY"
echo "[INFO] Camera03-only validation mode: $CAMERA03_ONLY"
echo "[INFO] Distractor placement clearance: ${DISTRACTOR_PLACEMENT_CLEARANCE}m; settle steps: $DISTRACTOR_SETTLE_STEPS"
echo "[INFO] Save exact policy RGB previews: $POLICY_INPUT_PREVIEW"
echo "[INFO] Policy pilot-only diagnostics: $POLICY_PILOT_ONLY"
echo "[INFO] Two-distinct-scenes-per-task schedule: $TASK_PAIR_MODE"

if [[ "$TASK_PAIR_MODE" == "1" ]]; then
  if [[ "$FIVE_CAMERA_TRAJECTORY" != "1" ]]; then
    echo "[ERROR] TASK_PAIR_MODE=1 requires FIVE_CAMERA_TRAJECTORY=1" >&2
    exit 1
  fi
  if [[ "$GEOMETRY_BACKGROUND_STYLE" != "random" ]]; then
    echo "[ERROR] TASK_PAIR_MODE=1 requires GEOMETRY_BACKGROUND_STYLE=random" >&2
    exit 1
  fi
  if [[ "${#GEOMETRY_BACKGROUND_STYLES[@]}" -lt 2 ]]; then
    echo "[ERROR] TASK_PAIR_MODE=1 requires at least two geometry backgrounds" >&2
    exit 1
  fi
  if [[ -n "$TASK_OVERRIDE" ]]; then
    if [[ "$N" -ne 2 ]]; then
      echo "[ERROR] TASK_PAIR_MODE=1 with TASK_OVERRIDE requires N=2" >&2
      exit 1
    fi
  elif [[ "$N" -ne $((2 * ${#TASKS[@]})) ]]; then
    echo "[ERROR] Full TASK_PAIR_MODE=1 requires N=$((2 * ${#TASKS[@]}))" >&2
    exit 1
  fi
fi

success_count=0
fail_count=0

for ((idx = 0; idx < N; idx++)); do
  if [[ -n "$TASK_OVERRIDE" ]]; then
    task="$TASK_OVERRIDE"
    if [[ -z "${OBJECTS[$task]+x}" ]]; then
      echo "[ERROR] Unknown TASK_OVERRIDE=$task" >&2
      exit 1
    fi
  elif [[ "$TASK_PAIR_MODE" == "1" ]]; then
    task="${TASKS[$((idx / 2))]}"
  else
    task="${TASKS[$(rand_int 0 $((${#TASKS[@]} - 1)))]}"
  fi
  lighting="${LIGHTS[$(rand_int 0 $((${#LIGHTS[@]} - 1)))]}"
  if [[ "$GEOMETRY_BACKGROUND_STYLE" == "random" ]]; then
    if [[ "$TASK_PAIR_MODE" == "1" ]]; then
      task_schedule_index=$((idx / 2))
      scene_slot=$((idx % 2))
      background_count=${#GEOMETRY_BACKGROUND_STYLES[@]}
      first_background=$(((PAIR_SCENE_SEED + task_schedule_index * 7919) % background_count))
      if [[ "$scene_slot" -eq 0 ]]; then
        background_index=$first_background
      else
        background_index=$(((first_background + 1 + task_schedule_index * 17) % background_count))
        if [[ "$background_index" -eq "$first_background" ]]; then
          background_index=$(((background_index + 1) % background_count))
        fi
      fi
      geometry_background_style="${GEOMETRY_BACKGROUND_STYLES[$background_index]}"
    else
      geometry_background_style="${GEOMETRY_BACKGROUND_STYLES[$(rand_int 0 $((${#GEOMETRY_BACKGROUND_STYLES[@]} - 1)))]}"
    fi
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
      --geometry-background-shell "$GEOMETRY_BACKGROUND_SHELL"
      --geometry-background-placement "$GEOMETRY_BACKGROUND_PLACEMENT"
      --geometry-background-los-clearance "$GEOMETRY_BACKGROUND_LOS_CLEARANCE"
      --geometry-background-los-samples "$GEOMETRY_BACKGROUND_LOS_SAMPLES"
      --geometry-background-position-candidates "$GEOMETRY_BACKGROUND_POSITION_CANDIDATES"
      --geometry-background-max-position-shift "$GEOMETRY_BACKGROUND_MAX_POSITION_SHIFT"
    )
    if [[ "$GEOMETRY_BACKGROUND_SAFETY" != "1" ]]; then
      geometry_background_args+=(--no-geometry-background-safety)
    fi
    if [[ "$GEOMETRY_BACKGROUND_RANDOMIZE_POSITION" != "1" ]]; then
      geometry_background_args+=(--no-geometry-background-randomize-position)
    fi
  fi

  camera_geometry_gate_args=(--camera-geometry-gate)
  if [[ "$CAMERA_GEOMETRY_GATE" != "1" ]]; then
    camera_geometry_gate_args=(--no-camera-geometry-gate)
  fi
  camera_visual_gate_args=(--camera-visual-quality-gate)
  if [[ "$CAMERA_VISUAL_QUALITY_GATE" != "1" ]]; then
    camera_visual_gate_args=(--no-camera-visual-quality-gate)
  fi
  camera_depth_gate_args=(--camera-depth-coverage-gate)
  if [[ "$CAMERA_DEPTH_COVERAGE_GATE" != "1" ]]; then
    camera_depth_gate_args=(--no-camera-depth-coverage-gate)
  fi
  policy_interaction_gate_args=(--policy-interaction-gate)
  if [[ "$POLICY_INTERACTION_GATE" != "1" ]]; then
    policy_interaction_gate_args=(--no-policy-interaction-gate)
  fi
  injected_scene_stability_gate_args=(--injected-scene-stability-gate)
  if [[ "$INJECTED_SCENE_STABILITY_GATE" != "1" ]]; then
    injected_scene_stability_gate_args=(--no-injected-scene-stability-gate)
  fi
  five_camera_args=()
  if [[ "$FIVE_CAMERA_TRAJECTORY" == "1" || "$CAMERA03_ONLY" == "1" ]]; then
    read -r -a front_camera_position <<< "$FRONT_CAMERA_POSITION"
    read -r -a front_camera_look_at <<< "$FRONT_CAMERA_LOOK_AT"
    read -r -a front_camera_position_jitter <<< "$FRONT_CAMERA_POSITION_JITTER"
    read -r -a front_camera_look_at_jitter <<< "$FRONT_CAMERA_LOOK_AT_JITTER"
    read -r -a front_camera_distance_range <<< "$FRONT_CAMERA_DISTANCE_RANGE"
    read -r -a front_camera_elevation_range <<< "$FRONT_CAMERA_ELEVATION_RANGE_DEG"
    read -r -a front_camera_azimuth_range <<< "$FRONT_CAMERA_AZIMUTH_RANGE_DEG"
    if [[ "${#front_camera_position[@]}" -ne 3 || "${#front_camera_look_at[@]}" -ne 3 || "${#front_camera_position_jitter[@]}" -ne 3 || "${#front_camera_look_at_jitter[@]}" -ne 3 ]]; then
      echo "[ERROR] FRONT_CAMERA_POSITION, FRONT_CAMERA_LOOK_AT, and their JITTER values require three values" >&2
      exit 1
    fi
    five_camera_args=(
      --multi-camera-seed-stride "$MULTI_CAMERA_SEED_STRIDE"
      --front-camera-position "${front_camera_position[@]}"
      --front-camera-look-at "${front_camera_look_at[@]}"
      --front-camera-position-jitter "${front_camera_position_jitter[@]}"
      --front-camera-look-at-jitter "${front_camera_look_at_jitter[@]}"
      --front-camera-action-follow-gain "$FRONT_CAMERA_ACTION_FOLLOW_GAIN"
      --front-camera-vertical-pitch-coupling "$FRONT_CAMERA_VERTICAL_PITCH_COUPLING"
      --front-camera-distance-range "${front_camera_distance_range[@]}"
      --front-camera-elevation-range-deg "${front_camera_elevation_range[@]}"
      --front-camera-azimuth-range-deg "${front_camera_azimuth_range[@]}"
      --front-camera-target-height "$FRONT_CAMERA_TARGET_HEIGHT"
      --front-camera-min-arm-distance "$FRONT_CAMERA_MIN_ARM_DISTANCE"
      --front-camera-smoothing-window "$FRONT_CAMERA_SMOOTHING_WINDOW"
      --front-camera-overhead-elevation-threshold-deg "$FRONT_CAMERA_OVERHEAD_ELEVATION_THRESHOLD_DEG"
      --front-camera-overhead-transition-deg "$FRONT_CAMERA_OVERHEAD_TRANSITION_DEG"
      --front-camera-overhead-min-distance "$FRONT_CAMERA_OVERHEAD_MIN_DISTANCE"
      --front-camera-random-keyframes "$FRONT_CAMERA_RANDOM_KEYFRAMES"
    )
    if [[ "$FIVE_CAMERA_TRAJECTORY" == "1" ]]; then
      five_camera_args+=(--five-camera-trajectory --wrist-baseline "$WRIST_BASELINE")
    else
      five_camera_args+=(--camera03-only)
    fi
    if [[ "$FRONT_CAMERA_MOTION" != "1" ]]; then
      five_camera_args+=(--no-front-camera-motion)
    fi
  fi

  hdr_background_args=()
  if [[ "$RANDOMIZE_HDR_BACKGROUND" == "1" ]]; then
    hdr_background_args=(
      --randomize-background
      --background-seed "$background_seed"
    )
  fi

  policy_preview_args=()
  policy_pilot_args=()
  if [[ "$POLICY_PILOT_ONLY" == "1" ]]; then
    policy_pilot_args=(--policy-pilot-only)
  fi
  if [[ "$POLICY_INPUT_PREVIEW" == "1" ]]; then
    policy_preview_args=(
      --policy-input-preview-dir "$OUT_DIR/policy_input_preview/${file_name%.hdf5}"
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
    "${five_camera_args[@]}" \
    --policy-backend "$POLICY_BACKEND" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    --moving-camera-pose \
    --camera-motion-mode table_full_random \
    --camera-orbit-jitter-seed "$camera_seed" \
    --camera-candidate-count "$CAMERA_CANDIDATES" \
    --camera-composition-profile "$CAMERA_COMPOSITION_PROFILE" \
    "${camera_geometry_gate_args[@]}" \
    --camera-scene-clearance "$CAMERA_SCENE_CLEARANCE" \
    --camera-robot-clearance "$CAMERA_ROBOT_CLEARANCE" \
    "${camera_visual_gate_args[@]}" \
    --camera-max-weak-background-ratio "$CAMERA_MAX_WEAK_BACKGROUND_RATIO" \
    --camera-min-texture-score "$CAMERA_MIN_TEXTURE_SCORE" \
    --camera-max-bad-frame-ratio "$CAMERA_MAX_BAD_FRAME_RATIO" \
    "${camera_depth_gate_args[@]}" \
    --camera-min-depth-valid-ratio "$CAMERA_MIN_DEPTH_VALID_RATIO" \
    --camera-max-depth-bad-frame-ratio "$CAMERA_MAX_DEPTH_BAD_FRAME_RATIO" \
    --camera-max-translation-step "$CAMERA_MAX_TRANSLATION_STEP" \
    --camera-max-translation-accel "$CAMERA_MAX_TRANSLATION_ACCEL" \
    --camera-max-rotation-step-deg "$CAMERA_MAX_ROTATION_STEP_DEG" \
    --camera-max-rotation-accel-deg "$CAMERA_MAX_ROTATION_ACCEL_DEG" \
    --camera-max-focus-step "$CAMERA_MAX_FOCUS_STEP" \
    --camera-max-focus-accel "$CAMERA_MAX_FOCUS_ACCEL" \
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
    --camera-table-full-distance-range 0.55 1.35 \
    --camera-dynamic-context \
    --min-policy-ee-motion "$MIN_POLICY_EE_MOTION" \
    "${policy_interaction_gate_args[@]}" \
    --min-policy-closed-contact-steps "$MIN_POLICY_CLOSED_CONTACT_STEPS" \
    --camera-table-full-look-at-jitter 0.10 \
    --camera-table-full-max-retreat 1.50 \
    --camera-table-full-distance-padding 0.04 \
    --camera-table-full-distance-change-per-step "$CAMERA_DISTANCE_CHANGE_PER_STEP" \
    "${hdr_background_args[@]}" \
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
    --distractor-placement-clearance "$DISTRACTOR_PLACEMENT_CLEARANCE" \
    --distractor-placement-attempts "$DISTRACTOR_PLACEMENT_ATTEMPTS" \
    --distractor-settle-steps "$DISTRACTOR_SETTLE_STEPS" \
    --distractor-max-settle-motion "$DISTRACTOR_MAX_SETTLE_MOTION" \
    "${injected_scene_stability_gate_args[@]}" \
    "${policy_preview_args[@]}" \
    "${policy_pilot_args[@]}" \
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

  if [[ "$POLICY_PILOT_ONLY" == "1" ]]; then
    success_count=$((success_count + 1))
    echo "[OK] Wrote policy pilot diagnostics for $file_name"
    continue
  fi

  scene_name="${file_name%.hdf5}"
  if [[ "$OUTPUT_FORMAT" == "directory" ]]; then
    if [[ "$CAMERA03_ONLY" == "1" ]]; then
      success_path="$OUT_DIR/$scene_name/camera03/.complete"
    else
      success_path="$OUT_DIR/$scene_name/camera00/.complete"
    fi
  elif [[ "$OUTPUT_FORMAT" == "both" ]]; then
    if [[ "$CAMERA03_ONLY" == "1" ]]; then
      success_path="$OUT_DIR/$scene_name/camera03/.complete"
    else
      success_path="$OUT_DIR/$scene_name/camera00/.complete"
    fi
  else
    success_path="$OUT_DIR/$file_name"
  fi

  if [[ -e "$success_path" ]]; then
    if [[ "$ARCHIVE_AFTER_EACH" == "1" && ( "$OUTPUT_FORMAT" == "directory" || "$OUTPUT_FORMAT" == "both" ) ]]; then
      scene_dir="$OUT_DIR/$scene_name"
      archive_path="$OUT_DIR/$scene_name.tar.gz"
      if [[ -n "${ARCHIVE_PREVIEW_ROOT:-}" ]]; then
        preview_dir="$ARCHIVE_PREVIEW_ROOT/$scene_name"
        echo "[INFO] Preparing human-review RGB previews: $preview_dir"
        python scripts/prepare_scene_rgb_previews.py \
          "$scene_dir" "$preview_dir" \
          --camera "${ARCHIVE_PREVIEW_CAMERA:-camera03}" \
          --side "${ARCHIVE_PREVIEW_SIDE:-left}" \
          --count "${ARCHIVE_PREVIEW_COUNT:-6}"
      fi
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
      if [[ "$CAMERA03_ONLY" == "1" ]]; then
        archive_required=("$scene_name/camera03/.complete")
      else
        archive_required=(
          "$scene_name/camera00/.complete"
          "$scene_name/camera00/flow/left/000000.npy"
          "$scene_name/camera00/flow_masks/left/000000.npz"
        )
      fi
      if [[ "$FIVE_CAMERA_TRAJECTORY" == "1" ]]; then
        for camera_index in 01 02; do
          archive_required+=(
            "$scene_name/camera${camera_index}/.complete"
            "$scene_name/camera${camera_index}/flow/left/000000.npy"
            "$scene_name/camera${camera_index}/flow_masks/left/000000.npz"
          )
        done
        for camera_index in 03 04; do
          archive_required+=("$scene_name/camera${camera_index}/.complete")
        done
      fi
      if ! tar -tzf "$archive_path" "${archive_required[@]}" >/dev/null 2>&1; then
        fail_count=$((fail_count + 1))
        echo "[ERROR] Archive validation failed, missing required camera or flow data: $archive_path" >&2
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
