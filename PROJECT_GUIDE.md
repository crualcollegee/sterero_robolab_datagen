# RoboLab Stereo Data Generation: Project Guide

Updated: 2026-09-04 UTC

Read this before changing code or running jobs.

## Goal

Generate policy-driven robot manipulation episodes for dynamic 3D reconstruction and camera-pose learning:

- synchronized global stereo RGB;
- per-eye metric z-depth;
- OpenCV optical `camera_to_world` poses and pinhole intrinsics;
- forward dense 2D optical flow;
- rich/moving foreground;
- varied, smooth external-camera trajectories;
- real USD room geometry with depth;
- validated per-episode `tar.gz` output.

The user prioritizes diverse camera pose changes while keeping the robot/table workspace visible. Avoid cameras behind walls, robot/table intersections with room geometry, repeated simple orbits, HDR-only backgrounds, and unstable tables.

## Repository and Git

```text
repo:    /autodl-fs/data/robolab
branch:  feature/camera-md-round1
origin:  https://github.com/NVLabs/RoboLab.git
userrepo:https://github.com/crualcollegee/sterero_robolab_datagen.git
```

Important recent commits:

```text
d938f57 feat: visualize ego-compensated optical flow
1b634e4 perf: remove wrist output and stored 3d flow
fe15e4e feat: export validated stereo scene flow archives
3bb1a28 Preserve InteriorAgent floors during workspace clearing
f6a8124 Restore depth for InteriorAgent floor meshes
65a5ba3 Revert floor depth experiments to stable RGB-D baseline
ca84442 Publish only validated datasets and bound preview resources
f05c0b4 Remove unsafe InteriorAgent double-sided mesh override
```

`fe15e4e` added persisted 3D flow, but the newer `1b634e4` intentionally removed it.

Do not delete or commit these longstanding untracked files without inspection:

```text
examples/generate_stereo_dataset.py.bak_20260709_094654
examples/generate_stereo_series.sh
robolab_stereo_dataset_changes.patch
scripts/1.bash
scripts/generate_random_complex_series.sh.bak_20260709_094654
tmp3k_qw1jo/
tmpg767dauc/
uv.lock
```

## Current output decisions

### Cameras

The legacy/default mode still records only the global stereo pair:

```text
stereo_left_camera
stereo_right_camera
```

The new explicitly enabled five-camera trajectory mode records five synchronized
stereo sequences from one pilot rollout and one set of restored scene states:

```text
camera00  random external trajectory, seed 0
camera01  independent random external trajectory, seed 1
camera02  independent random external trajectory, seed 2
camera03  close stereo rig with a tightly bounded random trajectory in front of the robot
camera04  rigid gripper-mounted wrist stereo rig
```

Only `camera00`-`camera02` store forward flow and `flow_masks`. The front close-up and
wrist rigs (`camera03`, `camera04`) intentionally store only RGB, depth, camera
parameters, metadata, and `.complete`. Their sensors do not request RTX motion
vectors or the semantic/instance auxiliary outputs used by hybrid flow.

Enable it with `--five-camera-trajectory` in the Python generator or
`FIVE_CAMERA_TRAJECTORY=1` in `generate_random_complex_series.sh`. The three
broad random trajectories use `--multi-camera-seed-stride`. Camera03 uses an
independent smooth random trajectory inside hard per-axis bounds around the
validated close-up pose `(0.82, 0.0, 0.75) -> (0.55, 0.0, 0.40)` and applies a
small, clipped end-effector-follow term to its look-at point. Its default center
jitter is `(0.080, 0.140, 0.100)` m and target jitter is
`(0.035, 0.065, 0.040)` m. Camera height has a `0.35` inverse coupling to target
height: descending makes the view flatter/upward, while ascending makes it more
downward. Set `FRONT_CAMERA_MOTION=0` only for a fixed-view
compatibility run. Front position/look-at, jitter, and wrist baseline are
separately configurable. All five cameras share task actions,
physics state, room, objects, distractors, timestamps, and frame IDs.

The bounded close-up signal is peak-to-peak normalized, so every seed traverses
both sides of its sampled per-axis range instead of occasionally producing a
nearly static one-sided excursion.

Camera03 production now uses a spherical operation-view trajectory around the
tabletop target. Defaults cover 0.32--0.72 m target distance, 8--78 degrees of
elevation above the table, and -38--38 degrees of azimuth. This intentionally
includes close framing where roughly half the table may be visible, standard
operation distance, near-horizontal views, and strongly top-down views. The
spherical look-at height is independently fixed at world z=0.08 m, the
tabletop object layer; the legacy Cartesian/fixed view may still use its old
look-at z value. This prevents high-elevation views from aiming through empty
space above the table and landing on the surrounding floor. When the pilot
end-effector trajectory is available, camera03 also maintains a nominal 0.45 m
center-to-end-effector clearance (`FRONT_CAMERA_MIN_ARM_DISTANCE`). Unsafe
samples retreat along their existing target-to-camera ray, preserving the
look-at point, azimuth/elevation and intrinsics while preventing the nearby arm
from filling most of the image. Camera03 also applies a 31-frame zero-phase
Hann pass after causal motion/safety limiting, which removes visible direction
corners without temporal lag. The near-distance/high-elevation combination is
excluded: distance begins increasing smoothly over the 20 degrees before the
68-degree overhead threshold, and overhead views retain at least 0.55 m target
distance (normally close to the configured far endpoint). Near low-angle views
and far overhead views remain available. Use
`CAMERA03_ONLY=1` with `FIVE_CAMERA_TRAJECTORY=0` for fast validation archives
that contain only `camera03` stereo RGB-D and no flow. This is an opt-in test
mode; formal production remains `FIVE_CAMERA_TRAJECTORY=1` with all five rigs.

Pi0.5 still requires RoboLab's original mono `wrist_cam` internally for policy
inference. It remains excluded from dataset output. `camera04` instead uses two
dedicated dataset sensors, so policy calibration is not modified.

### Flow

Current output stores only:

```text
flow/{left,right}        source-grid forward 2D flow, float32 [H,W,2]
flow_masks/{left,right}  NPZ containing flow_invalid bool [H,W]
```

In five-camera mode these optional directories exist only for
`camera00`-`camera02`; they are absent from `camera03` and `camera04`.

Do not add these back without explicit discussion:

```text
flow3d/
world scene flow
approximate occlusion masks
```

## Archive format

```text
<scene>/
  camera00/
    rgb/{left,right}/000000.png
    depth/{left,right}/000000.npy
    flow/{left,right}/000000.npy
    flow_masks/{left,right}/000000.npz  # flow_invalid: bool [H,W]
    camera/{left.npz,right.npz}
    camera/intrinsics.json
    metadata.json
    .complete
  camera01/  # present in five-camera mode; includes flow + flow_masks
  camera02/
  camera03/  # RGB/depth/camera only; no flow directories
  camera04/  # RGB/depth/camera only; no flow directories; 0.05 m baseline
```

Contract:

- RGB: PNG, uint8 RGB `[H,W,3]`.
- Depth: float32 NPY `[H,W]`, meters, optical-camera z-depth.
- Invalid depth is zero; NaN/Inf/non-positive values are converted to zero.
- Flow: forward pixel displacement `t -> t+1`.
- `N` RGB/depth/pose frames and `N-1` flow/valid frames.
- NPZ fields are exactly `frame_ids`, `intrinsics`, `camera_to_world`.
- Pose axes: OpenCV optical `+x` right, `+y` down, `+z` forward.
- `.complete` is written only after validation succeeds.
- Batch generation archives each validated directory, then removes the directory.

Older output folders may contain obsolete wrist or `flow3d` data. Do not mix them with the current series.

### Two trajectories per task

`TASK_PAIR_MODE=1 FIVE_CAMERA_TRAJECTORY=1` makes the batch script schedule all
44 tasks exactly twice (88 trajectory archives). The two runs for each task are
assigned two distinct InteriorAgent room styles; random object/distractor/camera
seeds also differ. With `TASK_OVERRIDE=<Task>`, pair mode requires `N=2` and is
useful for a single-task pair test.

## Backgrounds

Production defaults:

```text
GEOMETRY_BACKGROUND_SOURCE=interioragent
GEOMETRY_BACKGROUND_STYLE=random
GEOMETRY_BACKGROUND_PLACEMENT=surround
GEOMETRY_BACKGROUND_SHELL=off
RANDOMIZE_HDR_BACKGROUND=0
```

InteriorAgent assets:

```text
/autodl-fs/data/spatialverse/InteriorAgent
```

The script currently reports 25 usable room backgrounds.

HDR/EXR dome images have RGB but no geometry depth and are disabled. The user rejected fake procedural wall/floor shells and strange terrain as a replacement for real room geometry. Black depth can mean missing/invisible geometry or HDR-only content; inspect depth rather than trusting RGB.

## Camera trajectory

Batch generation uses:

```text
--camera-motion-mode table_full_random
```

The current system includes candidate generation, smooth random motion, geometry gating, visual-quality gating, random composition, room safety, and policy pilot/state replay. Do not replace it with a simple fixed orbit.

Historical design reference:

```text
/root/autodl-fs/tmp/camera.md
```

Primary camera implementation:

```text
robolab/core/camera_trajectory.py
```

## Sampling rates

Typical current values:

```text
physics:             120 Hz (dt about 0.008333 s)
environment/render:   15 Hz (dt about 0.066667 s)
RECORD_EVERY:           2
saved data:           7.5 Hz
```

Camera pose changes at 15 Hz; every second environment step is saved. The first second (15 environment steps) is skipped for physics settling.

## How 2D flow is generated

Isaac exposes current-to-previous screen-space motion vectors. The generator now
uses a hybrid method:

1. records RGB-D, exact camera poses, and exact pilot scene states in normal
   temporal order;
2. replays only saved frames in reverse temporal order;
3. reads native current-to-previous vectors, which are original-time `t -> t+1`
   vectors defined directly on frame `t`'s source grid;
4. labels each static background root with the current Isaac Sim `add_labels`
   API and combines the transient semantic mask with strict USD-root selection
   from uncolorized `instance_id_segmentation_fast`; selected pixels are replaced
   with depth-and-camera-pose geometric forward flow, while robot, task objects,
   distractors, and other paths retain native flow;
5. writes `flow_invalid` from invalid source depth or non-finite/native-invalid
   vectors, without folding occlusion or out-of-frame into the mask.

The transient `motion:static_background` semantic image is only used to select
the geometric branch and is not stored in the final dataset. No residual-motion
threshold is used as ground truth and no approximate occlusion field is written.
Old HDF5 files without reverse-replay forward vectors are exported without both
optional flow directories rather than being splat-inverted.

Important 2026-08-22 motion-mask fix: do not restore per-Gprim semantic labeling.
Calling the old deprecated semantic API on about 2,000 InteriorAgent descendants
made the instance annotator collapse visible IDs into background-owned mappings,
which selected 100% of the image and/or caused Kit to exit before recording.
Root-only `add_labels` preserved real per-object USD mappings. A GPU short test on
`interioragent_kujiale_0022` reported 46/45 visible left/right instance IDs,
semantic background ratios 0.280/0.309, and strict background path ratios
0.900/0.901; robot and task-object IDs mapped to their own non-background paths.
The diagnostic output is printed once per eye. Existing v5 archives retain their
old flow and must be regenerated to receive this fix.

Production enables the fallback geometry shell by default. An independent depth
coverage gate remains enabled even when the RGB/texture visual-quality gate is
disabled: by default at least 90% of stereo pixels must have valid depth, and at
most 10% of sampled frames may fall below that threshold.

The temporary HDF5 forward-flow and invalid-mask datasets use chunks with a
single-frame temporal dimension and at most `180x320` spatial tiles. This is
required because reverse per-frame writes otherwise repeatedly recompress h5py's
automatic nine-frame chunks and can exceed the per-run timeout.

## Fixed-observer motion visualization

Commit `d938f57` updated:

```text
scripts/render_stereo_motion_video.py
robolab/core/scene_flow.py
tests/test_scene_flow.py
```

It computes camera-motion flow from depth, intrinsics, and consecutive poses, then:

```text
residual_flow = observed_flow - camera_motion_flow
```

Expected result:

- static wall/floor/table becomes near zero residual motion;
- moving robot links and manipulated objects remain non-zero;
- invalid flow is excluded from residual motion.

The current preview is two eye rows (left/right) and five columns:

1. RGB;
2. metric depth;
3. derived motion mask (white=moving, dark red=invalid);
4. raw forward 2D flow (camera + objects);
5. explicit flow validity (white=valid, bright red=invalid, ratio in title).

The fifth validity column was added on 2026-08-24 because a conventional HSV
flow wheel makes near-zero valid flow and invalid flow both appear black. It
reads the persisted `flow_invalid` field and displays
`flow_valid = ~flow_invalid`; it is not a new stored dataset field. The fourth
flow panel keeps every valid pixel at full brightness and encodes flow magnitude
as saturation instead: small flow is pale/white, large flow is vivid, and only
invalid flow is pure black. Thus pure-black flow pixels correspond to
bright-red pixels in column 5. Output at `--eye-width 480` is `2400x540`.

The renderer now exposes `--ffmpeg-preset` and `--ffmpeg-threads` (default 2).
For large AutoFS archives use `TMPDIR=/autodl-fs/data/tmp`; the root filesystem
can fill with a 3.2 GB extraction plus rendered PNG frames. A robust low-resource
retry uses `--ffmpeg-preset ultrafast --ffmpeg-threads 1 --crf 22`.

On 2026-08-24 a native-flow validity bug was found from the right-eye robot
border in frame 66 of the PutTwoMugs sample. RTX returned exact `(-1, -1)`
vectors for pixels without a temporal correspondence. Because the vector is
finite and source depth is valid, the old sanitizer incorrectly kept it as a
valid 1.414-pixel flow. `sanitize_native_forward_flow()` now treats only the
exact two-component `(-1, -1)` sentinel as `native_invalid`; other negative
vectors and finite out-of-frame endpoints remain valid. Archives generated
before this fix can contain unmasked sentinels and must be repaired or
regenerated before publication. The ModelScope sample validator also rejects an
unmasked exact sentinel.

## Important files

```text
examples/generate_stereo_dataset.py
scripts/generate_random_complex_series.sh
robolab/core/camera_trajectory.py
robolab/core/scene_flow.py
scripts/render_stereo_archive_rgbd_video.py
scripts/render_stereo_motion_video.py
tests/test_scene_flow.py
```

## Environments and paths

OpenPI:

```text
repo:       /autodl-fs/data/openpi
environment:/root/venvs/openpi
checkpoint: /autodl-fs/data/openpi_cache/openpi-assets-simeval/pi05_droid_jointpos
```

RoboLab:

```text
repo:       /autodl-fs/data/robolab
environment:/autodl-fs/data/robolab/.venv
```

Use:

```text
TMPDIR=/autodl-fs/data/tmp
OMNI_USER_CACHE_DIR=/autodl-fs/data/ov_cache
```

The root overlay has filled before. `/autodl-fs/data` has a 200,000 inode limit and has recently been around 87-88% used. Avoid abandoned extracted archives.

## Commands after restart

### Five-camera one-trajectory demo

After starting the Pi0.5 policy server, run:

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
mkdir -p /root/autodl-tmp/robolab_interioragent_cache /root/autodl-tmp/optix_cache /root/autodl-tmp/robolab_tmp
export INTERIORAGENT_RUNTIME_CACHE=/root/autodl-tmp/robolab_interioragent_cache
export OPTIX_CACHE_PATH=/root/autodl-tmp/optix_cache
export TMPDIR=/root/autodl-tmp/robolab_tmp
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8
export UV_LINK_MODE=copy
export OMNI_USER_CACHE_DIR=/autodl-fs/data/ov_cache

FIVE_CAMERA_TRAJECTORY=1 \
TASK_OVERRIDE=ClearOrganicObjectsTask \
OUT_DIR=output/stereo_dataset/five_camera_demo \
GEOMETRY_BACKGROUND_SOURCE=interioragent \
GEOMETRY_BACKGROUND_STYLE=random \
GEOMETRY_BACKGROUND_PLACEMENT=surround \
GEOMETRY_BACKGROUND_SHELL=off \
RANDOMIZE_HDR_BACKGROUND=0 \
ADD_GEOMETRY_BACKGROUND=1 \
CAMERA_VISUAL_QUALITY_GATE=0 \
STEPS=300 RECORD_EVERY=2 WIDTH=1280 HEIGHT=720 \
BASELINE=0.16 WRIST_BASELINE=0.05 DISTRACTORS=12 \
FRONT_CAMERA_POSITION="0.82 0.0 0.75" \
FRONT_CAMERA_LOOK_AT="0.55 0.0 0.40" \
ARCHIVE_COMPRESSION_LEVEL=1 RUN_TIMEOUT_SECONDS=18000 \
bash scripts/generate_random_complex_series.sh 1
```

`camera03` used to be behind the robot at `(-0.65, 0, 1.25)`. It was first
mirrored to a wide front view at `(1.55, 0, 1.25)`. Its current close-operation
trajectory is centered at `(0.82, 0, 0.75)` looking at `(0.55, 0, 0.40)`. This
gives an approximately 52-degree downward oblique base view: the tabletop remains
dominant, while the gripper side and object-contact region remain visible instead
of being hidden by the arm in a near-vertical top-down view. The center and target
now move smoothly inside the tight bounds documented above; camera intrinsics and
the 0.16 m parallel-stereo baseline do not change.
Camera03 deliberately keeps the same 2.1 mm pinhole intrinsics as camera00-02;
the close framing is produced only by extrinsics. The background
visibility-corridor check includes this fixed pose.
The old `REAR_CAMERA_POSITION`/`REAR_CAMERA_LOOK_AT` environment variables and
`--rear-camera-*` CLI spellings remain compatibility aliases only.

A 2026-09-03 GPU smoke render at 640x360 validated the oblique external pose
without changing camera03 intrinsics. Four synchronized left/right frames show
the tabletop, task objects, bin, and gripper work area together. The temporary
left/right contact sheet is at:

```text
/root/autodl-tmp/camera03_oblique_candidate3_contact.png
```

The smoke source is
`/root/autodl-tmp/camera03_oblique_candidate3/candidate3.hdf5`.
This is not production data.

Render all five views with one archive extraction pass:

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
DATA=$(find output/stereo_dataset/five_camera_demo -maxdepth 1 -type f -name '*.tar.gz' | sort | tail -1)
mkdir -p output/stereo_dataset/five_camera_demo/preview

TMPDIR=/root/autodl-tmp/robolab_tmp \
uv run --no-sync python scripts/render_stereo_motion_video.py \
  "$DATA" \
  --output-dir output/stereo_dataset/five_camera_demo/preview \
  --cameras camera00 camera01 camera02 camera03 camera04 \
  --fps 7.5 --eye-width 480 --motion-threshold-px 1.5 \
  --ffmpeg-preset fast --ffmpeg-threads 2
```

The renderer writes five videos. `camera00`-`camera02` use the two-row,
five-column motion/flow layout. `camera03` and `camera04` automatically use a
two-row, two-column RGB/depth-only layout and an `_rgbd_verify.mp4` suffix.

As of 2026-08-27 the implementation passed Python/Bash syntax, 34 camera/flow/
preview tests, a one-trajectory five-camera batch dry run, and a full GPU/Pi0.5
smoke render. The completed demo is:

```text
output/stereo_dataset/five_camera_demo/
  20260827_131122_00000_ClearOrganicObjectsTask_seed1788056718.tar.gz
  preview/*camera00_motion_verify.mp4
  preview/*camera01_motion_verify.mp4
  preview/*camera02_motion_verify.mp4
  preview/*camera03_rgbd_verify.mp4
  preview/*camera04_rgbd_verify.mp4
```

The archive is 9,867,652,122 bytes and has 143 synchronized RGB/depth frames
per camera. `camera00`-`camera02` each have 142 forward-flow/mask frames and
0.16 m baselines; their center-trajectory spans are 1.7891, 0.9376, and
2.0937 m. `camera03` is fixed with a 0.16 m baseline and no flow. `camera04`
moves 0.3523 m with the wrist, has a 0.05 m baseline, and has no flow. All
numeric/directory checks passed, including exact shared frame IDs, dtype/shape,
finite depth/flow, pose matrices, flow-mask pairing, and final `.complete`
markers. The run took about 67 minutes end to end, including roughly 12 minutes
of CPU-only planning and roughly 11 minutes of gzip/archive validation. Plan
future production capacity using about one hour per five-camera trajectory at
this resolution unless planning/archive throughput is improved.

The preview renderer now resizes depth, validity, and source-grid flow with
consistent nearest-neighbour sampling before removing camera motion. Mixing
bilinearly resized flow with nearest-neighbour depth had created false thin
motion outlines at static depth discontinuities; this was a visualization
artifact and did not modify the stored flow.

### Terminal 1: OpenPI policy server

```bash
cd /autodl-fs/data/openpi
source /root/venvs/openpi/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p /autodl-fs/data/tmp
export TMPDIR=/autodl-fs/data/tmp
export OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=/autodl-fs/data/openpi_cache/openpi-assets-simeval/pi05_droid_jointpos
```

If port 8000 is occupied, check first:

```bash
ss -ltnp | grep ':8000'
```

### Terminal 2: generate 5 episodes

Change the final `5` to `50` for 50 episodes.

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8
export UV_LINK_MODE=copy
export OMNI_USER_CACHE_DIR=/autodl-fs/data/ov_cache
export TMPDIR=/autodl-fs/data/tmp

OUT_DIR=output/stereo_dataset/global_stereo_2dflow_series \
GEOMETRY_BACKGROUND_SOURCE=interioragent \
GEOMETRY_BACKGROUND_STYLE=random \
GEOMETRY_BACKGROUND_PLACEMENT=surround \
GEOMETRY_BACKGROUND_SHELL=off \
RANDOMIZE_HDR_BACKGROUND=0 \
ADD_GEOMETRY_BACKGROUND=1 \
STEPS=300 \
RECORD_EVERY=2 \
WIDTH=1280 \
HEIGHT=720 \
BASELINE=0.16 \
DISTRACTORS=12 \
OUTPUT_FORMAT=directory \
ARCHIVE_AFTER_EACH=1 \
ARCHIVE_COMPRESSION_LEVEL=1 \
bash scripts/generate_random_complex_series.sh 5
```

Use a repository-relative `OUT_DIR`.

### RGB/depth preview

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
DATA=$(find output/stereo_dataset/global_stereo_2dflow_series \
  -maxdepth 1 -type f -name '*.tar.gz' | sort | tail -1)
mkdir -p output/stereo_dataset/global_stereo_2dflow_series/preview

TMPDIR=/autodl-fs/data/tmp \
python scripts/render_stereo_archive_rgbd_video.py \
  "$DATA" \
  --output-dir output/stereo_dataset/global_stereo_2dflow_series/preview \
  --camera camera00 \
  --fps 7.5
```

### Residual/fixed-view motion preview

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
DATA=$(find output/stereo_dataset/global_stereo_2dflow_series \
  -maxdepth 1 -type f -name '*.tar.gz' | sort | tail -1)
mkdir -p output/stereo_dataset/global_stereo_2dflow_series/preview

TMPDIR=/autodl-fs/data/tmp \
python scripts/render_stereo_motion_video.py \
  "$DATA" \
  --output-dir output/stereo_dataset/global_stereo_2dflow_series/preview \
  --camera camera00 \
  --fps 7.5 \
  --eye-width 480 \
  --fixed-view-frame 0 \
  --motion-threshold-px 1.5
```

Controls:

```text
--fixed-view-frame      fixed observer frame
--motion-threshold-px   residual threshold in original-image pixels
--flow2d-max-px         flow color saturation
--eye-width             preview width; smaller is faster
```

## Latest known current-format data

```text
output/stereo_dataset/global_stereo_2dflow_series/
20260817_212646_00000_BlackItemsInBinTask_seed1787189565.tar.gz
```

## Verification status

For `1b634e4`:

- Python and Bash syntax passed;
- 2D flow unit tests passed;
- batch `DRY_RUN=1` passed;
- archive checks require only `camera00`, 2D flow, validity, and `.complete`.

For `d938f57`:

- `tests/test_scene_flow.py`: 4 passed;
- static-scene camera translation cancels to zero residual;
- true object motion remains in residual;
- a synthetic 3-frame stereo directory generated a valid five-row MP4.

A full preview of the large current tar was not completed during agent validation because sequential gzip extraction on AutoFS was very slow. Two validation processes ran in the background for 10-12 minutes and were explicitly killed; their temporary extraction directories were removed. This was archive I/O, not an algorithm failure.

## Performance

Compared with the old four-eye 2D+3D flow version, current production should have:

- two recorded cameras instead of four;
- no full-resolution 3D lifting;
- no `flow3d` files/compression;
- roughly 75-80% less flow-related uncompressed storage;
- often roughly 50-65% smaller archives;
- estimated 35-55% lower total episode time, not yet rigorously benchmarked.

Remaining costs include RTX motion vectors, rendering every intermediate step, InteriorAgent USD loading, policy inference/state replay, PNG/NPY writes, and tar compression.

## Common failures

- Port 8000 occupied: existing policy server may already be valid.
- `VIRTUAL_ENV` mismatch: OpenPI and RoboLab use different environments.
- `/tmp` or `/` full: export `TMPDIR=/autodl-fs/data/tmp`.
- Inode pressure: check `df -i /autodl-fs/data`.
- Large tar preview slow: prefer previewing an unarchived directory during iteration.
- RGB background with black depth: likely no renderable geometry or HDR-only content.
- Headless GLFW, deprecated API, and some MDL warnings are often non-fatal.
- CUDA illegal access, Vulkan OOM, traceback, or missing `.complete` are fatal.

## Next-agent checklist

```bash
cd /autodl-fs/data/robolab
git status --short --branch
git log --oneline -8
nvidia-smi
df -h / /autodl-fs/data
df -i /autodl-fs/data
ss -ltnp | grep ':8000' || true
ps -eo pid,etime,cmd | grep -E 'generate_stereo|generate_random_complex|serve_policy|render_stereo' | grep -v grep
```

## ModelScope publication

`scripts/publish_modelscope_dataset.py` prepares a validated eight-frame sample
from the first v6 archive and uploads the complete release to
`bairuide/robolab_stereo_2`. It reads authentication only from
`MODELSCOPE_API_TOKEN`, verifies the repository before uploading, and uses the
official `modelscope-hub` resumable folder uploader. Remote layout:

```text
data/global_stereo_2dflow_v6/       # three full tar.gz files + env_cfg.json
previews/global_stereo_2dflow_v6/   # all motion verification MP4 files
samples/                            # loader-compatible 8-frame sample + checksums
```

The local sample is already prepared at
`output/stereo_dataset/global_stereo_2dflow_v6/samples` (170 MB, 68 files).
Upload after exporting the token:

```bash
export MODELSCOPE_API_TOKEN='...'
python scripts/publish_modelscope_dataset.py --skip-prepare
```

Use `--skip-prepare --dry-run` to verify the plan without network access. Use
`--force-sample --prepare-only` only when intentionally replacing the local
sample; gzip extraction requires one sequential pass through the source archive.

Before finishing a change:

- preserve unrelated files;
- run `git diff --check`;
- run relevant tests;
- run batch dry-run if command assembly changed;
- generate a small preview before launching 50 episodes;
- make a focused commit;
- report the commit hash and exact commands.
