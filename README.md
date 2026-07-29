# RoboLab Stereo RGB-D Data Generation

This repository contains the RoboLab-side code changes for generating stereo RGB-D reconstruction data.

Copy these files into a RoboLab checkout with the same relative paths:

```text
examples/generate_stereo_dataset.py
scripts/generate_random_complex_series.sh
scripts/repair_stereo_camera_pose_quat_order.py
pyproject.toml
robolab/registrations/droid/auto_env_registrations_jointpos.py
```

Main features:

```text
global stereo RGB-D
wrist stereo RGB-D
camera_to_world pose export
directory dataset format
100 procedural geometry backgrounds: bg000-bg099
random task/background/light/object/distractor/camera trajectory batch generation
archive-after-each output
pose repair script for old quat order bug
```

Run OpenPI policy server first:

```bash
cd /autodl-fs/data/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/autodl-fs/data/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

Generate 10 trajectories:

```bash
cd /autodl-fs/data/robolab
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8
export UV_LINK_MODE=copy

ARCHIVE_AFTER_EACH=1 \
RUN_TIMEOUT_SECONDS=1800 \
bash scripts/generate_random_complex_series.sh 10
```

Output:

```text
output/stereo_dataset/random_complex_series/*.tar.gz
```

