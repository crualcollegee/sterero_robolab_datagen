# RoboLab 多相机双目数据生成指南

本文面向首次拿到本项目代码、希望从零搭建环境并生成数据的使用者。项目代码基于 NVIDIA RoboLab 扩展，增加了 InteriorAgent 三维室内背景、π0.5 闭环机器人操作、五组同步双目相机、深度、相机参数和前向二维光流。

## 1. 发布分支包含和不包含什么

代码仓库：

```text
https://github.com/crualcollegee/sterero_robolab_datagen
```

使用分支：

```text
release/stereo-datagen-v1
```

该分支包含数据生成器、相机轨迹、数据校验、可视化、批量生成脚本和测试，但为了避免重复上传数 GB 的第三方文件，不包含下列内容：

1. NVIDIA RoboLab 的 Git LFS 资产；
2. SpatialVerse InteriorAgent 房间资产；
3. OpenPI 源码和 Python 环境；
4. π0.5 DROID JointPos 模型权重；
5. Isaac Sim、Isaac Lab、系统驱动和 ffmpeg。

这些内容必须按下文准备后才能运行正式生产命令。

## 2. 硬件与系统要求

- Ubuntu 22.04 或兼容 Linux；
- Python 3.11；
- NVIDIA RTX GPU 和可用的 NVIDIA 驱动；
- 建议至少 48 GB GPU 显存，以便 π0.5 与 Isaac Sim 同机运行；
- `git-lfs`、`uv`、`ffmpeg`、`rsync`；
- 建议预留至少 100 GB 可用空间。一个 1280×720、500-step 的五相机样本及其临时文件可能占用十几 GB，InteriorAgent 房间池和 π0.5 权重还需要额外空间。

Ubuntu 基础工具示例：

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg rsync
git lfs install
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. 下载代码及所需内容

建议让 `robolab` 和 `openpi` 位于同一父目录，因为 RoboLab 的 `pyproject.toml` 从 `../openpi/packages/openpi-client` 安装客户端。以下示例使用 `/data/robolab_workspace`，可替换为容量充足的目录。

### 3.1 下载本项目代码

```bash
mkdir -p /data/robolab_workspace
cd /data/robolab_workspace

git clone \
  --branch release/stereo-datagen-v1 \
  --single-branch \
  https://github.com/crualcollegee/sterero_robolab_datagen.git \
  robolab
```

### 3.2 补齐 NVIDIA RoboLab 官方资产

发布分支没有携带官方 Git LFS 内容。请从 NVIDIA 官方仓库下载：

```text
https://github.com/NVLabs/RoboLab
```

执行：

```bash
cd /data/robolab_workspace

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/NVLabs/RoboLab.git \
  robolab_upstream

cd robolab_upstream
git lfs pull

rsync -a assets/ ../robolab/assets/
```

这里只复制运行所需的 `assets/`。不要把复制得到的大文件重新提交到本项目仓库。官方资产通常约 7 GB，具体大小会随上游版本变化。

### 3.3 下载并安装 OpenPI

当前验证过的 OpenPI 来源是：

```text
https://github.com/xuningy/openpi
commit: aa64205
```

执行：

```bash
cd /data/robolab_workspace

git clone --recurse-submodules https://github.com/xuningy/openpi.git openpi
cd openpi
git checkout aa64205
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

OpenPI 必须使用独立环境，不要安装到 RoboLab 的 `.venv` 中。

### 3.4 安装 RoboLab 环境

RoboLab 当前依赖 Python 3.11、Isaac Sim 5.0 和 Isaac Lab 2.2.0。`uv sync` 会安装 Python 侧依赖，包括 Isaac Sim/Isaac Lab 包和相邻目录中的 OpenPI client。

```bash
cd /data/robolab_workspace/robolab

uv venv --python 3.11
source .venv/bin/activate
uv sync
```

先做基础检查：

```bash
uv run --no-sync python -c "import isaacsim, isaaclab; print('Isaac imports OK')"
uv run --no-sync pytest -q \
  tests/test_camera_trajectory.py \
  tests/test_scene_flow.py \
  tests/test_load_background.py
```

### 3.5 下载 InteriorAgent 三维房间

房间来源：

```text
https://huggingface.co/datasets/spatialverse/InteriorAgent
```

生产管线要求真实三维房间；普通 HDR 图片只有 RGB，没有正确的背景深度。InteriorAgent 数据较大且文件数多，建议先以 LFS pointer 方式克隆，再使用项目脚本逐个下载、校验并压缩房间，避免长期占用大量 inode。

```bash
mkdir -p /data/spatialverse

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/datasets/spatialverse/InteriorAgent \
  /data/spatialverse/InteriorAgent

cd /data/robolab_workspace/robolab

INTERIORAGENT_ROOT=/data/spatialverse/InteriorAgent \
INTERIORAGENT_ARCHIVE_ROOT=/data/spatialverse/InteriorAgent_archives \
bash scripts/download_and_archive_interioragent.sh
```

该步骤可能耗时较长。正式运行前，至少应在 `/data/spatialverse/InteriorAgent_archives` 中看到完整的 `kujiale_*.tar.gz` 房间文件。

### 3.6 π0.5 权重

当前生产使用：

```text
config: pi05_droid_jointpos
checkpoint: gs://openpi-assets-simeval/pi05_droid_jointpos
```

权重约 12 GB。OpenPI 第一次启动服务时会自动下载。通过 `OPENPI_DATA_HOME` 指定缓存到容量充足的磁盘：

```bash
export OPENPI_DATA_HOME=/data/openpi_cache
```

## 4. 启动 π0.5 服务

在终端 1 执行：

```bash
cd /data/robolab_workspace/openpi

export OPENPI_DATA_HOME=/data/openpi_cache
export TMPDIR=/data/tmp
export OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "$OPENPI_DATA_HOME" "$TMPDIR"

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

服务默认监听 `localhost:8000`。看到服务开始监听后再启动数据生成。

检查端口：

```bash
ss -ltnp | grep ':8000'
```

## 5. 正式生产命令

在终端 2 执行。下面命令生成一个随机任务、随机 InteriorAgent 房间的 500-step 五相机样本；将最后的 `1` 改成需要生成的样本数量。

```bash
cd /data/robolab_workspace/robolab
source .venv/bin/activate

mkdir -p \
  /data/robolab_runtime/interioragent_cache \
  /data/robolab_runtime/optix_cache \
  /data/robolab_runtime/tmp \
  /data/robolab_runtime/ov_cache

export INTERIORAGENT_ROOT=/data/spatialverse/InteriorAgent
export INTERIORAGENT_ARCHIVE_ROOT=/data/spatialverse/InteriorAgent_archives
export INTERIORAGENT_RUNTIME_CACHE=/data/robolab_runtime/interioragent_cache
export OPTIX_CACHE_PATH=/data/robolab_runtime/optix_cache
export TMPDIR=/data/robolab_runtime/tmp
export OMNI_USER_CACHE_DIR=/data/robolab_runtime/ov_cache
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8
export UV_LINK_MODE=copy

FIVE_CAMERA_TRAJECTORY=1 \
CAMERA03_ONLY=0 \
TASK_PAIR_MODE=0 \
OUT_DIR=output/stereo_dataset/five_camera_formal_v1 \
GEOMETRY_BACKGROUND_SOURCE=interioragent \
GEOMETRY_BACKGROUND_STYLE=random \
GEOMETRY_BACKGROUND_PLACEMENT=surround \
GEOMETRY_BACKGROUND_SHELL=off \
GEOMETRY_BACKGROUND_RANDOMIZE_POSITION=1 \
RANDOMIZE_HDR_BACKGROUND=0 \
ADD_GEOMETRY_BACKGROUND=1 \
CAMERA_VISUAL_QUALITY_GATE=0 \
CAMERA_DEPTH_COVERAGE_GATE=0 \
POLICY_INTERACTION_GATE=0 \
INJECTED_SCENE_STABILITY_GATE=1 \
STEPS=500 \
RECORD_EVERY=2 \
WIDTH=1280 \
HEIGHT=720 \
BASELINE=0.16 \
WRIST_BASELINE=0.05 \
DISTRACTORS=12 \
DISTRACTOR_MODE=real \
ARCHIVE_COMPRESSION_LEVEL=1 \
RUN_TIMEOUT_SECONDS=18000 \
bash scripts/generate_random_complex_series.sh 1
```

说明：

- 未设置 `TASK_OVERRIDE`，因此从脚本当前启用的 44 个任务中随机选择；
- `GEOMETRY_BACKGROUND_STYLE=random` 会随机选择可用房间；
- 每个样本只执行一次策略，再回放同一状态序列渲染五组同步双目相机；
- `camera00`、`camera01`、`camera02` 保存 RGB、深度、相机参数和前向二维光流；
- `camera03` 是操作近景双目相机，不保存光流；
- `camera04` 是腕部双目相机，使用较小的 0.05 m baseline，不保存光流；
- 成功结果写成一个经过内部结构校验的 `.tar.gz`；
- 上述命令关闭视觉、深度覆盖率和策略交互三个可选门控，与当前正式生产设置一致。关闭门控不等于数据一定优质，必须做人工抽检；如需严格深度筛选，可改为 `CAMERA_DEPTH_COVERAGE_GATE=1`。

固定任务可增加：

```bash
TASK_OVERRIDE=ClearOrganicObjectsTask
```

只快速测试 camera03 可使用：

```bash
FIVE_CAMERA_TRAJECTORY=0 CAMERA03_ONLY=1
```

## 6. 首次运行建议

不要在环境尚未验证时直接批量生成。建议先运行一个低分辨率短样本：

```bash
FIVE_CAMERA_TRAJECTORY=1 \
CAMERA03_ONLY=0 \
TASK_OVERRIDE=ClearOrganicObjectsTask \
OUT_DIR=output/stereo_dataset/smoke_test \
GEOMETRY_BACKGROUND_SOURCE=interioragent \
GEOMETRY_BACKGROUND_STYLE=random \
GEOMETRY_BACKGROUND_PLACEMENT=surround \
GEOMETRY_BACKGROUND_SHELL=off \
GEOMETRY_BACKGROUND_RANDOMIZE_POSITION=1 \
RANDOMIZE_HDR_BACKGROUND=0 \
ADD_GEOMETRY_BACKGROUND=1 \
CAMERA_VISUAL_QUALITY_GATE=0 \
CAMERA_DEPTH_COVERAGE_GATE=0 \
POLICY_INTERACTION_GATE=0 \
STEPS=60 RECORD_EVERY=2 WIDTH=640 HEIGHT=360 \
BASELINE=0.16 WRIST_BASELINE=0.05 DISTRACTORS=2 \
ARCHIVE_COMPRESSION_LEVEL=1 RUN_TIMEOUT_SECONDS=7200 \
bash scripts/generate_random_complex_series.sh 1
```

## 7. 生成可视化

以下命令为最新一个五相机归档分别生成预览视频：

```bash
cd /data/robolab_workspace/robolab
source .venv/bin/activate

OUT_DIR=output/stereo_dataset/five_camera_formal_v1
DATA=$(find "$OUT_DIR" -maxdepth 1 -type f -name '*.tar.gz' | sort | tail -1)
mkdir -p "$OUT_DIR/preview"

TMPDIR=/data/robolab_runtime/tmp \
uv run --no-sync python scripts/render_stereo_motion_video.py \
  "$DATA" \
  --output-dir "$OUT_DIR/preview" \
  --cameras camera00 camera01 camera02 camera03 camera04 \
  --fps 7.5 \
  --eye-width 480 \
  --motion-threshold-px 1.5 \
  --ffmpeg-preset ultrafast \
  --ffmpeg-threads 1 \
  --crf 22
```

预期得到五个视频：

- `camera00` 至 `camera02`：左右目两行，RGB、深度、运动 mask、光流和 flow validity 五列；
- `camera03`、`camera04`：左右目两行，RGB 和深度两列。

人工检查至少包括：

1. 左右目时间同步且没有单目被完全遮挡；
2. 房间、墙体和家具具有有效深度，不是只有 RGB 的贴图背景；
3. 相机没有穿墙、穿桌或进入机械臂；
4. camera03 持续覆盖桌面、夹爪和任务物体；
5. 光流黑色无效区域与 validity 红色区域一致；
6. 机器人是否确实接近、接触或移动任务物体。

## 8. 输出概览

每个成功样本是一个 `.tar.gz`，内部包含五个同步双目序列。数据包括 RGB、米制 z-depth、针孔内参、OpenCV optical camera-to-world 位姿，以及前三个自由相机的 source-grid forward 2D flow 与 invalid mask。完整字段定义见 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。

## 9. 常见问题

- **端口 8000 被占用**：先确认已有 π0.5 服务是否可用，不要重复启动。
- **OpenPI client 找不到**：确认 `openpi` 与 `robolab` 是相邻目录，然后重新执行 `uv sync`。
- **背景有 RGB 但没有深度**：确认使用 InteriorAgent 三维房间，而不是 HDR 背景，并检查房间归档是否完整。
- **系统盘或 `/tmp` 爆满**：将 `TMPDIR`、OptiX、Omniverse 和 InteriorAgent runtime cache 放到大容量数据盘。
- **只生成 HDF5、没有 tar.gz**：通常表示生成尚未完成、进程异常退出或最终校验未通过。
- **生成很慢**：1280×720 五相机包含状态回放、光流、PNG/NPY 写入和压缩；一条数据可能需要约一小时，具体取决于 GPU、策略推理和存储性能。
- **π0.5 第一次启动很慢**：首次需要下载约 12 GB 权重并完成 JAX 初始化。
- **代理错误**：下载 GitHub、Hugging Face 或 GCS 资源时检查 `HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY` 是否指向仍在运行的代理。

## 10. 版本与复现信息

本文对应：

```text
data-generation release branch: release/stereo-datagen-v1
source snapshot: f29b30a
OpenPI fork commit: aa64205
InteriorAgent commit used during development: ac599fe
Python: 3.11
Isaac Sim: 5.0.0
Isaac Lab: 2.2.0
```

为了复现实验，应同时记录最终使用的代码提交、任务名、随机种子、OpenPI 权重、InteriorAgent 房间和生产命令。
