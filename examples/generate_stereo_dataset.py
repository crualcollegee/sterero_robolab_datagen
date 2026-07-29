#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate a small stereo RGB-D dataset from RoboLab.

This script is intentionally independent from RoboLab's built-in recorder so we
can save exactly the fields needed for stereo reconstruction experiments:
left/right RGB, depth, camera pose, actions, and basic metadata.
"""

import argparse
import shutil
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401  must be imported before isaaclab
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Generate stereo RGB-D HDF5 data from RoboLab.")
parser.add_argument("--task", default="BananaInBowlTask", help="RoboLab task name.")
parser.add_argument("--num-steps", type=int, default=60, help="Number of environment steps to run.")
parser.add_argument("--record-every", type=int, default=1, help="Save one HDF5 frame every N environment steps. Use 2 to halve the saved sampling rate.")
parser.add_argument("--skip-initial-seconds", type=float, default=1.0, help="Do not record the initial settling period, in seconds. Simulation still runs during this period.")
parser.add_argument("--env-step-dt", type=float, default=8.0 / 120.0, help="Approximate environment step duration in seconds, used for recording metadata and --skip-initial-seconds.")
parser.add_argument("--toggle-every", type=int, default=15, help="Gripper toggle period when no action HDF5 is provided.")
parser.add_argument("--action-hdf5", default=None, help="Replay actions from an HDF5 file instead of gripper-toggle actions.")
parser.add_argument("--action-demo", default="demo_0", help="Demo group name inside --action-hdf5, e.g. demo_0.")
parser.add_argument("--policy-backend", default="none", choices=["none", "pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"], help="Use an online policy server as the action source.")
parser.add_argument("--remote-host", default="localhost", help="Policy server host for pi0-family clients.")
parser.add_argument("--remote-port", type=int, default=8000, help="Policy server port for pi0-family clients.")
parser.add_argument("--remote-uri", default=None, help="Full policy server WebSocket URI; overrides host/port.")
parser.add_argument("--open-loop-horizon", type=int, default=None, help="Override policy client open-loop action horizon.")
parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to output/stereo_dataset/<task>.")
parser.add_argument("--file-name", default="run_0.hdf5", help="HDF5 file name.")
parser.add_argument("--output-format", default="hdf5", choices=["hdf5", "directory", "both"], help="Dataset output format. directory writes ROOT/<scene>/cameraXX with PNG/NPY/NPZ files.")
parser.add_argument("--scene-name", default=None, help="Scene directory name for --output-format directory/both. Defaults to --file-name stem.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the target HDF5 file if it exists.")
parser.add_argument("--width", type=int, default=1280, help="Camera width.")
parser.add_argument("--height", type=int, default=720, help="Camera height.")
parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"], help="HDF5 compression.")
parser.add_argument("--baseline", type=float, default=0.16, help="Stereo baseline in meters.")
parser.add_argument("--rig-x", type=float, default=1.5, help="Stereo rig center x position in world/env coordinates.")
parser.add_argument("--rig-y", type=float, default=0.0, help="Stereo rig center y position in world/env coordinates.")
parser.add_argument("--rig-z", type=float, default=1.0, help="Stereo rig center z position in world/env coordinates.")
parser.add_argument("--include-wrist-camera", action="store_true", help="Also record the robot-mounted wrist_cam as RGB-D.")
parser.add_argument("--include-wrist-stereo-camera", action="store_true", help="Also record a stereo RGB-D camera pair mounted on the robot wrist.")
parser.add_argument("--exclude-policy-wrist-camera-from-output", action="store_true", help="Use wrist_cam for policy inference, but do not save wrist_cam RGB-D/pose into the HDF5 output.")
parser.add_argument("--wrist-baseline", type=float, default=0.04, help="Wrist stereo baseline in meters.")
parser.add_argument("--randomize-background", action="store_true", help="Sample a non-default HDR/EXR background when registering the task.")
parser.add_argument("--background-seed", type=int, default=None, help="Seed for random background selection.")
parser.add_argument("--add-geometry-background", action="store_true", help="Add static USD geometry floor/walls/cabinets/shelves so background pixels have valid depth.")
parser.add_argument("--geometry-background-seed", type=int, default=0, help="Seed for geometry background color and layout jitter.")
parser.add_argument("--geometry-background-style", default="room_shelves", help="Static geometric background layout. Built-ins: room, room_shelves, room_cabinets, corner_shelves, warehouse_racks, office_partition, kitchen_counter, clutter_wall, lab_benches, garage_storage, library_shelves, factory_bins, pegboard_tools, apartment_furniture, corridor_depth, stacked_crates, or procedural bg000-bg099.")
parser.add_argument("--geometry-background-scale", type=float, default=1.0, help="Uniform scale multiplier for generated room/background geometry dimensions.")
parser.add_argument("--lighting", default="sphere", choices=["sphere", "dim", "top", "left", "right", "front", "behind", "red", "green", "blue"], help="Lighting preset.")
parser.add_argument("--randomize-camera-pose", action="store_true", help="Randomize fixed stereo camera pose at reset.")
parser.add_argument("--camera-pos-range", type=float, default=0.03, help="Uniform +/- meters for stereo camera x/y/z pose jitter.")
parser.add_argument("--camera-rot-range", type=float, default=0.03, help="Uniform +/- radians for stereo camera roll/pitch/yaw jitter.")
parser.add_argument("--moving-camera-pose", action="store_true", help="Move the fixed stereo rig during the rollout while preserving left/right baseline.")
parser.add_argument("--camera-motion-mode", default="sine", choices=["sine", "orbit", "line", "lookat_line", "lookat_random", "table_full_random"], help="Moving camera trajectory. Use table_full_random for smooth random poses that keep the tabletop bounds fully visible.")
parser.add_argument("--camera-motion-x", type=float, default=0.0, help="Sinusoidal stereo rig x-motion amplitude in meters.")
parser.add_argument("--camera-motion-y", type=float, default=0.03, help="Sinusoidal stereo rig y-motion amplitude in meters.")
parser.add_argument("--camera-motion-z", type=float, default=0.02, help="Sinusoidal stereo rig z-motion amplitude in meters.")
parser.add_argument("--camera-motion-period", type=float, default=120.0, help="Camera motion period in environment steps.")
parser.add_argument("--camera-orbit-radius", type=float, default=1.25, help="Orbit-mode stereo rig distance from look-at target in meters.")
parser.add_argument("--camera-orbit-angle", type=float, default=140.0, help="Orbit-mode total sweep angle in degrees across the rollout.")
parser.add_argument("--camera-orbit-height", type=float, default=1.05, help="Orbit-mode stereo rig center height in meters.")
parser.add_argument("--camera-orbit-start-deg", type=float, default=-70.0, help="Orbit-mode start angle around the look-at target in degrees.")
parser.add_argument("--camera-orbit-randomize", action="store_true", help="Add smooth random jitter to orbit-mode camera motion for more natural reconstruction trajectories.")
parser.add_argument("--camera-orbit-sample-random", action="store_true", help="Randomly sample orbit start/sweep/radius/height/look-at once per run using --camera-orbit-jitter-seed.")
parser.add_argument("--camera-orbit-radius-range", nargs=2, type=float, default=(0.72, 1.05), metavar=("MIN", "MAX"), help="Sampled orbit radius range when --camera-orbit-sample-random is used.")
parser.add_argument("--camera-orbit-angle-range", nargs=2, type=float, default=(75.0, 135.0), metavar=("MIN", "MAX"), help="Sampled orbit sweep range in degrees.")
parser.add_argument("--camera-orbit-start-range", nargs=2, type=float, default=(-80.0, -25.0), metavar=("MIN", "MAX"), help="Sampled orbit start angle range in degrees.")
parser.add_argument("--camera-orbit-height-range", nargs=2, type=float, default=(0.82, 1.28), metavar=("MIN", "MAX"), help="Sampled orbit height range in meters.")
parser.add_argument("--camera-look-at-x-range", nargs=2, type=float, default=(0.30, 0.62), metavar=("MIN", "MAX"), help="Sampled look-at x range when random orbit sampling is enabled.")
parser.add_argument("--camera-look-at-y-range", nargs=2, type=float, default=(-0.18, 0.18), metavar=("MIN", "MAX"), help="Sampled look-at y range when random orbit sampling is enabled.")
parser.add_argument("--camera-look-at-z-range", nargs=2, type=float, default=(0.62, 0.90), metavar=("MIN", "MAX"), help="Sampled look-at z range when random orbit sampling is enabled.")
parser.add_argument("--camera-orbit-jitter-seed", type=int, default=0, help="Seed for smooth orbit camera jitter.")
parser.add_argument("--camera-orbit-radius-jitter", type=float, default=0.08, help="Orbit-mode smooth radius jitter amplitude in meters.")
parser.add_argument("--camera-orbit-height-jitter", type=float, default=0.08, help="Orbit-mode smooth height jitter amplitude in meters.")
parser.add_argument("--camera-orbit-angle-jitter", type=float, default=8.0, help="Orbit-mode smooth angle jitter amplitude in degrees.")
parser.add_argument("--camera-look-at-jitter", type=float, default=0.06, help="Orbit-mode smooth look-at target jitter amplitude in meters.")
parser.add_argument("--camera-look-at", nargs=3, type=float, default=(0.45, 0.0, 0.72), metavar=("X", "Y", "Z"), help="Orbit-mode target point to keep in view.")
parser.add_argument("--camera-table-anchor-frames", action="store_true", help="Blend in several table-facing anchor moments during orbit motion for reconstruction-friendly views.")
parser.add_argument("--camera-table-anchor-progress", nargs="+", type=float, default=None, help="Optional fixed rollout progress values in [0, 1] where the global stereo rig should face the table. If omitted, positions are sampled randomly per run.")
parser.add_argument("--camera-table-anchor-count", type=int, default=3, help="Number of random table-facing anchor moments when --camera-table-anchor-progress is omitted.")
parser.add_argument("--camera-table-anchor-progress-range", nargs=2, type=float, default=(0.12, 0.88), metavar=("MIN", "MAX"), help="Progress range used for random table-facing anchor sampling.")
parser.add_argument("--camera-table-anchor-min-separation", type=float, default=0.18, help="Minimum progress separation between random table-facing anchors.")
parser.add_argument("--camera-table-anchor-width", type=float, default=0.07, help="Half-width in rollout progress for each table-facing anchor blend.")
parser.add_argument("--camera-table-anchor-pos", nargs=3, type=float, default=(0.95, 0.0, 1.15), metavar=("X", "Y", "Z"), help="Stereo rig center position for table-facing anchor frames.")
parser.add_argument("--camera-table-anchor-look-at", nargs=3, type=float, default=(0.45, 0.0, 0.72), metavar=("X", "Y", "Z"), help="Look-at target for table-facing anchor frames.")
parser.add_argument("--camera-line-start", nargs=3, type=float, default=(1.35, -0.95, 0.85), metavar=("X", "Y", "Z"), help="Line/lookat_line start position for the stereo rig center.")
parser.add_argument("--camera-line-end", nargs=3, type=float, default=(1.35, 0.95, 1.35), metavar=("X", "Y", "Z"), help="Line/lookat_line end position for the stereo rig center.")
parser.add_argument("--camera-random-keyframes", type=int, default=5, help="Number of smooth random keyframes for lookat_random camera motion.")
parser.add_argument("--camera-random-x-range", nargs=2, type=float, default=(0.55, 0.95), metavar=("MIN", "MAX"), help="lookat_random stereo rig center x range.")
parser.add_argument("--camera-random-y-range", nargs=2, type=float, default=(-0.45, 0.45), metavar=("MIN", "MAX"), help="lookat_random stereo rig center y range.")
parser.add_argument("--camera-random-z-range", nargs=2, type=float, default=(0.70, 1.20), metavar=("MIN", "MAX"), help="lookat_random stereo rig center z range.")
parser.add_argument("--camera-random-look-at-x-range", nargs=2, type=float, default=(0.30, 0.62), metavar=("MIN", "MAX"), help="lookat_random table target x range.")
parser.add_argument("--camera-random-look-at-y-range", nargs=2, type=float, default=(-0.20, 0.20), metavar=("MIN", "MAX"), help="lookat_random table target y range.")
parser.add_argument("--camera-random-look-at-z-range", nargs=2, type=float, default=(0.55, 0.90), metavar=("MIN", "MAX"), help="lookat_random table target z range.")
parser.add_argument("--camera-random-start-table", action="store_true", help="Force the first lookat_random keyframe to a stable table-facing view.")
parser.add_argument("--camera-random-start-pos", nargs=3, type=float, default=(0.70, 0.0, 0.95), metavar=("X", "Y", "Z"), help="Initial stereo rig center when --camera-random-start-table is used.")
parser.add_argument("--camera-random-start-look-at", nargs=3, type=float, default=(0.45, 0.0, 0.72), metavar=("X", "Y", "Z"), help="Initial table target when --camera-random-start-table is used.")
parser.add_argument("--camera-table-bounds-min", nargs=3, type=float, default=(0.05, -0.55, 0.50), metavar=("X", "Y", "Z"), help="Minimum xyz corner of the tabletop/operation volume that must stay fully visible in table_full_random mode.")
parser.add_argument("--camera-table-bounds-max", nargs=3, type=float, default=(0.90, 0.55, 0.95), metavar=("X", "Y", "Z"), help="Maximum xyz corner of the tabletop/operation volume that must stay fully visible in table_full_random mode.")
parser.add_argument("--camera-table-full-margin", type=float, default=0.04, help="Fractional angular margin kept around the table bounds in table_full_random mode.")
parser.add_argument("--camera-table-full-vertical-fov-deg", type=float, default=75.0, help="Approximate vertical camera FOV used for table_full_random visibility constraints.")
parser.add_argument("--camera-table-full-azimuth-range", nargs=2, type=float, default=(-75.0, 75.0), metavar=("MIN", "MAX"), help="Random azimuth range in degrees around the table for table_full_random keyframes.")
parser.add_argument("--camera-table-full-elevation-range", nargs=2, type=float, default=(28.0, 62.0), metavar=("MIN", "MAX"), help="Random elevation range in degrees for table_full_random keyframes.")
parser.add_argument("--camera-table-full-distance-range", nargs=2, type=float, default=(0.55, 0.95), metavar=("MIN", "MAX"), help="Initial random camera distance range before visibility correction in table_full_random mode.")
parser.add_argument("--camera-table-full-look-at-jitter", type=float, default=0.10, help="Target jitter around table center for table_full_random keyframes.")
parser.add_argument("--camera-table-full-max-retreat", type=float, default=1.50, help="Maximum camera distance after backing up to keep table bounds visible.")
parser.add_argument("--camera-table-full-distance-padding", type=float, default=0.04, help="Extra distance beyond the minimum full-table distance in table_full_random mode.")
parser.add_argument("--camera-table-full-distance-change-per-step", type=float, default=0.008, help="Maximum distance envelope change per env step for smooth table_full_random zoom motion.")
parser.add_argument("--randomize-object-pose", action="store_true", help="Randomize listed object initial poses at reset.")
parser.add_argument("--randomize-objects", nargs="+", default=None, help="Scene object names to randomize, e.g. banana bowl. Defaults to banana bowl for BananaInBowlTask.")
parser.add_argument("--object-xy-range", type=float, default=0.05, help="Uniform +/- meters for randomized object x/y offsets.")
parser.add_argument("--object-yaw-range", type=float, default=0.5, help="Uniform +/- radians for randomized object yaw.")
parser.add_argument("--add-random-distractors", action="store_true", help="Add random clutter objects to the table after reset.")
parser.add_argument("--distractor-mode", default="real", choices=["real", "geometry"], help="Use realistic USD assets or simple geometry for random distractors.")
parser.add_argument("--distractor-assets", nargs="+", default=None, help="Optional USD asset paths or names to sample for real distractors.")
parser.add_argument("--distractor-scale-range", nargs=2, type=float, default=(0.85, 1.15), metavar=("MIN", "MAX"), help="Random scale range for real USD distractor assets.")
parser.add_argument("--distractor-mass-range", nargs=2, type=float, default=(0.05, 0.35), metavar=("MIN", "MAX"), help="Random mass range in kg for physical real distractors.")
parser.add_argument("--distractor-collider-approx", default="convexHull", choices=["convexHull", "convexDecomposition", "boundingCube", "boundingSphere", "none"], help="Collider approximation for real USD distractors.")
parser.add_argument("--distractor-settle-steps", type=int, default=0, help="Reserved for future physics settling; currently objects are placed by bounding-box contact.")
parser.add_argument("--num-distractors", type=int, default=12, help="Number of random distractor objects to add.")
parser.add_argument("--distractor-seed", type=int, default=0, help="Seed for random distractor placement/colors.")
parser.add_argument("--distractor-x-range", nargs=2, type=float, default=(0.18, 0.82), metavar=("MIN", "MAX"), help="Distractor x placement range.")
parser.add_argument("--distractor-y-range", nargs=2, type=float, default=(-0.42, 0.42), metavar=("MIN", "MAX"), help="Distractor y placement range.")
parser.add_argument("--distractor-table-z", type=float, default=0.08, help="Approximate tabletop z coordinate for distractors.")
parser.add_argument("--distractor-size-range", nargs=2, type=float, default=(0.035, 0.09), metavar=("MIN", "MAX"), help="Distractor size range in meters.")
parser.add_argument("--num_envs", type=int, default=1, help="Keep this at 1 for the first 4090 run.")
AppLauncher.add_app_launcher_args(parser)

args_cli, unknown_args = parser.parse_known_args()
if unknown_args:
    parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
args_cli.enable_cameras = True
args_cli.save_videos = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.core.events.reset_camera import reset_camera_pose_uniform  # noqa: E402
from robolab.core.events.reset_pose import reset_pose_uniform  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402


@configclass
class StereoLeftRGDBCameraCfg:
    stereo_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/stereo_left_camera",
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(args_cli.rig_x, args_cli.rig_y + args_cli.baseline / 2.0, args_cli.rig_z),
            rot=(0.0, 0.474, 0.0, 0.881),
            convention="opengl",
        ),
    )


@configclass
class StereoRightRGDBCameraCfg:
    stereo_right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/stereo_right_camera",
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(args_cli.rig_x, args_cli.rig_y - args_cli.baseline / 2.0, args_cli.rig_z),
            rot=(0.0, 0.474, 0.0, 0.881),
            convention="opengl",
        ),
    )


@configclass
class WristLeftRGDBCameraCfg:
    wrist_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_left_camera",
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031 + args_cli.wrist_baseline / 2.0, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )


@configclass
class WristRightRGDBCameraCfg:
    wrist_right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_right_camera",
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031 - args_cli.wrist_baseline / 2.0, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )


BASE_CAMERA_NAMES = ("stereo_left_camera", "stereo_right_camera")
WRIST_STEREO_CAMERA_NAMES = ("wrist_left_camera", "wrist_right_camera")
DEFAULT_REAL_DISTRACTOR_ASSETS = (
    "assets/objects/hope/bbq_sauce_bottle.usd",
    "assets/objects/hope/canned_mushrooms.usd",
    "assets/objects/hope/canned_peaches.usd",
    "assets/objects/hope/canned_tuna.usd",
    "assets/objects/hope/corn_can.usd",
    "assets/objects/hope/granola_bars.usd",
    "assets/objects/hope/ketchup_bottle.usd",
    "assets/objects/hope/macaroni_and_cheese.usd",
    "assets/objects/hope/mayonnaise_bottle.usd",
    "assets/objects/hope/milk_carton.usd",
    "assets/objects/hope/mustard_bottle.usd",
    "assets/objects/hope/oatmeal_raisin_cookies.usd",
    "assets/objects/hope/orange_juice_carton.usd",
    "assets/objects/hope/popcorn_box.usd",
    "assets/objects/hope/raisin_box.usd",
    "assets/objects/hope/spaghetti.usd",
    "assets/objects/hope/tomato_sauce_can.usd",
    "assets/objects/hope/yogurt_cup.usd",
    "assets/objects/hot3d/ceramic_mug.usd",
    "assets/objects/hot3d/computer_mouse.usd",
    "assets/objects/hot3d/keyboard.usd",
    "assets/objects/hot3d/remote_control.usd",
    "assets/objects/hot3d/smartphone.usd",
    "assets/objects/hot3d/soup_can.usd",
    "assets/objects/hot3d/wooden_bowl.usd",
    "assets/objects/objaverse/apple_01.usd",
    "assets/objects/objaverse/bagel_00.usd",
    "assets/objects/objaverse/gregorys_coffee_cup.usd",
    "assets/objects/objaverse/snickers_bar.usd",
)


def _clean_name(text: str) -> str:
    return re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_from_sensor(sensor, key: str, env_id: int = 0) -> np.ndarray:
    if key not in sensor.data.output:
        raise KeyError(f"Camera '{sensor.cfg.prim_path}' has no '{key}' output. Available: {list(sensor.data.output)}")
    arr = _to_numpy(sensor.data.output[key][env_id])
    if key == "rgb" and arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255)
        arr = arr.astype(np.uint8)
    if key == "depth":
        arr = arr.astype(np.float32)
    return arr


def _append_dataset(group, name: str, frame: np.ndarray, compression):
    frame = np.expand_dims(frame, axis=0)
    if name not in group:
        kwargs = {"maxshape": (None,) + frame.shape[1:], "chunks": True}
        if compression != "none":
            kwargs["compression"] = compression
        group.create_dataset(name, data=frame, **kwargs)
    else:
        ds = group[name]
        old_size = ds.shape[0]
        ds.resize(old_size + 1, axis=0)
        ds[old_size] = frame[0]


def _capture_camera_poses(sensors: dict[str, object], camera_names: tuple[str, ...]):
    return {
        name: (
            sensors[name].data.pos_w.clone(),
            sensors[name].data.quat_w_ros.clone(),
        )
        for name in camera_names
    }


def _sample_orbit_params():
    rng = np.random.default_rng(args_cli.camera_orbit_jitter_seed)
    params = {
        "radius": args_cli.camera_orbit_radius,
        "angle": args_cli.camera_orbit_angle,
        "start_deg": args_cli.camera_orbit_start_deg,
        "height": args_cli.camera_orbit_height,
        "look_at": tuple(args_cli.camera_look_at),
    }
    if args_cli.camera_orbit_sample_random:
        angle = float(rng.uniform(*args_cli.camera_orbit_angle_range))
        start_min, start_max = args_cli.camera_orbit_start_range
        start = float(rng.uniform(start_min, start_max))
        # Keep the sampled sweep from drifting too far behind the robot by clamping the end angle.
        end = max(min(start + angle, 85.0), -85.0)
        angle = end - start
        params = {
            "radius": float(rng.uniform(*args_cli.camera_orbit_radius_range)),
            "angle": angle,
            "start_deg": start,
            "height": float(rng.uniform(*args_cli.camera_orbit_height_range)),
            "look_at": (
                float(rng.uniform(*args_cli.camera_look_at_x_range)),
                float(rng.uniform(*args_cli.camera_look_at_y_range)),
                float(rng.uniform(*args_cli.camera_look_at_z_range)),
            ),
        }
    return params


def _sample_table_anchor_progress():
    if not args_cli.camera_table_anchor_frames:
        return []
    if args_cli.camera_table_anchor_progress is not None:
        values = [float(v) for v in args_cli.camera_table_anchor_progress]
        return sorted(max(0.0, min(1.0, v)) for v in values)

    count = max(0, int(args_cli.camera_table_anchor_count))
    if count == 0:
        return []

    lo, hi = args_cli.camera_table_anchor_progress_range
    lo = max(0.0, min(1.0, float(lo)))
    hi = max(0.0, min(1.0, float(hi)))
    if hi < lo:
        lo, hi = hi, lo

    min_sep = max(0.0, float(args_cli.camera_table_anchor_min_separation))
    rng = np.random.default_rng(args_cli.camera_orbit_jitter_seed + 7919)
    anchors = []
    max_attempts = 2000
    for _ in range(max_attempts):
        if len(anchors) >= count:
            break
        candidate = float(rng.uniform(lo, hi))
        if all(abs(candidate - existing) >= min_sep for existing in anchors):
            anchors.append(candidate)

    if len(anchors) < count:
        available = max(hi - lo, 1.0e-6)
        feasible_sep = min(min_sep, available / max(count - 1, 1))
        anchors = [lo + available * (i + 0.5) / count for i in range(count)]
        rng.shuffle(anchors)
        jitter = min(feasible_sep * 0.25, available / max(count, 1) * 0.25)
        anchors = [max(lo, min(hi, a + float(rng.uniform(-jitter, jitter)))) for a in anchors]

    return sorted(anchors[:count])


def _sample_lookat_random_params():
    rng = np.random.default_rng(args_cli.camera_orbit_jitter_seed + 104729)
    count = max(2, int(args_cli.camera_random_keyframes))

    def sample_points(x_range, y_range, z_range):
        return np.stack(
            [
                rng.uniform(*x_range, size=count),
                rng.uniform(*y_range, size=count),
                rng.uniform(*z_range, size=count),
            ],
            axis=1,
        ).astype(np.float32)

    centers = sample_points(args_cli.camera_random_x_range, args_cli.camera_random_y_range, args_cli.camera_random_z_range)
    targets = sample_points(
        args_cli.camera_random_look_at_x_range,
        args_cli.camera_random_look_at_y_range,
        args_cli.camera_random_look_at_z_range,
    )

    if args_cli.camera_random_start_table:
        centers[0] = np.asarray(args_cli.camera_random_start_pos, dtype=np.float32)
        targets[0] = np.asarray(args_cli.camera_random_start_look_at, dtype=np.float32)

    return {
        "centers": centers.tolist(),
        "targets": targets.tolist(),
    }


def _sample_table_full_random_params():
    rng = np.random.default_rng(args_cli.camera_orbit_jitter_seed + 524287)
    count = max(2, int(args_cli.camera_random_keyframes))
    bounds_min = np.asarray(args_cli.camera_table_bounds_min, dtype=np.float32)
    bounds_max = np.asarray(args_cli.camera_table_bounds_max, dtype=np.float32)
    table_center = (bounds_min + bounds_max) * 0.5
    key_centers = []
    key_targets = []
    key_distances = []
    for _ in range(count):
        target = table_center.copy()
        target[0] += float(rng.uniform(-args_cli.camera_table_full_look_at_jitter, args_cli.camera_table_full_look_at_jitter))
        target[1] += float(rng.uniform(-args_cli.camera_table_full_look_at_jitter, args_cli.camera_table_full_look_at_jitter))
        target[2] += float(rng.uniform(-args_cli.camera_table_full_look_at_jitter * 0.5, args_cli.camera_table_full_look_at_jitter * 0.5))
        azimuth = math.radians(float(rng.uniform(*args_cli.camera_table_full_azimuth_range)))
        elevation = math.radians(float(rng.uniform(*args_cli.camera_table_full_elevation_range)))
        distance = float(rng.uniform(*args_cli.camera_table_full_distance_range))
        direction = np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        key_centers.append((target + direction * distance).tolist())
        key_targets.append(target.tolist())
        key_distances.append(distance)

    if args_cli.camera_random_start_table:
        key_centers[0] = list(args_cli.camera_random_start_pos)
        key_targets[0] = list(args_cli.camera_random_start_look_at)
        key_distances[0] = float(np.linalg.norm(np.asarray(key_centers[0], dtype=np.float32) - np.asarray(key_targets[0], dtype=np.float32)))

    def catmull_np(points, progress: float):
        pts = np.asarray(points, dtype=np.float32)
        n = int(pts.shape[0])
        if n == 1:
            return pts[0]
        scaled = max(0.0, min(1.0, progress)) * (n - 1)
        i1 = min(int(math.floor(scaled)), n - 2)
        t = float(scaled - i1)
        i0 = max(i1 - 1, 0)
        i2 = i1 + 1
        i3 = min(i2 + 1, n - 1)
        p0, p1, p2, p3 = pts[i0], pts[i1], pts[i2], pts[i3]
        tt = t * t
        ttt = tt * t
        return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * ttt)

    def required_distance_np(target, direction):
        vertical_fov = math.radians(max(1.0, args_cli.camera_table_full_vertical_fov_deg))
        aspect = max(float(args_cli.width) / max(float(args_cli.height), 1.0), 1.0e-6)
        horizontal_fov = 2.0 * math.atan(math.tan(vertical_fov * 0.5) * aspect)
        margin = max(0.0, min(0.45, args_cli.camera_table_full_margin))
        tan_x = math.tan(horizontal_fov * 0.5 * (1.0 - margin))
        tan_y = math.tan(vertical_fov * 0.5 * (1.0 - margin))
        corners = np.asarray(
            [[x, y, z] for x in (bounds_min[0], bounds_max[0]) for y in (bounds_min[1], bounds_max[1]) for z in (bounds_min[2], bounds_max[2])],
            dtype=np.float32,
        )
        forward = -direction / max(float(np.linalg.norm(direction)), 1.0e-6)
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(up, forward)
        if np.linalg.norm(side) < 1.0e-6:
            side = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            side = side / np.linalg.norm(side)
        camera_up = np.cross(forward, side)
        camera_up = camera_up / max(float(np.linalg.norm(camera_up)), 1.0e-6)
        rel = corners - target[None, :]
        f = rel @ forward
        sx = rel @ side
        sy = rel @ camera_up
        # Camera center is target + direction * d, so corner depth is f + d.
        req = float(np.max(np.maximum(np.abs(sx) / max(tan_x, 1.0e-6) - f, np.abs(sy) / max(tan_y, 1.0e-6) - f)))
        req = max(req, 0.25)
        return min(req + args_cli.camera_table_full_distance_padding, args_cli.camera_table_full_max_retreat)

    dense_count = max(2, int(args_cli.num_steps))
    raw_centers = []
    dense_targets = []
    desired_distances = []
    directions = []
    for i in range(dense_count):
        progress = 0.0 if dense_count <= 1 else i / float(dense_count - 1)
        raw_center = catmull_np(key_centers, progress)
        target = catmull_np(key_targets, progress)
        direction = raw_center - target
        direction_norm = max(float(np.linalg.norm(direction)), 1.0e-6)
        direction = direction / direction_norm
        req_distance = required_distance_np(target, direction)
        sampled_distance = min(direction_norm, req_distance + 0.18)
        desired_distance = max(req_distance, sampled_distance)
        raw_centers.append(raw_center.tolist())
        dense_targets.append(target.tolist())
        desired_distances.append(desired_distance)
        directions.append(direction.tolist())

    distances = np.asarray(desired_distances, dtype=np.float32)
    max_delta = max(float(args_cli.camera_table_full_distance_change_per_step), 1.0e-5)
    # Build a Lipschitz upper envelope so required backing distance changes smoothly and never dips below the full-view requirement.
    for i in range(1, dense_count):
        distances[i] = max(distances[i], distances[i - 1] - max_delta)
    for i in range(dense_count - 2, -1, -1):
        distances[i] = max(distances[i], distances[i + 1] - max_delta)
    distances = np.minimum(distances, float(args_cli.camera_table_full_max_retreat))

    dense_centers = []
    for target, direction, distance in zip(dense_targets, directions, distances):
        target_np = np.asarray(target, dtype=np.float32)
        direction_np = np.asarray(direction, dtype=np.float32)
        dense_centers.append((target_np + direction_np * float(distance)).tolist())

    return {
        "centers": dense_centers,
        "targets": dense_targets,
        "key_centers": key_centers,
        "key_targets": key_targets,
        "desired_distances": [float(v) for v in desired_distances],
        "smoothed_distances": [float(v) for v in distances.tolist()],
    }


def _table_bounds_corners(device):
    mn = torch.tensor(args_cli.camera_table_bounds_min, device=device, dtype=torch.float32)
    mx = torch.tensor(args_cli.camera_table_bounds_max, device=device, dtype=torch.float32)
    corners = []
    for x in (mn[0], mx[0]):
        for y in (mn[1], mx[1]):
            for z in (mn[2], mx[2]):
                corners.append(torch.stack([x, y, z]))
    return torch.stack(corners, dim=0)


def _fit_table_full_view(center, target):
    device = center.device
    corners = _table_bounds_corners(device)
    vertical_fov = math.radians(max(1.0, args_cli.camera_table_full_vertical_fov_deg))
    aspect = max(float(args_cli.width) / max(float(args_cli.height), 1.0), 1.0e-6)
    horizontal_fov = 2.0 * math.atan(math.tan(vertical_fov * 0.5) * aspect)
    margin = max(0.0, min(0.45, args_cli.camera_table_full_margin))
    max_x_angle = horizontal_fov * 0.5 * (1.0 - margin)
    max_y_angle = vertical_fov * 0.5 * (1.0 - margin)

    forward = target - center
    forward_norm = torch.linalg.norm(forward)
    if float(forward_norm.detach().cpu()) < 1.0e-6:
        return center, target
    forward = forward / forward_norm
    up_world = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    distance = float(forward_norm.detach().cpu())
    max_distance = max(distance, args_cli.camera_table_full_max_retreat)
    for _ in range(64):
        side = torch.linalg.cross(up_world, forward)
        side_norm = torch.linalg.norm(side)
        if float(side_norm.detach().cpu()) < 1.0e-6:
            side = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        else:
            side = side / side_norm
        camera_up = torch.linalg.cross(forward, side)
        camera_up = camera_up / torch.linalg.norm(camera_up)

        rel = corners - center.unsqueeze(0)
        z = rel @ forward
        x = rel @ side
        y = rel @ camera_up
        if bool(torch.all(z > 1.0e-4).detach().cpu()):
            x_angles = torch.atan2(torch.abs(x), z)
            y_angles = torch.atan2(torch.abs(y), z)
            if bool((torch.max(x_angles) <= max_x_angle and torch.max(y_angles) <= max_y_angle).detach().cpu()):
                return center, target

        distance = min(distance * 1.08 + 0.03, max_distance)
        center = target - forward * distance
        if distance >= max_distance - 1.0e-5:
            return center, target

    return center, target


def _catmull_rom(points, progress: float, device):
    pts = torch.tensor(points, device=device, dtype=torch.float32)
    n = int(pts.shape[0])
    if n == 1:
        return pts[0]
    scaled = max(0.0, min(1.0, progress)) * (n - 1)
    i1 = min(int(math.floor(scaled)), n - 2)
    t = float(scaled - i1)
    i0 = max(i1 - 1, 0)
    i2 = i1 + 1
    i3 = min(i2 + 1, n - 1)
    p0, p1, p2, p3 = pts[i0], pts[i1], pts[i2], pts[i3]
    tt = t * t
    ttt = tt * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * ttt
    )


def _set_parallel_stereo_view(sensors: dict[str, object], center, target):
    forward = target - center
    forward_norm = torch.linalg.norm(forward)
    if float(forward_norm.detach().cpu()) < 1.0e-6:
        forward = torch.tensor([-1.0, 0.0, -0.2], device=center.device, dtype=torch.float32)
    else:
        forward = forward / forward_norm
    up = torch.tensor([0.0, 0.0, 1.0], device=center.device, dtype=torch.float32)
    side = torch.linalg.cross(up, forward)
    side_norm = torch.linalg.norm(side)
    if float(side_norm.detach().cpu()) < 1.0e-6:
        side = torch.tensor([0.0, 1.0, 0.0], device=center.device, dtype=torch.float32)
    else:
        side = side / side_norm
    half_baseline = args_cli.baseline / 2.0
    positions = {
        "stereo_left_camera": center + side * half_baseline,
        "stereo_right_camera": center - side * half_baseline,
    }
    for name in BASE_CAMERA_NAMES:
        eye = positions[name].unsqueeze(0)
        target_parallel = (positions[name] + forward).unsqueeze(0)
        sensors[name].set_world_poses_from_view(eyes=eye, targets=target_parallel)


def _apply_stereo_rig_motion(sensors: dict[str, object], base_poses, step: int, orbit_params=None, table_anchor_progress=None, lookat_random_params=None, table_full_random_params=None):
    if not args_cli.moving_camera_pose:
        return
    progress = 0.0 if args_cli.num_steps <= 1 else step / float(args_cli.num_steps - 1)
    if args_cli.camera_motion_mode == "orbit":
        orbit_params = orbit_params or _sample_orbit_params()
        angle_jitter = 0.0
        radius_jitter = 0.0
        height_jitter = 0.0
        look_at_jitter = torch.zeros(3, device=sensors["stereo_left_camera"].device, dtype=torch.float32)
        if args_cli.camera_orbit_randomize:
            seed_phase = float(args_cli.camera_orbit_jitter_seed) * 1.61803398875
            angle_jitter = args_cli.camera_orbit_angle_jitter * math.sin(2.0 * math.pi * progress * 1.7 + seed_phase)
            radius_jitter = args_cli.camera_orbit_radius_jitter * math.sin(2.0 * math.pi * progress * 2.3 + seed_phase * 0.7)
            height_jitter = args_cli.camera_orbit_height_jitter * math.sin(2.0 * math.pi * progress * 1.3 + seed_phase * 1.1)
            look_at_jitter = torch.tensor(
                [
                    args_cli.camera_look_at_jitter * math.sin(2.0 * math.pi * progress * 1.1 + seed_phase * 0.3),
                    args_cli.camera_look_at_jitter * math.sin(2.0 * math.pi * progress * 1.5 + seed_phase * 0.5),
                    args_cli.camera_look_at_jitter * 0.5 * math.sin(2.0 * math.pi * progress * 0.9 + seed_phase * 0.9),
                ],
                device=sensors["stereo_left_camera"].device,
                dtype=torch.float32,
            )
        theta = math.radians(orbit_params["start_deg"] + orbit_params["angle"] * progress + angle_jitter)
        target = torch.tensor(orbit_params["look_at"], device=sensors["stereo_left_camera"].device, dtype=torch.float32) + look_at_jitter
        radius = max(0.35, orbit_params["radius"] + radius_jitter)
        center = torch.tensor(
            [
                target[0] + radius * math.cos(theta),
                target[1] + radius * math.sin(theta),
                orbit_params["height"] + height_jitter,
            ],
            device=sensors["stereo_left_camera"].device,
            dtype=torch.float32,
        )
        if args_cli.camera_table_anchor_frames:
            anchor_weight = 0.0
            width = max(args_cli.camera_table_anchor_width, 1.0e-6)
            for anchor_progress in table_anchor_progress or []:
                distance = abs(progress - anchor_progress)
                if distance < width:
                    local = 0.5 + 0.5 * math.cos(math.pi * distance / width)
                    anchor_weight = max(anchor_weight, local)
            if anchor_weight > 0.0:
                anchor_center = torch.tensor(args_cli.camera_table_anchor_pos, device=center.device, dtype=torch.float32)
                anchor_target = torch.tensor(args_cli.camera_table_anchor_look_at, device=center.device, dtype=torch.float32)
                center = center * (1.0 - anchor_weight) + anchor_center * anchor_weight
                target = target * (1.0 - anchor_weight) + anchor_target * anchor_weight
        _set_parallel_stereo_view(sensors, center, target)
        return
    if args_cli.camera_motion_mode == "lookat_random":
        device = sensors["stereo_left_camera"].device
        params = lookat_random_params or _sample_lookat_random_params()
        center = _catmull_rom(params["centers"], progress, device)
        target = _catmull_rom(params["targets"], progress, device)
        _set_parallel_stereo_view(sensors, center, target)
        return
    if args_cli.camera_motion_mode == "table_full_random":
        device = sensors["stereo_left_camera"].device
        params = table_full_random_params or _sample_table_full_random_params()
        dense_len = max(1, len(params["centers"]))
        dense_idx = min(max(int(round(progress * (dense_len - 1))), 0), dense_len - 1)
        center = torch.tensor(params["centers"][dense_idx], device=device, dtype=torch.float32)
        target = torch.tensor(params["targets"][dense_idx], device=device, dtype=torch.float32)
        _set_parallel_stereo_view(sensors, center, target)
        return
    if args_cli.camera_motion_mode in {"line", "lookat_line"}:
        device = sensors["stereo_left_camera"].device
        start = torch.tensor(args_cli.camera_line_start, device=device, dtype=torch.float32)
        end = torch.tensor(args_cli.camera_line_end, device=device, dtype=torch.float32)
        center = start + (end - start) * progress
        if args_cli.camera_motion_mode == "line":
            base_left, _ = base_poses["stereo_left_camera"]
            base_right, _ = base_poses["stereo_right_camera"]
            base_center = (base_left + base_right) * 0.5
            delta = center.unsqueeze(0) - base_center
            for name in BASE_CAMERA_NAMES:
                base_pos, base_quat = base_poses[name]
                sensors[name].set_world_poses(positions=base_pos + delta, orientations=base_quat)
        else:
            target = torch.tensor(args_cli.camera_look_at, device=device, dtype=torch.float32)
            forward = target - center
            forward = forward / torch.linalg.norm(forward)
            up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            side = torch.linalg.cross(up, forward)
            side = side / torch.linalg.norm(side)
            half_baseline = args_cli.baseline / 2.0
            positions = {
                "stereo_left_camera": center + side * half_baseline,
                "stereo_right_camera": center - side * half_baseline,
            }
            for name in BASE_CAMERA_NAMES:
                eye = positions[name].unsqueeze(0)
                target_parallel = (positions[name] + forward).unsqueeze(0)
                sensors[name].set_world_poses_from_view(eyes=eye, targets=target_parallel)
        return

    period = max(args_cli.camera_motion_period, 1.0)
    phase = 2.0 * math.pi * (step / period)
    delta = torch.tensor(
        [
            args_cli.camera_motion_x * math.sin(phase),
            args_cli.camera_motion_y * math.sin(phase),
            args_cli.camera_motion_z * math.sin(phase + math.pi / 2.0),
        ],
        device=sensors["stereo_left_camera"].device,
        dtype=torch.float32,
    )
    for name in BASE_CAMERA_NAMES:
        base_pos, base_quat = base_poses[name]
        sensors[name].set_world_poses(positions=base_pos + delta.unsqueeze(0), orientations=base_quat)


def _load_replay_actions():
    if args_cli.action_hdf5 is None:
        return None
    with h5py.File(args_cli.action_hdf5, "r") as f:
        key = f"data/{args_cli.action_demo}/actions"
        if key not in f:
            raise KeyError(f"{key} not found in {args_cli.action_hdf5}")
        actions = np.asarray(f[key], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, A), got {actions.shape}")
    print(f"[INFO] Loaded replay actions from {args_cli.action_hdf5}:{args_cli.action_demo}, shape={actions.shape}")
    return actions


def _make_policy_client():
    if args_cli.policy_backend == "none":
        return None
    if args_cli.action_hdf5 is not None:
        raise ValueError("Use either --policy-backend or --action-hdf5, not both.")
    from policies.pi0_family.client import Pi0DroidJointposClient

    print(f"[INFO] Connecting policy backend {args_cli.policy_backend}...")
    return Pi0DroidJointposClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        remote_uri=args_cli.remote_uri,
        open_loop_horizon=args_cli.open_loop_horizon,
        policy_variant=args_cli.policy_backend,
    )



def _write_json_dataset(group, name: str, obj):
    payload = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    group.create_dataset(name, data=np.void(payload))


def _decode_json_dataset(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, np.void):
        payload = value.tobytes()
    else:
        payload = bytes(value)
    return json.loads(payload.decode("utf-8"))


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _camera_to_world_from_pose(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    mats = np.zeros((len(position), 4, 4), dtype=np.float32)
    mats[:, 3, 3] = 1.0
    for i, (pos, quat) in enumerate(zip(position, quat_wxyz)):
        mats[i, :3, :3] = _quat_wxyz_to_matrix(quat)
        mats[i, :3, 3] = np.asarray(pos, dtype=np.float32)
    return mats


def _intrinsics_for_camera(camera_name: str, width: int, height: int) -> np.ndarray:
    focal_length = 2.8 if camera_name.startswith("wrist_") else 2.1
    horizontal_aperture = 5.376
    vertical_aperture = 3.024
    fx = focal_length / horizontal_aperture * float(width)
    fy = focal_length / vertical_aperture * float(height)
    return np.asarray([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def _stereo_pairs_for_directory(record_camera_names: tuple[str, ...]) -> list[tuple[str, str, str, str]]:
    names = set(record_camera_names)
    pairs = []
    if {"stereo_left_camera", "stereo_right_camera"}.issubset(names):
        pairs.append(("camera00", "stereo_left_camera", "stereo_right_camera", "global_stereo"))
    if {"wrist_left_camera", "wrist_right_camera"}.issubset(names):
        pairs.append((f"camera{len(pairs):02d}", "wrist_left_camera", "wrist_right_camera", "wrist_stereo"))
    if not pairs:
        raise ValueError("Directory output requires at least one complete stereo pair in recorded cameras.")
    return pairs


def _write_directory_pair(h5, scene_dir: Path, camera_dir_name: str, left_name: str, right_name: str, pair_type: str, meta: dict):
    sequence_dir = scene_dir / camera_dir_name
    for rel in ("rgb/left", "rgb/right", "depth/left", "depth/right", "camera"):
        (sequence_dir / rel).mkdir(parents=True, exist_ok=True)

    demo = h5["data/demo_0"]
    left_rgb = demo[f"obs/{left_name}/rgb"]
    right_rgb = demo[f"obs/{right_name}/rgb"]
    left_depth = demo[f"obs/{left_name}/depth"]
    right_depth = demo[f"obs/{right_name}/depth"]
    n = int(left_rgb.shape[0])
    if not (right_rgb.shape[0] == left_depth.shape[0] == right_depth.shape[0] == n):
        raise ValueError(f"Unsynchronized stereo lengths for {camera_dir_name}")
    if left_rgb.shape[1:3] != right_rgb.shape[1:3] or left_depth.shape[1:3] != right_depth.shape[1:3]:
        raise ValueError(f"Mismatched left/right resolution for {camera_dir_name}")

    frame_ids = np.arange(n, dtype=np.int64)
    width = int(meta.get("width", left_rgb.shape[2]))
    height = int(meta.get("height", left_rgb.shape[1]))
    left_k = _intrinsics_for_camera(left_name, width, height)
    right_k = _intrinsics_for_camera(right_name, width, height)

    left_pos = np.asarray(demo[f"camera_pose/{left_name}/position"], dtype=np.float32)
    right_pos = np.asarray(demo[f"camera_pose/{right_name}/position"], dtype=np.float32)
    left_quat = np.asarray(demo[f"camera_pose/{left_name}/orientation_wxyz"], dtype=np.float32)
    right_quat = np.asarray(demo[f"camera_pose/{right_name}/orientation_wxyz"], dtype=np.float32)
    left_c2w = _camera_to_world_from_pose(left_pos, left_quat)
    right_c2w = _camera_to_world_from_pose(right_pos, right_quat)

    for i in range(n):
        frame_id = f"{i:06d}"
        for side, data in (("left", left_rgb[i]), ("right", right_rgb[i])):
            rgb = np.asarray(data, dtype=np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            ok = cv2.imwrite(str(sequence_dir / "rgb" / side / f"{frame_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"Failed to write RGB frame {camera_dir_name}/{side}/{frame_id}.png")
        for side, data in (("left", left_depth[i]), ("right", right_depth[i])):
            depth = np.asarray(data, dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)
            np.save(sequence_dir / "depth" / side / f"{frame_id}.npy", depth)

    np.savez_compressed(sequence_dir / "camera" / "left.npz", frame_ids=frame_ids, intrinsics=left_k, camera_to_world=left_c2w)
    np.savez_compressed(sequence_dir / "camera" / "right.npz", frame_ids=frame_ids, intrinsics=right_k, camera_to_world=right_c2w)
    intrinsics_json = {
        "left": {"model": "pinhole", "width": width, "height": height, "matrix": left_k.tolist()},
        "right": {"model": "pinhole", "width": width, "height": height, "matrix": right_k.tolist()},
    }
    (sequence_dir / "camera" / "intrinsics.json").write_text(json.dumps(intrinsics_json, indent=2), encoding="utf-8")

    seq_meta = dict(meta)
    seq_meta.update(
        {
            "format": "stereo_directory_v1",
            "pair_type": pair_type,
            "left_source_camera": left_name,
            "right_source_camera": right_name,
            "num_frames": n,
            "frame_ids": frame_ids.tolist(),
            "depth_unit": "meter",
            "depth_invalid_value": 0,
            "pose_type": "camera_to_world",
            "camera_axes": "+x right, +y down, +z forward",
            "rgb_layout": "HWC RGB uint8 PNG",
            "depth_layout": "HW float32 NPY",
        }
    )
    if "env_step" in demo:
        seq_meta["env_steps"] = np.asarray(demo["env_step"], dtype=np.int32).tolist()
    (sequence_dir / "metadata.json").write_text(json.dumps(seq_meta, indent=2, sort_keys=True), encoding="utf-8")

    _validate_directory_sequence(sequence_dir, n, height, width)
    (sequence_dir / ".complete").write_text("ok\n", encoding="utf-8")
    return sequence_dir


def _validate_directory_sequence(sequence_dir: Path, n: int, height: int, width: int):
    for side in ("left", "right"):
        rgb_files = sorted((sequence_dir / "rgb" / side).glob("*.png"))
        depth_files = sorted((sequence_dir / "depth" / side).glob("*.npy"))
        if len(rgb_files) != n or len(depth_files) != n:
            raise ValueError(f"Expected {n} RGB/depth frames for {sequence_dir}/{side}, got {len(rgb_files)}/{len(depth_files)}")
        if [p.stem for p in rgb_files] != [f"{i:06d}" for i in range(n)]:
            raise ValueError(f"Non-contiguous RGB frame ids in {sequence_dir}/{side}")
        sample_rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_UNCHANGED)
        sample_depth = np.load(depth_files[0])
        if sample_rgb is None or sample_rgb.shape[:2] != (height, width) or sample_rgb.shape[2] != 3:
            raise ValueError(f"Invalid RGB sample shape in {rgb_files[0]}")
        if sample_depth.shape != (height, width) or sample_depth.dtype != np.float32:
            raise ValueError(f"Invalid depth sample shape/dtype in {depth_files[0]}")
        cam_npz = np.load(sequence_dir / "camera" / f"{side}.npz")
        if set(cam_npz.files) != {"frame_ids", "intrinsics", "camera_to_world"}:
            raise ValueError(f"Invalid NPZ fields in {side}.npz: {cam_npz.files}")
        if cam_npz["frame_ids"].shape != (n,) or cam_npz["intrinsics"].shape != (3, 3) or cam_npz["camera_to_world"].shape != (n, 4, 4):
            raise ValueError(f"Invalid NPZ shapes in {side}.npz")
        if not np.array_equal(cam_npz["frame_ids"], np.arange(n)):
            raise ValueError(f"Invalid frame_ids in {side}.npz")


def _convert_hdf5_to_directory(hdf5_path: Path, root_dir: Path, scene_name: str, record_camera_names: tuple[str, ...], overwrite: bool) -> list[Path]:
    scene_dir = root_dir / _clean_name(scene_name)
    if scene_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{scene_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as h5:
        meta = _decode_json_dataset(h5["data/demo_0/metadata_json"])
        meta["source_hdf5"] = str(hdf5_path)
        written = []
        for camera_dir_name, left_name, right_name, pair_type in _stereo_pairs_for_directory(record_camera_names):
            written.append(_write_directory_pair(h5, scene_dir, camera_dir_name, left_name, right_name, pair_type, meta))
    return written


def _resolve_distractor_asset(asset_name: str) -> str:
    path = Path(asset_name)
    if path.is_absolute() and path.exists():
        return str(path)
    candidates = [Path(PACKAGE_DIR) / asset_name]
    if not asset_name.startswith("assets/"):
        candidates.extend((Path(PACKAGE_DIR) / "assets" / "objects").glob(f"**/{asset_name}"))
        candidates.extend((Path(PACKAGE_DIR) / "assets" / "objects").glob(f"**/{asset_name}.usd"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not resolve distractor asset {asset_name!r}")


def _set_prim_transform(stage, prim_path: str, position, yaw_deg: float, scale: float):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(yaw_deg)
    xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


def _snap_prim_bottom_to_z(stage, prim_path: str, target_z: float):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_z = float(box.GetMin()[2])
    if not np.isfinite(min_z):
        return None
    delta_z = target_z - min_z
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return None
    current = translate_ops[0].Get()
    translate_ops[0].Set((float(current[0]), float(current[1]), float(current[2]) + delta_z))
    return min_z + delta_z


def _add_physics_to_real_distractor(stage, prim_path: str, mass: float):
    if args_cli.distractor_collider_approx == "none":
        return {"physics": "disabled"}
    from robolab.core.utils.physics_utils import add_mass_api, add_rigid_body_collider

    prim = stage.GetPrimAtPath(prim_path)
    try:
        add_rigid_body_collider(
            prim,
            approximation=args_cli.distractor_collider_approx,
            kinematic=False,
            contact_offset=0.01,
            rest_offset=0.001,
            enable_ccd=True,
        )
        add_mass_api(mass, root_prim_path=prim_path)
        return {"physics": "rigid_body", "mass_kg": mass, "collider_approx": args_cli.distractor_collider_approx}
    except Exception as exc:
        print(f"[WARN] Could not add physics to {prim_path}: {exc}")
        return {"physics": "failed", "error": str(exc)}



def _add_static_geometry_background():
    if not args_cli.add_geometry_background:
        return []

    from pxr import Gf, Sdf, UsdGeom, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    rng = np.random.default_rng(args_cli.geometry_background_seed)
    scale = float(max(args_cli.geometry_background_scale, 0.1))
    root_path = "/World/envs/env_0/scene/geometric_background"
    UsdGeom.Xform.Define(stage, root_path)

    def make_material(name: str, color, roughness: float = 0.75):
        mat_path = f"{root_path}/materials/{name}"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/preview_surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    floor_mat = make_material("floor_mat", rng.uniform([0.35, 0.34, 0.31], [0.58, 0.56, 0.52]), 0.82)
    wall_mat = make_material("wall_mat", rng.uniform([0.55, 0.57, 0.58], [0.82, 0.82, 0.80]), 0.9)
    cabinet_mat = make_material("cabinet_mat", rng.uniform([0.30, 0.22, 0.14], [0.55, 0.42, 0.28]), 0.68)
    shelf_mat = make_material("shelf_mat", rng.uniform([0.18, 0.18, 0.18], [0.38, 0.34, 0.30]), 0.72)
    accent_mat = make_material("accent_mat", rng.uniform([0.10, 0.18, 0.25], [0.48, 0.55, 0.60]), 0.65)

    created = []

    def cube(name: str, center, dims, material):
        prim_path = f"{root_path}/{name}"
        geom = UsdGeom.Cube.Define(stage, prim_path)
        geom.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        xform.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in dims]))
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)
        created.append({"path": prim_path, "center": list(map(float, center)), "dims": list(map(float, dims))})

    # Coordinate assumptions used by RoboLab table tasks: operation volume is near x=[0, 1], y=[-0.6, 0.6], z>=0.
    # The shell is placed outside that workspace to provide valid depth without touching task objects.
    floor_z = -0.58 * scale
    wall_h = 2.4 * scale
    wall_center_z = floor_z + wall_h * 0.5
    cube("floor", (0.25, 0.0, floor_z), (4.0 * scale, 3.2 * scale, 0.035 * scale), floor_mat)
    cube("back_wall", (-1.05 * scale, 0.0, wall_center_z), (0.04 * scale, 3.2 * scale, wall_h), wall_mat)
    cube("left_wall", (0.25, -1.55 * scale, wall_center_z), (4.0 * scale, 0.04 * scale, wall_h), wall_mat)
    cube("right_wall", (0.25, 1.55 * scale, wall_center_z), (4.0 * scale, 0.04 * scale, wall_h), wall_mat)

    def add_rear_shelves(prefix: str, y_values=(-0.92, 0.92), x=-0.74, cabinet_x=-0.92):
        for side, y0 in enumerate(y_values):
            y = float(y0) * scale
            cube(f"{prefix}_cabinet_{side}", (cabinet_x * scale, y, 0.05 * scale), (0.22 * scale, 0.44 * scale, 1.10 * scale), cabinet_mat)
            for level, z in enumerate((0.40 * scale, 0.78 * scale, 1.16 * scale)):
                cube(f"{prefix}_shelf_{side}_{level}", (x * scale, y, z), (0.32 * scale, 0.58 * scale, 0.035 * scale), shelf_mat)
            for item in range(4):
                iy = y + float(rng.uniform(-0.20, 0.20)) * scale
                iz = float(rng.uniform(0.48, 1.28)) * scale
                dims = np.asarray([0.08, 0.08, 0.16], dtype=float) * float(rng.uniform(0.7, 1.4)) * scale
                cube(f"{prefix}_box_{side}_{item}", ((x + 0.06) * scale, iy, iz), dims, accent_mat)

    if args_cli.geometry_background_style in {"room_shelves", "room_cabinets"}:
        add_rear_shelves("rear")

    if args_cli.geometry_background_style == "room_cabinets":
        for idx, y in enumerate(np.linspace(-0.55, 0.55, 4)):
            cube(f"low_cabinet_{idx}", (-0.72 * scale, float(y) * scale, -0.03 * scale), (0.28 * scale, 0.24 * scale, 0.42 * scale), cabinet_mat)
        cube("countertop_background", (-0.70 * scale, 0.0, 0.20 * scale), (0.34 * scale, 1.35 * scale, 0.04 * scale), shelf_mat)

    if args_cli.geometry_background_style == "corner_shelves":
        add_rear_shelves("corner_rear", y_values=(-1.05, -0.62, -0.19), x=-0.78, cabinet_x=-0.98)
        for idx, x in enumerate((-0.45, 0.05, 0.55)):
            cube(f"corner_side_shelf_{idx}", (x * scale, -1.30 * scale, 0.72 * scale), (0.42 * scale, 0.18 * scale, 0.035 * scale), shelf_mat)
            cube(f"corner_side_box_{idx}", (x * scale, -1.20 * scale, 0.85 * scale), (0.14 * scale, 0.12 * scale, 0.22 * scale), accent_mat)

    if args_cli.geometry_background_style == "warehouse_racks":
        for rack, y0 in enumerate((-1.02, -0.52, 0.52, 1.02)):
            y = y0 * scale
            for post_x in (-0.98, -0.58):
                cube(f"rack_{rack}_post_a_{post_x}", (post_x * scale, y - 0.22 * scale, 0.45 * scale), (0.035 * scale, 0.035 * scale, 1.55 * scale), shelf_mat)
                cube(f"rack_{rack}_post_b_{post_x}", (post_x * scale, y + 0.22 * scale, 0.45 * scale), (0.035 * scale, 0.035 * scale, 1.55 * scale), shelf_mat)
            for level, z in enumerate((0.18, 0.55, 0.92, 1.29)):
                cube(f"rack_{rack}_level_{level}", (-0.78 * scale, y, z * scale), (0.46 * scale, 0.54 * scale, 0.035 * scale), shelf_mat)
                for item in range(2):
                    cube(f"rack_{rack}_bin_{level}_{item}", ((-0.88 + 0.20 * item) * scale, (y0 + rng.uniform(-0.13, 0.13)) * scale, (z + 0.11) * scale), (0.15 * scale, 0.16 * scale, 0.18 * scale), accent_mat)

    if args_cli.geometry_background_style == "office_partition":
        for idx, y in enumerate((-0.90, -0.30, 0.30, 0.90)):
            cube(f"partition_panel_{idx}", (-0.78 * scale, y * scale, 0.45 * scale), (0.05 * scale, 0.42 * scale, 0.90 * scale), wall_mat)
            cube(f"desk_back_{idx}", (-0.52 * scale, y * scale, -0.10 * scale), (0.34 * scale, 0.38 * scale, 0.28 * scale), cabinet_mat)
            cube(f"monitor_block_{idx}", (-0.42 * scale, y * scale, 0.18 * scale), (0.05 * scale, 0.24 * scale, 0.18 * scale), accent_mat)

    if args_cli.geometry_background_style == "kitchen_counter":
        cube("kitchen_back_counter", (-0.66 * scale, 0.0, 0.00 * scale), (0.38 * scale, 1.45 * scale, 0.48 * scale), cabinet_mat)
        cube("kitchen_countertop", (-0.62 * scale, 0.0, 0.26 * scale), (0.46 * scale, 1.55 * scale, 0.045 * scale), shelf_mat)
        for idx, y in enumerate(np.linspace(-0.58, 0.58, 5)):
            cube(f"kitchen_upper_cabinet_{idx}", (-0.92 * scale, float(y) * scale, 0.94 * scale), (0.16 * scale, 0.22 * scale, 0.36 * scale), cabinet_mat)
            cube(f"kitchen_canister_{idx}", (-0.37 * scale, float(y) * scale, 0.43 * scale), (0.10 * scale, 0.10 * scale, 0.18 * scale), accent_mat)

    if args_cli.geometry_background_style == "clutter_wall":
        for idx in range(18):
            x = float(rng.uniform(-0.98, -0.62)) * scale
            y = float(rng.uniform(-1.18, 1.18)) * scale
            z = float(rng.uniform(0.05, 1.30)) * scale
            dims = np.asarray(rng.uniform([0.05, 0.06, 0.06], [0.18, 0.24, 0.32]), dtype=float) * scale
            cube(f"clutter_wall_box_{idx}", (x, y, z), dims, accent_mat if idx % 3 else cabinet_mat)
        for idx, z in enumerate((0.28, 0.62, 0.96, 1.30)):
            cube(f"clutter_wall_rail_{idx}", (-0.76 * scale, 0.0, z * scale), (0.08 * scale, 2.30 * scale, 0.025 * scale), shelf_mat)

    if args_cli.geometry_background_style == "lab_benches":
        for idx, y in enumerate((-0.82, 0.0, 0.82)):
            cube(f"lab_bench_{idx}", (-0.58 * scale, y * scale, -0.06 * scale), (0.42 * scale, 0.42 * scale, 0.34 * scale), cabinet_mat)
            cube(f"lab_bench_top_{idx}", (-0.54 * scale, y * scale, 0.13 * scale), (0.50 * scale, 0.50 * scale, 0.035 * scale), shelf_mat)
            for item in range(3):
                cube(f"lab_equipment_{idx}_{item}", ((-0.36 + 0.02 * item) * scale, (y + rng.uniform(-0.14, 0.14)) * scale, (0.26 + 0.10 * item) * scale), (0.10 * scale, 0.10 * scale, 0.16 * scale), accent_mat)

    if args_cli.geometry_background_style == "garage_storage":
        for idx, y in enumerate(np.linspace(-1.08, 1.08, 5)):
            cube(f"garage_tall_cabinet_{idx}", (-0.90 * scale, float(y) * scale, 0.32 * scale), (0.24 * scale, 0.30 * scale, 1.12 * scale), cabinet_mat)
        for idx in range(10):
            cube(f"garage_storage_bin_{idx}", (-0.56 * scale, float(rng.uniform(-1.10, 1.10)) * scale, float(rng.uniform(0.00, 0.90)) * scale), (0.18 * scale, 0.22 * scale, 0.16 * scale), accent_mat)

    if args_cli.geometry_background_style == "library_shelves":
        for col, y in enumerate(np.linspace(-1.15, 1.15, 7)):
            cube(f"library_vertical_{col}", (-0.94 * scale, float(y) * scale, 0.55 * scale), (0.04 * scale, 0.04 * scale, 1.55 * scale), shelf_mat)
        for level, z in enumerate(np.linspace(0.12, 1.34, 6)):
            cube(f"library_shelf_level_{level}", (-0.88 * scale, 0.0, float(z) * scale), (0.16 * scale, 2.50 * scale, 0.035 * scale), shelf_mat)
            for book in range(8):
                cube(f"library_book_{level}_{book}", (-0.76 * scale, float(-1.05 + book * 0.30 + rng.uniform(-0.04, 0.04)) * scale, float(z + 0.11) * scale), (0.045 * scale, 0.08 * scale, 0.20 * scale), accent_mat)

    if args_cli.geometry_background_style == "factory_bins":
        for row, z in enumerate((0.05, 0.32, 0.59, 0.86, 1.13)):
            for col, y in enumerate(np.linspace(-1.05, 1.05, 6)):
                cube(f"factory_bin_{row}_{col}", (-0.72 * scale, float(y) * scale, z * scale), (0.22 * scale, 0.24 * scale, 0.20 * scale), accent_mat if (row + col) % 2 else cabinet_mat)
        cube("factory_back_rail", (-0.60 * scale, 0.0, 1.42 * scale), (0.12 * scale, 2.45 * scale, 0.05 * scale), shelf_mat)

    if args_cli.geometry_background_style == "pegboard_tools":
        cube("pegboard", (-0.96 * scale, 0.0, 0.78 * scale), (0.035 * scale, 2.25 * scale, 1.20 * scale), cabinet_mat)
        for idx in range(16):
            y = float(rng.uniform(-1.00, 1.00)) * scale
            z = float(rng.uniform(0.25, 1.28)) * scale
            dims = np.asarray(rng.uniform([0.03, 0.12, 0.04], [0.06, 0.24, 0.09]), dtype=float) * scale
            cube(f"pegboard_tool_{idx}", (-0.88 * scale, y, z), dims, accent_mat)
        for idx, y in enumerate(np.linspace(-1.0, 1.0, 5)):
            cube(f"pegboard_hook_rail_{idx}", (-0.84 * scale, float(y) * scale, 0.72 * scale), (0.08 * scale, 0.015 * scale, 1.05 * scale), shelf_mat)

    if args_cli.geometry_background_style == "apartment_furniture":
        cube("sofa_back", (-0.82 * scale, -0.58 * scale, 0.10 * scale), (0.28 * scale, 0.82 * scale, 0.42 * scale), accent_mat)
        cube("sofa_seat", (-0.54 * scale, -0.58 * scale, -0.10 * scale), (0.48 * scale, 0.82 * scale, 0.18 * scale), accent_mat)
        cube("side_table", (-0.50 * scale, 0.48 * scale, -0.16 * scale), (0.28 * scale, 0.28 * scale, 0.26 * scale), cabinet_mat)
        cube("bookcase_apartment", (-0.92 * scale, 0.90 * scale, 0.54 * scale), (0.18 * scale, 0.36 * scale, 1.18 * scale), shelf_mat)
        for idx in range(6):
            cube(f"apartment_decor_{idx}", (-0.70 * scale, float(rng.uniform(0.72, 1.05)) * scale, float(rng.uniform(0.10, 1.05)) * scale), (0.08 * scale, 0.08 * scale, 0.12 * scale), accent_mat)

    if args_cli.geometry_background_style == "corridor_depth":
        for idx, x in enumerate((-0.85, -0.35, 0.15, 0.65)):
            cube(f"corridor_left_panel_{idx}", (x * scale, -1.22 * scale, 0.52 * scale), (0.28 * scale, 0.05 * scale, 1.20 * scale), wall_mat)
            cube(f"corridor_right_panel_{idx}", (x * scale, 1.22 * scale, 0.52 * scale), (0.28 * scale, 0.05 * scale, 1.20 * scale), wall_mat)
        cube("corridor_back_door", (-1.02 * scale, 0.0, 0.42 * scale), (0.05 * scale, 0.46 * scale, 0.86 * scale), cabinet_mat)

    if args_cli.geometry_background_style == "stacked_crates":
        for row, z in enumerate((0.02, 0.30, 0.58, 0.86)):
            for col, y in enumerate(np.linspace(-1.05, 1.05, 5)):
                x = (-0.72 + 0.05 * ((row + col) % 2)) * scale
                cube(f"stacked_crate_{row}_{col}", (x, float(y) * scale, z * scale), (0.24 * scale, 0.32 * scale, 0.24 * scale), cabinet_mat if (row + col) % 2 else accent_mat)
                cube(f"stacked_crate_lip_{row}_{col}", (x + 0.01 * scale, float(y) * scale, (z + 0.13) * scale), (0.26 * scale, 0.34 * scale, 0.025 * scale), shelf_mat)

    if args_cli.geometry_background_style.startswith("bg") and len(args_cli.geometry_background_style) == 5:
        try:
            template_id = int(args_cli.geometry_background_style[2:])
        except ValueError as exc:
            raise ValueError(f"Invalid geometry background template: {args_cli.geometry_background_style}") from exc
        if not 0 <= template_id <= 99:
            raise ValueError(f"Geometry background template must be bg000-bg099, got {args_cli.geometry_background_style}")

        trng = np.random.default_rng(args_cli.geometry_background_seed + template_id * 1009)
        family = template_id % 10
        variant = template_id // 10
        density = 0.75 + 0.06 * variant
        span_y = (0.82 + 0.035 * variant) * scale
        back_x = (-0.98 + 0.025 * (variant % 4)) * scale
        mid_x = (-0.68 + 0.018 * (variant % 5)) * scale

        def rand_box_dims(min_dims, max_dims):
            return np.asarray(trng.uniform(min_dims, max_dims), dtype=float) * scale

        def add_grid_boxes(prefix, rows, cols, x, y_min, y_max, z_min, z_gap, dims_min, dims_max, mat_cycle=True):
            for r in range(rows):
                for c in range(cols):
                    y = float(np.interp(c, [0, max(cols - 1, 1)], [y_min, y_max]) + trng.uniform(-0.035, 0.035)) * scale
                    z = float(z_min + r * z_gap + trng.uniform(-0.02, 0.03)) * scale
                    dims = rand_box_dims(dims_min, dims_max)
                    mat = accent_mat if (r + c + template_id) % 3 else cabinet_mat
                    if not mat_cycle:
                        mat = shelf_mat
                    cube(f"{prefix}_{r}_{c}", (x, y, z), dims, mat)

        if family == 0:
            # Dense rear storage wall.
            for level, z in enumerate(np.linspace(0.18, 1.30, 4 + variant % 3)):
                cube(f"bg{template_id:03d}_storage_shelf_{level}", (mid_x, 0.0, float(z) * scale), (0.22 * scale, 2.20 * scale, 0.035 * scale), shelf_mat)
            add_grid_boxes(f"bg{template_id:03d}_storage_box", 4 + variant % 3, 5 + variant % 4, mid_x + 0.08 * scale, -1.02, 1.02, 0.28, 0.28, [0.07, 0.08, 0.08], [0.16, 0.20, 0.22])

        elif family == 1:
            # Asymmetric office/lab benches.
            for idx, y in enumerate(np.linspace(-span_y, span_y, 3 + variant % 3)):
                cube(f"bg{template_id:03d}_bench_{idx}", (mid_x, float(y), -0.10 * scale), (0.42 * scale, 0.36 * scale, 0.28 * scale), cabinet_mat)
                cube(f"bg{template_id:03d}_bench_top_{idx}", (mid_x + 0.04 * scale, float(y), 0.07 * scale), (0.50 * scale, 0.43 * scale, 0.035 * scale), shelf_mat)
                for item in range(2 + variant % 3):
                    cube(f"bg{template_id:03d}_bench_item_{idx}_{item}", (mid_x + 0.18 * scale, float(y + trng.uniform(-0.12, 0.12) * scale), float(trng.uniform(0.20, 0.62)) * scale), rand_box_dims([0.05, 0.05, 0.08], [0.14, 0.16, 0.28]), accent_mat)

        elif family == 2:
            # Warehouse rack with strong vertical/horizontal lines.
            rack_count = 3 + variant % 4
            for rack, y in enumerate(np.linspace(-1.05, 1.05, rack_count)):
                for px in (-0.96, -0.52):
                    cube(f"bg{template_id:03d}_rack_post_{rack}_{px}_a", (px * scale, float(y - 0.20) * scale, 0.48 * scale), (0.032 * scale, 0.032 * scale, 1.52 * scale), shelf_mat)
                    cube(f"bg{template_id:03d}_rack_post_{rack}_{px}_b", (px * scale, float(y + 0.20) * scale, 0.48 * scale), (0.032 * scale, 0.032 * scale, 1.52 * scale), shelf_mat)
                for level, z in enumerate((0.18, 0.52, 0.86, 1.20)):
                    cube(f"bg{template_id:03d}_rack_level_{rack}_{level}", (-0.74 * scale, float(y) * scale, z * scale), (0.50 * scale, 0.48 * scale, 0.032 * scale), shelf_mat)
                    cube(f"bg{template_id:03d}_rack_bin_{rack}_{level}", (-0.70 * scale, float(y + trng.uniform(-0.12, 0.12)) * scale, (z + 0.10) * scale), rand_box_dims([0.10, 0.10, 0.11], [0.22, 0.20, 0.20]), accent_mat)

        elif family == 3:
            # Kitchen/counter profile.
            cube(f"bg{template_id:03d}_counter_base", (-0.64 * scale, 0.0, -0.02 * scale), (0.42 * scale, 1.48 * scale, 0.46 * scale), cabinet_mat)
            cube(f"bg{template_id:03d}_counter_top", (-0.58 * scale, 0.0, 0.24 * scale), (0.52 * scale, 1.62 * scale, 0.04 * scale), shelf_mat)
            for idx, y in enumerate(np.linspace(-0.65, 0.65, 4 + variant % 3)):
                cube(f"bg{template_id:03d}_upper_cabinet_{idx}", (back_x, float(y) * scale, float(0.82 + 0.05 * (idx % 2)) * scale), rand_box_dims([0.14, 0.18, 0.28], [0.22, 0.28, 0.44]), cabinet_mat)
                cube(f"bg{template_id:03d}_counter_item_{idx}", (-0.34 * scale, float(y + trng.uniform(-0.05, 0.05)) * scale, 0.42 * scale), rand_box_dims([0.06, 0.06, 0.10], [0.12, 0.12, 0.22]), accent_mat)

        elif family == 4:
            # Bookcase/library wall.
            for col, y in enumerate(np.linspace(-1.18, 1.18, 6 + variant % 4)):
                cube(f"bg{template_id:03d}_bookcase_post_{col}", (back_x, float(y) * scale, 0.58 * scale), (0.035 * scale, 0.035 * scale, 1.50 * scale), shelf_mat)
            for level, z in enumerate(np.linspace(0.16, 1.34, 5 + variant % 3)):
                cube(f"bg{template_id:03d}_bookcase_level_{level}", (back_x + 0.08 * scale, 0.0, float(z) * scale), (0.17 * scale, 2.48 * scale, 0.03 * scale), shelf_mat)
                add_grid_boxes(f"bg{template_id:03d}_books_{level}", 1, 7 + variant % 4, back_x + 0.18 * scale, -1.05, 1.05, float(z + 0.10), 0.0, [0.035, 0.05, 0.12], [0.060, 0.10, 0.24])

        elif family == 5:
            # Tool pegboard/workshop wall.
            cube(f"bg{template_id:03d}_pegboard", (back_x, 0.0, 0.78 * scale), (0.035 * scale, 2.30 * scale, 1.25 * scale), cabinet_mat)
            for rail, z in enumerate(np.linspace(0.28, 1.24, 4 + variant % 3)):
                cube(f"bg{template_id:03d}_peg_rail_{rail}", (back_x + 0.10 * scale, 0.0, float(z) * scale), (0.06 * scale, 2.08 * scale, 0.018 * scale), shelf_mat)
            for idx in range(int(12 + variant * density)):
                cube(f"bg{template_id:03d}_tool_{idx}", (back_x + 0.13 * scale, float(trng.uniform(-1.00, 1.00)) * scale, float(trng.uniform(0.25, 1.26)) * scale), rand_box_dims([0.025, 0.08, 0.03], [0.07, 0.24, 0.12]), accent_mat)

        elif family == 6:
            # Apartment/furniture silhouette.
            y0 = float(trng.uniform(-0.75, -0.35)) * scale
            cube(f"bg{template_id:03d}_sofa_back", (-0.82 * scale, y0, 0.12 * scale), (0.28 * scale, 0.82 * scale, 0.42 * scale), accent_mat)
            cube(f"bg{template_id:03d}_sofa_seat", (-0.54 * scale, y0, -0.10 * scale), (0.48 * scale, 0.82 * scale, 0.18 * scale), accent_mat)
            cube(f"bg{template_id:03d}_side_table", (-0.48 * scale, 0.48 * scale, -0.16 * scale), (0.28 * scale, 0.28 * scale, 0.26 * scale), cabinet_mat)
            add_grid_boxes(f"bg{template_id:03d}_decor", 2 + variant % 3, 3, -0.72 * scale, 0.58, 1.10, 0.12, 0.30, [0.04, 0.04, 0.08], [0.12, 0.12, 0.20])

        elif family == 7:
            # Corridor/depth panels.
            panel_count = 4 + variant % 4
            for idx, x0 in enumerate(np.linspace(-0.95, 0.62, panel_count)):
                cube(f"bg{template_id:03d}_corridor_left_{idx}", (float(x0) * scale, -1.24 * scale, 0.54 * scale), (0.24 * scale, 0.045 * scale, 1.18 * scale), wall_mat)
                cube(f"bg{template_id:03d}_corridor_right_{idx}", (float(x0) * scale, 1.24 * scale, 0.54 * scale), (0.24 * scale, 0.045 * scale, 1.18 * scale), wall_mat)
                if idx % 2 == variant % 2:
                    cube(f"bg{template_id:03d}_corridor_box_{idx}", (float(x0 + 0.10) * scale, float(trng.choice([-0.92, 0.92])) * scale, -0.08 * scale), (0.18 * scale, 0.20 * scale, 0.28 * scale), cabinet_mat)

        elif family == 8:
            # Stacked crates / logistics wall.
            rows = 3 + variant % 4
            cols = 4 + variant % 4
            for r in range(rows):
                for c, y in enumerate(np.linspace(-1.02, 1.02, cols)):
                    x = (-0.74 + 0.045 * ((r + c + variant) % 3)) * scale
                    z = (0.03 + r * 0.26) * scale
                    cube(f"bg{template_id:03d}_crate_{r}_{c}", (x, float(y) * scale, z), rand_box_dims([0.18, 0.22, 0.18], [0.28, 0.34, 0.26]), cabinet_mat if (r + c) % 2 else accent_mat)

        else:
            # Mixed dense clutter: rails, blocks, partial partitions.
            for level, z in enumerate((0.24, 0.54, 0.84, 1.14)):
                cube(f"bg{template_id:03d}_mixed_rail_{level}", (back_x + 0.10 * scale, 0.0, z * scale), (0.10 * scale, 2.26 * scale, 0.025 * scale), shelf_mat)
            for idx in range(14 + variant):
                cube(f"bg{template_id:03d}_mixed_block_{idx}", (float(trng.uniform(-0.92, -0.48)) * scale, float(trng.uniform(-1.12, 1.12)) * scale, float(trng.uniform(0.02, 1.25)) * scale), rand_box_dims([0.04, 0.05, 0.05], [0.18, 0.24, 0.30]), accent_mat if idx % 2 else cabinet_mat)

    print(f"[INFO] Added static geometry background: {len(created)} prims, style={args_cli.geometry_background_style}.")
    return created

def _add_random_visual_distractors():
    if not args_cli.add_random_distractors:
        return []

    from pxr import Gf, Sdf, UsdGeom, UsdShade
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    rng = np.random.default_rng(args_cli.distractor_seed)
    created = []
    root_path = "/World/envs/env_0/scene/random_distractors"
    UsdGeom.Xform.Define(stage, root_path)

    if args_cli.distractor_mode == "real":
        asset_names = args_cli.distractor_assets or list(DEFAULT_REAL_DISTRACTOR_ASSETS)
        asset_paths = [_resolve_distractor_asset(name) for name in asset_names]
        for idx in range(args_cli.num_distractors):
            asset_path = str(rng.choice(asset_paths))
            asset_stem = _clean_name(Path(asset_path).stem)
            prim_path = f"{root_path}/distractor_{idx:03d}_{asset_stem}"
            root_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
            root_prim.GetReferences().AddReference(asset_path)

            x = float(rng.uniform(*args_cli.distractor_x_range))
            y = float(rng.uniform(*args_cli.distractor_y_range))
            z = float(args_cli.distractor_table_z + 0.20)
            yaw = float(rng.uniform(-180.0, 180.0))
            scale = float(rng.uniform(*args_cli.distractor_scale_range))
            _set_prim_transform(stage, prim_path, (x, y, z), yaw, scale)
            snapped_bottom_z = _snap_prim_bottom_to_z(stage, prim_path, args_cli.distractor_table_z)
            mass = float(rng.uniform(*args_cli.distractor_mass_range))
            physics = _add_physics_to_real_distractor(stage, prim_path, mass)

            created.append(
                {
                    "path": prim_path,
                    "mode": "real",
                    "asset": asset_path,
                    "position_xy": [x, y],
                    "table_z": args_cli.distractor_table_z,
                    "snapped_bottom_z": snapped_bottom_z,
                    "yaw_deg": yaw,
                    "scale": scale,
                    **physics,
                }
            )
        print(f"[INFO] Added {len(created)} real USD distractors with physics metadata.")
        return created

    shapes = ("cube", "sphere", "cylinder")
    for idx in range(args_cli.num_distractors):
        shape = str(rng.choice(shapes))
        size = float(rng.uniform(*args_cli.distractor_size_range))
        x = float(rng.uniform(*args_cli.distractor_x_range))
        y = float(rng.uniform(*args_cli.distractor_y_range))
        z = float(args_cli.distractor_table_z + size * 0.5)
        prim_path = f"{root_path}/distractor_{idx:03d}_{shape}"

        if shape == "cube":
            geom = UsdGeom.Cube.Define(stage, prim_path)
            geom.CreateSizeAttr(size)
        elif shape == "sphere":
            geom = UsdGeom.Sphere.Define(stage, prim_path)
            geom.CreateRadiusAttr(size * 0.5)
        else:
            geom = UsdGeom.Cylinder.Define(stage, prim_path)
            geom.CreateRadiusAttr(size * 0.45)
            geom.CreateHeightAttr(size)

        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xform.AddRotateZOp().Set(float(rng.uniform(-180.0, 180.0)))

        color = tuple(float(v) for v in rng.uniform(0.05, 0.95, size=3))
        mat_path = f"{root_path}/materials/distractor_mat_{idx:03d}"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/preview_surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rng.uniform(0.35, 0.9)))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

        created.append({"path": prim_path, "mode": "geometry", "shape": shape, "position": [x, y, z], "size": size, "color": list(color)})

    print(f"[INFO] Added {len(created)} simple geometry distractors.")
    return created

def _save_hdf5_schema_summary(path: Path):
    print(f"\n[DATASET] {path}")
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}/")
        f.visititems(visitor)


def _lighting_cfg_from_name(name: str):
    from robolab.variations import lighting as lighting_presets

    mapping = {
        "sphere": lighting_presets.SphereLightCfg,
        "dim": lighting_presets.ExtremelyDimSphereLightCfg,
        "top": lighting_presets.TopDownDirectionalLightCfg,
        "left": lighting_presets.LeftDirectionalLightCfg,
        "right": lighting_presets.RightDirectionalLightCfg,
        "front": lighting_presets.FrontDirectionalLightCfg,
        "behind": lighting_presets.BehindDirectionalLightCfg,
        "red": lighting_presets.RedSphereLightCfg,
        "green": lighting_presets.GreenSphereLightCfg,
        "blue": lighting_presets.BlueSphereLightCfg,
    }
    return mapping[name]


def _build_reset_events(camera_names: tuple[str, ...]):
    events = {}
    if args_cli.randomize_camera_pose:
        pos = args_cli.camera_pos_range
        rot = args_cli.camera_rot_range
        events["randomize_stereo_camera_pose"] = EventTerm(
            func=reset_camera_pose_uniform,
            mode="reset",
            params={
                "camera_names": list(record_camera_names),
                "runtime_camera_names": list(runtime_camera_names),
                "pose_range": {
                    "x": (-pos, pos),
                    "y": (-pos, pos),
                    "z": (-pos, pos),
                    "roll": (-rot, rot),
                    "pitch": (-rot, rot),
                    "yaw": (-rot, rot),
                },
            },
        )

    if args_cli.randomize_object_pose:
        objects = args_cli.randomize_objects
        if objects is None:
            objects = ["banana", "bowl"] if args_cli.task == "BananaInBowlTask" else []
        if not objects:
            raise ValueError("--randomize-object-pose needs --randomize-objects for this task.")
        xy = args_cli.object_xy_range
        yaw = args_cli.object_yaw_range
        events["randomize_object_initial_pose"] = EventTerm(
            func=reset_pose_uniform,
            mode="reset",
            params={
                "asset_cfg": objects,
                "pose_range": {"x": (-xy, xy), "y": (-xy, xy), "z": (0.0, 0.0), "yaw": (-yaw, yaw)},
                "velocity_range": {},
                "collision_margin": 0.01,
                "max_retries": 100,
            },
        )

    return events or None


def run():
    output_dir = args_cli.output_dir
    if output_dir is None:
        output_dir = os.path.join(PACKAGE_DIR, "output", "stereo_dataset", _clean_name(args_cli.task))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(output_dir))

    camera_cfgs = [StereoLeftRGDBCameraCfg, StereoRightRGDBCameraCfg]
    camera_names = list(BASE_CAMERA_NAMES)
    needs_wrist = args_cli.include_wrist_camera or args_cli.include_wrist_stereo_camera or args_cli.policy_backend != "none"
    if needs_wrist:
        import robolab.robots.droid as droid_robot
        droid_robot._WRIST_CAM.height = args_cli.height
        droid_robot._WRIST_CAM.width = args_cli.width
        droid_robot._WRIST_CAM.data_types = ["rgb", "depth"]
        droid_robot._WRIST_CAM.update_latest_camera_pose = True
        camera_cfgs.append(droid_robot.WristCameraCfg)
        camera_names.append("wrist_cam")
    if args_cli.include_wrist_stereo_camera:
        import robolab.robots.droid as droid_robot
        droid_robot.DroidCfg.wrist_left_camera = WristLeftRGDBCameraCfg().wrist_left_camera
        droid_robot.DroidCfg.wrist_right_camera = WristRightRGDBCameraCfg().wrist_right_camera
        camera_names.extend(WRIST_STEREO_CAMERA_NAMES)
    if args_cli.policy_backend != "none":
        # Pi0-family clients expect this policy observation key. We attach it for inference
        # but keep the recorded dataset focused on stereo_left/right + wrist_cam.
        from robolab.variations.camera import OverShoulderLeftCameraCfg
        camera_cfgs.append(OverShoulderLeftCameraCfg)
    runtime_camera_names = tuple(camera_names)
    record_camera_names = tuple(
        name
        for name in runtime_camera_names
        if not (args_cli.exclude_policy_wrist_camera_from_output and name == "wrist_cam")
    )

    auto_register_droid_envs(
        task=args_cli.task,
        cameras=camera_cfgs,
        randomize_background=args_cli.randomize_background,
        background_seed=args_cli.background_seed,
        lighting_cfg=_lighting_cfg_from_name(args_cli.lighting),
    )
    task_envs = get_envs(task=[args_cli.task])
    if not task_envs:
        raise RuntimeError(f"No registered environments found for task {args_cli.task!r}")
    task_env = task_envs[0]

    print(f"[INFO] Creating environment {task_env}...", flush=True)
    try:
        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            events=_build_reset_events(runtime_camera_names),
            policy=args_cli.policy_backend if args_cli.policy_backend != "none" else None,
        )
    except SystemExit as exc:
        raise RuntimeError(f"create_env triggered SystemExit({exc.code}); treating as data-generation failure") from exc
    print(f"[INFO] create_env returned: env={env.__class__.__name__}, num_envs={env.num_envs}", flush=True)
    if env.num_envs != 1:
        raise ValueError("This first dataset script writes one environment per file. Please use --num_envs 1.")

    hdf5_path = output_dir / args_cli.file_name
    if hdf5_path.exists():
        if not args_cli.overwrite:
            raise FileExistsError(f"{hdf5_path} already exists. Pass --overwrite to replace it.")
        hdf5_path.unlink()

    try:
        print("[INFO] Rollout setup: getting robot...", flush=True)
        robot = env.scene["robot"]
        print("[INFO] Rollout setup: robot ready.", flush=True)
        if args_cli.record_every < 1:
            raise ValueError("--record-every must be >= 1")
        if args_cli.env_step_dt <= 0:
            raise ValueError("--env-step-dt must be > 0")
        skip_initial_steps = max(0, int(math.ceil(args_cli.skip_initial_seconds / args_cli.env_step_dt)))
        print(f"[INFO] Skip initial settling: {args_cli.skip_initial_seconds:.3f}s = {skip_initial_steps} env steps", flush=True)
        print("[INFO] Rollout setup: loading action source...", flush=True)
        replay_actions = _load_replay_actions()
        policy_client = _make_policy_client()
        print("[INFO] Rollout setup: resetting env...", flush=True)
        obs, _ = env.reset()
        print("[INFO] Rollout setup: adding geometry background...", flush=True)
        geometry_background = _add_static_geometry_background()
        print("[INFO] Rollout setup: adding distractors...", flush=True)
        distractors = _add_random_visual_distractors()

        print("[INFO] Rollout setup: collecting sensors...", flush=True)
        sensors = {name: env.scene.sensors[name] for name in runtime_camera_names}
        record_sensors = {name: sensors[name] for name in record_camera_names}
        base_camera_poses = _capture_camera_poses(sensors, runtime_camera_names)
        orbit_params = _sample_orbit_params()
        table_anchor_progress = _sample_table_anchor_progress()
        lookat_random_params = _sample_lookat_random_params() if args_cli.camera_motion_mode == "lookat_random" else None
        table_full_random_params = _sample_table_full_random_params() if args_cli.camera_motion_mode == "table_full_random" else None
        if args_cli.camera_motion_mode == "orbit":
            print(f"[INFO] Orbit camera params: {orbit_params}")
        if args_cli.camera_motion_mode == "lookat_random":
            print(f"[INFO] lookat_random camera params: {lookat_random_params}")
        if args_cli.camera_motion_mode == "table_full_random":
            print(f"[INFO] table_full_random camera params: {table_full_random_params}")
        if table_anchor_progress:
            print(f"[INFO] Table-facing anchor progress: {table_anchor_progress}")
        print("[INFO] Recording cameras:")
        for name, sensor in sensors.items():
            print(f"  - {name}: outputs={list(sensor.data.output.keys())}, size={args_cli.width}x{args_cli.height}")

        with h5py.File(hdf5_path, "w") as h5:
            h5.attrs["env_name"] = task_env
            h5.attrs["task"] = args_cli.task
            h5.attrs["instruction"] = str(env_cfg.instruction)
            h5.attrs["num_envs"] = env.num_envs
            h5.attrs["num_steps_requested"] = args_cli.num_steps
            h5.attrs["skip_initial_steps"] = skip_initial_steps
            h5.attrs["camera_names"] = json.dumps(record_camera_names)
            h5.attrs["runtime_camera_names"] = json.dumps(runtime_camera_names)

            demo = h5.create_group("data/demo_0")
            obs_group = demo.create_group("obs")
            pose_group = demo.create_group("camera_pose")
            for camera_name in record_camera_names:
                obs_group.create_group(camera_name)
                pose_group.create_group(camera_name)

            meta = {
                "task_env": task_env,
                "instruction": env_cfg.instruction,
                "camera_names": list(camera_names),
                "camera_data_types": ["rgb", "depth"],
                "width": args_cli.width,
                "height": args_cli.height,
                "record_every_env_steps": args_cli.record_every,
                "env_step_dt_s": args_cli.env_step_dt,
                "skip_initial_seconds": args_cli.skip_initial_seconds,
                "skip_initial_steps": skip_initial_steps,
                "recorded_dt_s": float(args_cli.record_every) * args_cli.env_step_dt,
                "recorded_fps_approx": 1.0 / (args_cli.env_step_dt * float(args_cli.record_every)),
                "left_right_baseline_m": args_cli.baseline,
                "rig_center_position": [args_cli.rig_x, args_cli.rig_y, args_cli.rig_z],
                "stereo_model": "parallel front-facing cameras, identical intrinsics and orientation",
                "include_wrist_camera": args_cli.include_wrist_camera,
                "include_wrist_stereo_camera": args_cli.include_wrist_stereo_camera,
                "exclude_policy_wrist_camera_from_output": args_cli.exclude_policy_wrist_camera_from_output,
                "wrist_stereo_baseline_m": args_cli.wrist_baseline,
                "randomize_background": args_cli.randomize_background,
                "background_seed": args_cli.background_seed,
                "lighting": args_cli.lighting,
                "randomize_camera_pose": args_cli.randomize_camera_pose,
                "camera_pos_range": args_cli.camera_pos_range,
                "camera_rot_range": args_cli.camera_rot_range,
                "moving_camera_pose": args_cli.moving_camera_pose,
                "camera_motion_mode": args_cli.camera_motion_mode,
                "camera_motion_xyz_m": [args_cli.camera_motion_x, args_cli.camera_motion_y, args_cli.camera_motion_z],
                "camera_motion_period_steps": args_cli.camera_motion_period,
                "camera_orbit_radius_m": args_cli.camera_orbit_radius,
                "camera_orbit_angle_deg": args_cli.camera_orbit_angle,
                "camera_orbit_height_m": args_cli.camera_orbit_height,
                "camera_orbit_start_deg": args_cli.camera_orbit_start_deg,
                "camera_orbit_randomize": args_cli.camera_orbit_randomize,
                "camera_orbit_sample_random": args_cli.camera_orbit_sample_random,
                "camera_orbit_sampled_params": orbit_params,
                "camera_orbit_radius_range": list(args_cli.camera_orbit_radius_range),
                "camera_orbit_angle_range": list(args_cli.camera_orbit_angle_range),
                "camera_orbit_start_range": list(args_cli.camera_orbit_start_range),
                "camera_orbit_height_range": list(args_cli.camera_orbit_height_range),
                "camera_look_at_x_range": list(args_cli.camera_look_at_x_range),
                "camera_look_at_y_range": list(args_cli.camera_look_at_y_range),
                "camera_look_at_z_range": list(args_cli.camera_look_at_z_range),
                "camera_orbit_jitter_seed": args_cli.camera_orbit_jitter_seed,
                "camera_orbit_radius_jitter_m": args_cli.camera_orbit_radius_jitter,
                "camera_orbit_height_jitter_m": args_cli.camera_orbit_height_jitter,
                "camera_orbit_angle_jitter_deg": args_cli.camera_orbit_angle_jitter,
                "camera_look_at_jitter_m": args_cli.camera_look_at_jitter,
                "camera_look_at": list(args_cli.camera_look_at),
                "camera_table_anchor_frames": args_cli.camera_table_anchor_frames,
                "camera_table_anchor_progress": table_anchor_progress,
                "camera_table_anchor_count": args_cli.camera_table_anchor_count,
                "camera_table_anchor_progress_range": list(args_cli.camera_table_anchor_progress_range),
                "camera_table_anchor_min_separation": args_cli.camera_table_anchor_min_separation,
                "camera_table_anchor_width": args_cli.camera_table_anchor_width,
                "camera_table_anchor_pos": list(args_cli.camera_table_anchor_pos),
                "camera_table_anchor_look_at": list(args_cli.camera_table_anchor_look_at),
                "camera_line_start": list(args_cli.camera_line_start),
                "camera_line_end": list(args_cli.camera_line_end),
                "camera_random_keyframes": args_cli.camera_random_keyframes,
                "camera_random_x_range": list(args_cli.camera_random_x_range),
                "camera_random_y_range": list(args_cli.camera_random_y_range),
                "camera_random_z_range": list(args_cli.camera_random_z_range),
                "camera_random_look_at_x_range": list(args_cli.camera_random_look_at_x_range),
                "camera_random_look_at_y_range": list(args_cli.camera_random_look_at_y_range),
                "camera_random_look_at_z_range": list(args_cli.camera_random_look_at_z_range),
                "camera_random_start_table": args_cli.camera_random_start_table,
                "camera_random_start_pos": list(args_cli.camera_random_start_pos),
                "camera_random_start_look_at": list(args_cli.camera_random_start_look_at),
                "camera_lookat_random_sampled_params": lookat_random_params,
                "camera_table_bounds_min": list(args_cli.camera_table_bounds_min),
                "camera_table_bounds_max": list(args_cli.camera_table_bounds_max),
                "camera_table_full_margin": args_cli.camera_table_full_margin,
                "camera_table_full_vertical_fov_deg": args_cli.camera_table_full_vertical_fov_deg,
                "camera_table_full_azimuth_range": list(args_cli.camera_table_full_azimuth_range),
                "camera_table_full_elevation_range": list(args_cli.camera_table_full_elevation_range),
                "camera_table_full_distance_range": list(args_cli.camera_table_full_distance_range),
                "camera_table_full_look_at_jitter": args_cli.camera_table_full_look_at_jitter,
                "camera_table_full_max_retreat": args_cli.camera_table_full_max_retreat,
                "camera_table_full_distance_padding": args_cli.camera_table_full_distance_padding,
                "camera_table_full_distance_change_per_step": args_cli.camera_table_full_distance_change_per_step,
                "camera_table_full_random_sampled_params": table_full_random_params,
                "add_geometry_background": args_cli.add_geometry_background,
                "geometry_background_seed": args_cli.geometry_background_seed,
                "geometry_background_style": args_cli.geometry_background_style,
                "geometry_background_scale": args_cli.geometry_background_scale,
                "geometry_background": geometry_background,
                "randomize_object_pose": args_cli.randomize_object_pose,
                "randomize_objects": args_cli.randomize_objects,
                "object_xy_range": args_cli.object_xy_range,
                "object_yaw_range": args_cli.object_yaw_range,
                "add_random_distractors": args_cli.add_random_distractors,
                "distractor_mode": args_cli.distractor_mode,
                "distractor_assets": args_cli.distractor_assets,
                "distractor_scale_range": list(args_cli.distractor_scale_range),
                "distractor_mass_range": list(args_cli.distractor_mass_range),
                "distractor_collider_approx": args_cli.distractor_collider_approx,
                "distractor_settle_steps": args_cli.distractor_settle_steps,
                "num_distractors": args_cli.num_distractors,
                "distractor_seed": args_cli.distractor_seed,
                "distractor_x_range": list(args_cli.distractor_x_range),
                "distractor_y_range": list(args_cli.distractor_y_range),
                "distractor_table_z": args_cli.distractor_table_z,
                "distractor_size_range": list(args_cli.distractor_size_range),
                "distractors": distractors,
                "action_hdf5": args_cli.action_hdf5,
                "action_demo": args_cli.action_demo,
                "policy_backend": args_cli.policy_backend,
                "remote_host": args_cli.remote_host,
                "remote_port": args_cli.remote_port,
                "remote_uri": args_cli.remote_uri,
                "open_loop_horizon": args_cli.open_loop_horizon,
                "action_source": "policy" if args_cli.policy_backend != "none" else ("hdf5_replay" if args_cli.action_hdf5 else "gripper_toggle"),
                "notes": "Camera orientation is quat_w_ros from Isaac Lab camera.data (IsaacLab order: w, x, y, z).",
            }
            _write_json_dataset(demo, "metadata_json", meta)

            toggle_gripper = False
            compression = args_cli.compression
            recorded_count = 0
            for step in range(args_cli.num_steps):
                if step % args_cli.toggle_every == 0:
                    toggle_gripper = not toggle_gripper
                    print(f"[Step {step:04d}] Gripper state: {'open' if toggle_gripper else 'closed'}")

                _apply_stereo_rig_motion(sensors, base_camera_poses, step, orbit_params, table_anchor_progress, lookat_random_params, table_full_random_params)

                if policy_client is not None:
                    ret = policy_client.infer(obs, str(env_cfg.instruction), env_id=0)
                    actions = torch.tensor(ret["action"], device=env.device, dtype=torch.float32).unsqueeze(0)
                elif replay_actions is not None:
                    action_np = replay_actions[min(step, len(replay_actions) - 1)]
                    actions = torch.tensor(action_np, device=env.device, dtype=torch.float32).unsqueeze(0)
                else:
                    current_joint_pos = robot.data.joint_pos[0, :7]
                    gripper_width = 0.0 if toggle_gripper else 0.785398163
                    gripper_action = torch.tensor([gripper_width], device=env.device)
                    actions = torch.cat([current_joint_pos, gripper_action]).unsqueeze(0)

                obs, _, term, trunc, _ = env.step(actions)

                if step < skip_initial_steps:
                    continue
                if (step - skip_initial_steps) % args_cli.record_every != 0:
                    continue

                _append_dataset(demo, "env_step", np.asarray(step, dtype=np.int32), compression)
                _append_dataset(demo, "actions", _to_numpy(actions[0]), compression)
                _append_dataset(demo, "terminated", _to_numpy(term[0]).astype(np.bool_), compression)
                _append_dataset(demo, "truncated", _to_numpy(trunc[0]).astype(np.bool_), compression)

                for camera_name, sensor in record_sensors.items():
                    cam_obs = obs_group[camera_name]
                    cam_pose = pose_group[camera_name]
                    _append_dataset(cam_obs, "rgb", _frame_from_sensor(sensor, "rgb"), compression)
                    _append_dataset(cam_obs, "depth", _frame_from_sensor(sensor, "depth"), compression)
                    _append_dataset(cam_pose, "position", _to_numpy(sensor.data.pos_w[0]).astype(np.float32), compression)
                    _append_dataset(cam_pose, "orientation_wxyz", _to_numpy(sensor.data.quat_w_ros[0]).astype(np.float32), compression)
                recorded_count += 1

            if "actions" not in demo or demo["actions"].shape[0] == 0:
                raise RuntimeError("No frames were recorded; check policy connection, action source, and camera outputs.")

            demo.attrs["num_samples"] = recorded_count
            demo.attrs["num_env_steps"] = args_cli.num_steps
            demo.attrs["record_every"] = args_cli.record_every
            h5["data"].attrs["total"] = recorded_count

        if args_cli.output_format in {"hdf5", "both"}:
            _save_hdf5_schema_summary(hdf5_path)
            print(f"\n[DONE] Wrote stereo RGB-D HDF5 dataset to: {hdf5_path}")

        if args_cli.output_format in {"directory", "both"}:
            scene_name = args_cli.scene_name or Path(args_cli.file_name).stem
            sequence_dirs = _convert_hdf5_to_directory(hdf5_path, output_dir, scene_name, record_camera_names, args_cli.overwrite)
            for sequence_dir in sequence_dirs:
                print(f"[DONE] Wrote stereo directory sequence to: {sequence_dir}")
            if args_cli.output_format == "directory":
                hdf5_path.unlink()
                print(f"[INFO] Removed temporary HDF5: {hdf5_path}")
    finally:
        try:
            if "policy_client" in locals() and policy_client is not None:
                policy_client.close()
        finally:
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    try:
        run()
    except BaseException as exc:
        print(f"Terminated with error ({type(exc).__name__}): {exc}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
