from __future__ import annotations

import numpy as np

from robolab.core.camera_trajectory import (
    AABBObstacle,
    CloseupMotionConfig,
    VisualQualityConfig,
    GeometryGateConfig,
    MotionLimits,
    PlannerConfig,
    TrajectoryContext,
    build_dynamic_context,
    camera_to_world_from_look_at,
    aggregate_depth_coverage,
    aggregate_multi_camera_depth_coverage,
    aggregate_visual_quality,
    geometry_gate,
    greedy_quality_diversity_selection,
    make_stereo_poses,
    motion_gate,
    plan_bounded_closeup_trajectory,
    project_stereo_clearance,
    sparse_quality_frame_indices,
    stereo_frame_quality,
    stereo_depth_frame_quality,
    strategy_aware_shortlist,
    trajectory_overlap_ratio,
    solve_episode,
    stable_seed,
    trajectory_motion_metrics,
)
from robolab.core import camera_trajectory as planner


def context(frame_count: int = 120) -> TrajectoryContext:
    progress = np.linspace(0.0, 1.0, frame_count)
    body = np.repeat([[0.15, 0.0, 0.25]], frame_count, axis=0)
    action = np.column_stack(
        (
            0.50 + 0.08 * np.sin(progress * np.pi),
            0.12 * np.sin(progress * 2.0 * np.pi),
            0.70 + 0.05 * progress,
        )
    )
    subject = 0.45 * body + 0.55 * action
    subject[:, 2] += 0.12
    return TrajectoryContext(body, action, subject)


def test_dynamic_context_tracks_end_effector_and_builds_reference_camera() -> None:
    frames = 24
    body = np.repeat([[0.1, 0.0, 0.2]], frames, axis=0)
    action = np.column_stack(
        (
            np.linspace(0.35, 0.70, frames),
            0.08 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames)),
            np.linspace(0.65, 0.82, frames),
        )
    )
    dynamic = build_dynamic_context(body, action)
    np.testing.assert_allclose(dynamic.subject_focus, 0.45 * body + 0.55 * action + [0.0, 0.0, 0.12])
    assert dynamic.reference_camera.shape == (frames, 4, 4)
    assert np.linalg.norm(np.ptp(dynamic.action_anchor, axis=0)) > 0.3
    for rotation in dynamic.reference_camera[:, :3, :3]:
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7)


def test_stable_seed_is_repeatable_and_order_sensitive() -> None:
    assert stable_seed(10, "task", 3) == stable_seed(10, "task", 3)
    assert stable_seed(10, "task", 3) != stable_seed(3, "task", 10)


def test_opencv_camera_axes_and_stereo_baseline() -> None:
    positions = np.array([[1.0, 0.0, 1.0], [0.8, -0.2, 1.2]])
    targets = np.array([[0.0, 0.0, 0.5], [0.2, 0.1, 0.6]])
    poses = camera_to_world_from_look_at(positions, targets)
    for pose, position, target in zip(poses, positions, targets):
        rotation = pose[:3, :3]
        expected_forward = (target - position) / np.linalg.norm(target - position)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1.0e-7)
        np.testing.assert_allclose(rotation[:, 2], expected_forward, atol=1.0e-7)
        assert rotation[2, 1] < 0.0  # local +y points down for this camera pose

    left, right = make_stereo_poses(poses, 0.16)
    np.testing.assert_allclose(np.linalg.norm(left[:, :3, 3] - right[:, :3, 3], axis=1), 0.16)
    np.testing.assert_allclose(left[:, :3, :3], right[:, :3, :3])


def test_bounded_closeup_trajectory_is_moving_deterministic_and_framing_safe() -> None:
    frames = 120
    progress = np.linspace(0.0, 1.0, frames)
    action = np.column_stack(
        (
            0.55 + 0.10 * np.sin(2.0 * np.pi * progress),
            0.12 * np.cos(2.0 * np.pi * progress),
            0.42 + 0.06 * np.sin(4.0 * np.pi * progress),
        )
    )
    base_position = np.asarray([0.82, 0.0, 0.75])
    base_target = np.asarray([0.55, 0.0, 0.40])
    config = CloseupMotionConfig()
    first = plan_bounded_closeup_trajectory(
        base_position, base_target, frames, 917, action_anchor=action, config=config
    )
    repeated = plan_bounded_closeup_trajectory(
        base_position, base_target, frames, 917, action_anchor=action, config=config
    )
    different = plan_bounded_closeup_trajectory(
        base_position, base_target, frames, 918, action_anchor=action, config=config
    )

    np.testing.assert_array_equal(first.center_camera_to_world, repeated.center_camera_to_world)
    assert not np.array_equal(first.center_camera_to_world, different.center_camera_to_world)
    position_delta = np.abs(first.positions - base_position)
    target_delta = np.abs(first.actual_look_at_points - base_target)
    assert np.all(position_delta <= np.asarray(config.position_jitter) + 1.0e-9)
    assert np.all(target_delta <= np.asarray(config.target_jitter) + 1.0e-9)
    metrics = trajectory_motion_metrics(first)
    assert metrics["position_span"] > 0.05
    assert metrics["position_span"] < 0.40
    assert motion_gate(first, config.motion_limits) == (True, "ok")
    np.testing.assert_allclose(
        first.actual_look_at_points[:, 2] - base_target[2],
        -config.vertical_pitch_coupling * (first.positions[:, 2] - base_position[2]),
        atol=1.0e-9,
    )

    vertical_only = plan_bounded_closeup_trajectory(
        base_position,
        base_target,
        frames,
        919,
        config=CloseupMotionConfig(
            position_jitter=(0.0, 0.0, 0.05),
            target_jitter=(0.0, 0.0, 0.03),
        ),
    )
    horizontal_distance = np.linalg.norm(
        vertical_only.positions[:, :2] - vertical_only.actual_look_at_points[:, :2], axis=1
    )
    downward_pitch = np.arctan2(
        vertical_only.positions[:, 2] - vertical_only.actual_look_at_points[:, 2],
        horizontal_distance,
    )
    lowest = int(np.argmin(vertical_only.positions[:, 2]))
    highest = int(np.argmax(vertical_only.positions[:, 2]))
    assert downward_pitch[lowest] < downward_pitch[highest]
    for pose, target in zip(first.center_camera_to_world, first.actual_look_at_points):
        expected_forward = target - pose[:3, 3]
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(pose[:3, 2], expected_forward, atol=1.0e-7)


def test_closeup_spherical_view_reaches_near_far_and_low_high_views() -> None:
    base_position = np.array([0.82, 0.0, 0.75])
    base_target = np.array([0.55, 0.0, 0.40])
    config = CloseupMotionConfig(
        view_distance_range=(0.32, 0.72),
        view_elevation_range_deg=(8.0, 78.0),
        view_azimuth_range_deg=(-38.0, 38.0),
        view_target_z=0.08,
        knot_count=6,
    )
    trajectory = plan_bounded_closeup_trajectory(
        base_position, base_target, 300, 12345, config=config
    )
    delta = trajectory.positions - trajectory.actual_look_at_points
    distance = np.linalg.norm(delta, axis=1)
    elevation = np.degrees(np.arctan2(delta[:, 2], np.linalg.norm(delta[:, :2], axis=1)))
    azimuth = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))

    np.testing.assert_allclose(trajectory.actual_look_at_points[:, 2], 0.08)
    assert distance.min() <= 0.34
    assert distance.max() >= 0.70
    assert elevation.min() <= 10.0
    assert elevation.max() >= 76.0
    assert azimuth.min() <= -35.5
    assert azimuth.max() >= 35.5
    overhead = elevation >= 68.0
    assert np.any(overhead)
    assert distance[overhead].min() >= 0.55 - 1.0e-4
    metrics = trajectory_motion_metrics(trajectory)
    assert metrics["max_rotation_step_change_deg"] < 0.25
    assert trajectory.diagnostics["smoothing_window_frames"] == 31
    assert trajectory.diagnostics["overhead_distance_adjusted_frames"] > 0
    assert motion_gate(trajectory, config.motion_limits) == (True, "ok")


def test_closeup_spherical_view_retreats_from_moving_arm_without_reframing() -> None:
    base_position = np.array([0.82, 0.0, 0.75])
    base_target = np.array([0.55, 0.0, 0.40])
    action = np.repeat(np.array([[0.55, 0.0, 0.08]]), 300, axis=0)
    common = dict(
        target_jitter=(0.0, 0.0, 0.0),
        action_follow_gain=0.0,
        view_distance_range=(0.32, 0.72),
        view_elevation_range_deg=(8.0, 78.0),
        view_azimuth_range_deg=(-38.0, 38.0),
        view_target_z=0.08,
        knot_count=6,
    )
    unrestricted = plan_bounded_closeup_trajectory(
        base_position,
        base_target,
        300,
        12345,
        action_anchor=action,
        config=CloseupMotionConfig(**common, min_action_camera_distance=0.0),
    )
    protected_config = CloseupMotionConfig(**common, min_action_camera_distance=0.45)
    protected = plan_bounded_closeup_trajectory(
        base_position,
        base_target,
        300,
        12345,
        action_anchor=action,
        config=protected_config,
    )

    clearance = np.linalg.norm(protected.positions - action, axis=1)
    assert clearance.min() >= 0.45 - 1.0e-4
    assert protected.diagnostics["action_clearance_adjusted_frames"] > 0
    assert protected.diagnostics["achieved_action_camera_distance_m"] >= 0.45 - 1.0e-4
    direction_cosine = np.sum(
        protected.center_camera_to_world[:, :3, 2]
        * unrestricted.center_camera_to_world[:, :3, 2],
        axis=1,
    )
    direction_difference_deg = np.degrees(
        np.arccos(np.clip(direction_cosine, -1.0, 1.0))
    )
    assert direction_difference_deg.max() < 1.0
    np.testing.assert_allclose(
        protected.actual_look_at_points,
        unrestricted.actual_look_at_points,
    )
    assert motion_gate(protected, protected_config.motion_limits) == (True, "ok")


def test_candidate_pool_is_deterministic_and_seeded() -> None:
    cfg = PlannerConfig(candidate_count=12)
    first = solve_episode(context(), 1234, config=cfg)
    repeated = solve_episode(context(), 1234, config=cfg)
    different = solve_episode(context(), 1235, config=cfg)
    np.testing.assert_array_equal(first.center_camera_to_world, repeated.center_camera_to_world)
    np.testing.assert_array_equal(first.composition_offsets, repeated.composition_offsets)
    assert not np.array_equal(first.center_camera_to_world, different.center_camera_to_world)


def test_motion_limits_hold_for_many_seeds() -> None:
    limits = MotionLimits(
        max_translation_step=0.03,
        max_translation_accel=0.01,
        max_rotation_step_deg=2.5,
        max_rotation_accel_deg=0.8,
        max_focus_step=0.025,
        max_focus_accel=0.01,
    )
    cfg = PlannerConfig(candidate_count=10, motion_limits=limits)
    for seed in range(20):
        trajectory = solve_episode(context(), seed, config=cfg)
        ok, reason = motion_gate(trajectory, limits)
        assert ok, (seed, reason, trajectory_motion_metrics(trajectory))
        metrics = trajectory_motion_metrics(trajectory)
        assert metrics["max_translation_step"] <= limits.max_translation_step * 1.05
        assert metrics["max_rotation_step_deg"] <= limits.max_rotation_step_deg * 1.05


def test_strategy_and_random_walk_modes_have_seed_diversity() -> None:
    cfg = PlannerConfig(candidate_count=1)
    short_context = context(40)
    strategies = {"orbit", "random_walk", "agent_follow", "focus_follow"}
    for index, strategy in enumerate(strategies):
        positions, targets, _ = planner._generate_strategy(
            short_context,
            strategy,
            np.random.default_rng(index),
            cfg,
        )
        assert positions.shape == targets.shape == (40, 3)

    random_walk_modes = set()
    for seed in range(200):
        _, _, metadata = planner._generate_strategy(
            short_context,
            "random_walk",
            np.random.default_rng(seed),
            cfg,
        )
        random_walk_modes.add(metadata["random_walk_mode"])
    assert random_walk_modes == {"arc_jitter", "lateral_truck", "dolly", "spiral", "focus_shift"}


def test_short_episode_degrades_to_single_focus_follow_strategy() -> None:
    trajectory = solve_episode(context(5), 99, config=PlannerConfig(candidate_count=3))
    assert trajectory.strategy_names == ["focus_follow"]
    assert trajectory.center_camera_to_world.shape == (5, 4, 4)
    assert trajectory.strategy_id_per_frame.tolist() == [0] * 5


def test_hybrid_candidates_preserve_segments_and_frame_labels() -> None:
    hybrid = None
    for seed in range(40):
        trajectory = solve_episode(context(60), seed, config=PlannerConfig(candidate_count=1))
        if len(trajectory.strategy_names) > 1:
            hybrid = trajectory
            break
    assert hybrid is not None
    assert len(hybrid.segments) == len(hybrid.strategy_names)
    assert hybrid.segments[0]["start"] == 0
    assert hybrid.segments[-1]["end"] == 60
    assert set(hybrid.strategy_id_per_frame) == set(range(len(hybrid.strategy_names)))
    lengths = np.bincount(hybrid.strategy_id_per_frame)
    assert np.all(lengths >= 12)
    assert np.all(lengths <= 36)


def test_composition_offsets_change_actual_view_direction() -> None:
    centered = solve_episode(context(), 44, config=PlannerConfig(candidate_count=4), profile="centered")
    off_center = solve_episode(context(), 44, config=PlannerConfig(candidate_count=4), profile="off_center")
    assert np.max(np.abs(off_center.composition_offsets[:, 0])) > 0.25
    assert not np.array_equal(centered.actual_look_at_points, off_center.actual_look_at_points)


def _straight_trajectory(positions: np.ndarray, targets: np.ndarray):
    frame_count = len(positions)
    return planner.CameraTrajectory(
        center_camera_to_world=camera_to_world_from_look_at(positions, targets),
        actual_look_at_points=targets,
        composition_offsets=np.zeros((frame_count, 2)),
        strategy_id_per_frame=np.zeros(frame_count, dtype=np.int32),
        strategy_names=["focus_follow"],
        segments=[{"start": 0, "end": frame_count}],
        composition_profile="centered",
        trajectory_seed=1,
        strategy_seeds=[2],
        strategy_metadata=[{}],
    )


def test_clearance_projection_checks_each_stereo_eye_and_preserves_target() -> None:
    positions = np.repeat([[0.0, 0.0, 0.8]], 20, axis=0)
    targets = np.repeat([[1.0, 0.0, 0.7]], 20, axis=0)
    # This box is close only to the right eye for the chosen optical x axis.
    obstacle = AABBObstacle([-0.04, -0.02, 0.74], [0.04, 0.07, 0.86], "scene", "one-eye")
    config = GeometryGateConfig(baseline=0.16, scene_clearance=0.06)
    projected, projected_targets, stats = project_stereo_clearance(
        positions, targets, [obstacle], config
    )
    assert stats["max_displacement_m"] > 0.0
    assert stats["max_displacement_m"] <= config.projection_max_displacement + 1.0e-8
    np.testing.assert_array_equal(projected_targets, targets)
    trajectory = _straight_trajectory(projected, projected_targets)
    result = geometry_gate(trajectory, targets, [obstacle], config)
    assert result.diagnostics["stereo_checked"] is True


def test_geometry_gate_rejects_scene_line_of_sight_occlusion() -> None:
    frames = 12
    positions = np.repeat([[0.0, 0.0, 0.8]], frames, axis=0)
    targets = np.repeat([[1.0, 0.0, 0.8]], frames, axis=0)
    wall = AABBObstacle([0.45, -0.4, 0.2], [0.55, 0.4, 1.4], "scene", "wall")
    result = geometry_gate(
        _straight_trajectory(positions, targets), targets, [wall],
        GeometryGateConfig(baseline=0.08, scene_clearance=0.02),
    )
    assert not result.ok
    assert result.reason in {"bad_frame_ratio", "consecutive_bad_frames"}
    assert result.diagnostics["reason_counts"]["line_of_sight"] == frames


def test_geometry_gate_uses_smaller_robot_clearance_but_never_relaxes_scene_clearance() -> None:
    frames = 8
    positions = np.repeat([[0.0, 0.0, 0.8]], frames, axis=0)
    targets = np.repeat([[1.0, 0.0, 0.8]], frames, axis=0)
    near = [-0.02, 0.065, 0.77]
    far = [0.02, 0.095, 0.83]
    config = GeometryGateConfig(baseline=0.08, robot_clearance=0.01, scene_clearance=0.05)
    trajectory = _straight_trajectory(positions, targets)
    robot_result = geometry_gate(
        trajectory, targets, [AABBObstacle(near, far, "robot")], config
    )
    scene_result = geometry_gate(
        trajectory, targets, [AABBObstacle(near, far, "scene")], config
    )
    assert robot_result.ok
    assert not scene_result.ok
    assert scene_result.reason == "scene_clearance"


def test_geometry_gate_allows_one_isolated_bad_framing_frame() -> None:
    frames = 20
    positions = np.repeat([[0.0, 0.0, 0.8]], frames, axis=0)
    targets = np.repeat([[1.0, 0.0, 0.8]], frames, axis=0)
    targets[10] = [1.0, 4.0, 0.8]
    result = geometry_gate(
        _straight_trajectory(positions, np.repeat([[1.0, 0.0, 0.8]], frames, axis=0)),
        targets,
        [],
        GeometryGateConfig(max_bad_frame_ratio=0.10, max_consecutive_bad_frames=1),
    )
    assert result.ok
    assert result.bad_frame_ratio == 0.05
    assert result.max_consecutive_bad_frames == 1


def test_sparse_quality_frames_include_uniform_and_view_extremes() -> None:
    trajectory = solve_episode(context(80), 811, config=PlannerConfig(candidate_count=2))
    frames = sparse_quality_frame_indices(trajectory)
    assert frames[0] == 0
    assert frames[-1] == 79
    delta = trajectory.positions - trajectory.actual_look_at_points
    azimuth = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    radius = np.linalg.norm(delta[:, :2], axis=1)
    for expected in (np.argmin(azimuth), np.argmax(azimuth), np.argmin(radius), np.argmin(trajectory.positions[:, 2])):
        assert int(expected) in frames


def test_stereo_quality_uses_worse_eye_and_marks_missing_segmentation() -> None:
    rgb = np.indices((32, 48)).sum(axis=0).astype(np.uint8)
    rgb = np.repeat(rgb[..., None] * 6, 3, axis=2)
    good_depth = np.ones((32, 48), dtype=np.float32)
    half_depth = good_depth.copy()
    half_depth[:, :30] = 0.0
    metrics = stereo_frame_quality(rgb, rgb, good_depth, half_depth)
    assert metrics["weak_background_ratio"] > 0.60
    assert metrics["subject_pixel_ratio"] is None
    assert metrics["subject_segmentation_available"] is False
    assert metrics["stereo_conservative_aggregation"] is True


def test_depth_coverage_gate_is_independent_of_texture() -> None:
    good = {"weak_background_ratio": 0.05, "texture_score": 0.0}
    bad = {"weak_background_ratio": 0.60, "texture_score": 1.0}
    accepted = aggregate_depth_coverage([good] * 9 + [bad])
    assert accepted["ok"]
    rejected = aggregate_depth_coverage([good] * 8 + [bad] * 2)
    assert not rejected["ok"]
    assert rejected["reason"] == "depth_bad_frame_ratio"


def test_stereo_depth_quality_uses_worse_eye_and_rejects_invalid_values() -> None:
    left = np.ones((20, 30), dtype=np.float32)
    right = left.copy()
    right[:, :9] = 0.0
    right[0, 9:12] = np.nan
    metrics = stereo_depth_frame_quality(left, right)
    expected = 1.0 - (9 * 20 + 3) / (20 * 30)
    assert np.isclose(metrics["valid_depth_ratio"], expected)
    assert np.isclose(metrics["weak_background_ratio"], 1.0 - expected)
    assert metrics["left_valid_depth_ratio"] == 1.0
    assert np.isclose(metrics["right_valid_depth_ratio"], expected)
    assert metrics["stereo_conservative_aggregation"] is True


def test_multi_camera_depth_gate_rejects_one_bad_stereo_rig() -> None:
    good = {"weak_background_ratio": 0.02}
    bad = {"weak_background_ratio": 0.45}
    result = aggregate_multi_camera_depth_coverage(
        {
            "camera00": [good] * 10,
            "camera01": [good] * 10,
            "camera02": [good] * 10,
            "camera03": [good] * 8 + [bad] * 2,
            "camera04": [good] * 10,
        }
    )
    assert not result["ok"]
    assert result["reason"] == "camera_depth_coverage_failed"
    assert result["failed_cameras"] == ["camera03"]
    assert result["worst_camera"] == "camera03"
    assert result["camera_count"] == 5


def test_visual_quality_rejects_consecutive_invalid_depth_frames() -> None:
    good = {"weak_background_ratio": 0.2, "texture_score": 0.3, "subject_segmentation_available": False}
    bad = {"weak_background_ratio": 0.95, "texture_score": 0.01, "subject_segmentation_available": False}
    result = aggregate_visual_quality(
        [good] * 4 + [bad, bad] + [good] * 4,
        VisualQualityConfig(max_bad_frame_ratio=0.25, max_consecutive_bad_frames=1),
    )
    assert not result.ok
    assert result.reason == "visual_consecutive_bad_frames"
    assert result.diagnostics["subject_segmentation_available"] is False


def test_visual_quality_default_keeps_valid_low_texture_surfaces() -> None:
    frame = {
        "weak_background_ratio": 0.001,
        "texture_score": 0.0733,
        "subject_segmentation_available": False,
    }
    result = aggregate_visual_quality([frame] * 20, VisualQualityConfig())
    assert result.ok
    assert result.reason == "ok"


def test_strategy_shortlist_protects_random_walk_candidates() -> None:
    candidates = []
    for seed in range(80):
        trajectory = solve_episode(context(30), seed, config=PlannerConfig(candidate_count=1))
        quality = 20.0 if "random_walk" not in trajectory.strategy_names else 1.0
        candidates.append((trajectory, quality))
    shortlist = strategy_aware_shortlist(candidates, 10)
    assert len(shortlist) == 10
    assert sum("random_walk" in trajectory.strategy_names for trajectory, _ in shortlist) >= 4


def test_overlap_gate_keeps_different_trajectory_in_multi_selection() -> None:
    first = solve_episode(context(40), 91, config=PlannerConfig(candidate_count=2))
    duplicate = planner.replace(first)
    duplicate.center_camera_to_world = first.center_camera_to_world.copy()
    duplicate.actual_look_at_points = first.actual_look_at_points.copy()
    duplicate.composition_offsets = first.composition_offsets.copy()
    duplicate.strategy_id_per_frame = first.strategy_id_per_frame.copy()
    second = solve_episode(context(40), 177, config=PlannerConfig(candidate_count=2))
    assert trajectory_overlap_ratio(first, duplicate) == 1.0
    selected = greedy_quality_diversity_selection(
        [(first, 10.0), (duplicate, 9.9), (second, 9.0)], requested_count=2
    )
    assert len(selected) == 2
    assert not (selected[0] is first and selected[1] is duplicate)
