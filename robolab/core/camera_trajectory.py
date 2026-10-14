"""Platform-independent external camera trajectory planning.

The planner uses OpenCV optical camera coordinates throughout: local +x points
right, +y points down, and +z points forward.  Simulator-specific pose
conversions belong at the adapter boundary.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


Array = np.ndarray
TrajectoryProjector = Callable[[Array, Array], tuple[Array, Array] | tuple[Array, Array, dict]]
TrajectoryValidator = Callable[["CameraTrajectory"], tuple[bool, float, str]]


def stable_seed(*parts: object) -> int:
    """Return a process-independent 63-bit seed."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _as_track(value: Array, frame_count: int, name: str) -> Array:
    track = np.asarray(value, dtype=np.float64)
    if track.shape == (3,):
        track = np.repeat(track[None, :], frame_count, axis=0)
    if track.shape != (frame_count, 3):
        raise ValueError(f"{name} must have shape ({frame_count}, 3), got {track.shape}")
    if not np.all(np.isfinite(track)):
        raise ValueError(f"{name} contains non-finite values")
    return track


@dataclass(slots=True)
class TrajectoryContext:
    body_anchor: Array
    action_anchor: Array
    subject_focus: Array
    reference_camera: Array | None = None

    def __post_init__(self) -> None:
        subject = np.asarray(self.subject_focus)
        if subject.ndim != 2 or subject.shape[1] != 3 or len(subject) == 0:
            raise ValueError("subject_focus must have shape (N, 3) with N > 0")
        frame_count = len(subject)
        self.body_anchor = _as_track(self.body_anchor, frame_count, "body_anchor")
        self.action_anchor = _as_track(self.action_anchor, frame_count, "action_anchor")
        self.subject_focus = _as_track(subject, frame_count, "subject_focus")
        if self.reference_camera is not None:
            reference = np.asarray(self.reference_camera, dtype=np.float64)
            if reference.shape != (frame_count, 4, 4):
                raise ValueError(f"reference_camera must have shape ({frame_count}, 4, 4)")
            self.reference_camera = reference

    @property
    def frame_count(self) -> int:
        return len(self.subject_focus)


@dataclass(slots=True, frozen=True)
class MotionLimits:
    max_translation_step: float = 0.045
    max_translation_accel: float = 0.018
    max_rotation_step_deg: float = 4.0
    max_rotation_accel_deg: float = 1.5
    max_focus_step: float = 0.035
    max_focus_accel: float = 0.015


@dataclass(slots=True, frozen=True)
class PlannerConfig:
    candidate_count: int = 16
    knot_count: int = 5
    radius_range: tuple[float, float] = (0.65, 1.35)
    elevation_range_deg: tuple[float, float] = (28.0, 62.0)
    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0)
    vertical_fov_deg: float = 75.0
    aspect_ratio: float = 16.0 / 9.0
    strategy_weights: tuple[tuple[str, float], ...] = (
        ("orbit", 0.25),
        ("random_walk", 0.55),
        ("agent_follow", 0.10),
        ("focus_follow", 0.10),
    )
    motion_limits: MotionLimits = MotionLimits()


@dataclass(slots=True, frozen=True)
class CloseupMotionConfig:
    """Hard-bounded motion for a close operation-view camera.

    The bounds are deliberately anisotropic: the camera may truck sideways a
    little more than it may dolly or move vertically.  This preserves the
    framing of a previously validated close-up pose while still producing a
    useful, non-static camera trajectory.
    """

    position_jitter: tuple[float, float, float] = (0.080, 0.140, 0.100)
    target_jitter: tuple[float, float, float] = (0.035, 0.065, 0.040)
    action_follow_gain: float = 0.12
    vertical_pitch_coupling: float = 0.35
    view_distance_range: tuple[float, float] | None = None
    view_elevation_range_deg: tuple[float, float] | None = None
    view_azimuth_range_deg: tuple[float, float] | None = None
    view_target_z: float | None = None
    min_action_camera_distance: float = 0.45
    smoothing_window: int = 31
    overhead_elevation_threshold_deg: float = 68.0
    overhead_transition_deg: float = 20.0
    overhead_min_distance: float = 0.55
    knot_count: int = 6
    motion_limits: MotionLimits = MotionLimits(
        max_translation_step=0.015,
        max_translation_accel=0.006,
        max_rotation_step_deg=2.5,
        max_rotation_accel_deg=0.8,
        max_focus_step=0.010,
        max_focus_accel=0.004,
    )


@dataclass(slots=True)
class CameraTrajectory:
    center_camera_to_world: Array
    actual_look_at_points: Array
    composition_offsets: Array
    strategy_id_per_frame: Array
    strategy_names: list[str]
    segments: list[dict]
    composition_profile: str
    trajectory_seed: int
    strategy_seeds: list[int]
    strategy_metadata: list[dict]
    clearance_projection_stats: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def positions(self) -> Array:
        return self.center_camera_to_world[:, :3, 3]


def _normalize(vector: Array, fallback: Array | None = None) -> Array:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-9:
        return vector / norm
    if fallback is None:
        raise ValueError("cannot normalize a zero vector")
    return _normalize(fallback)


def camera_to_world_from_look_at(positions: Array, targets: Array) -> Array:
    """Build OpenCV optical camera-to-world matrices."""
    positions = np.asarray(positions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(positions), axis=0)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for index, (position, target) in enumerate(zip(positions, targets)):
        forward = _normalize(target - position, np.array([-1.0, 0.0, -0.2]))
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1.0e-7:
            right = np.cross(forward, fallback_up)
        right = _normalize(right)
        down = _normalize(np.cross(forward, right))
        poses[index, :3, :3] = np.column_stack((right, down, forward))
        poses[index, :3, 3] = position
    return poses


def build_dynamic_context(
    body_anchor: Array,
    action_anchor: Array,
    *,
    reference_offset: tuple[float, float, float] = (-0.45, -0.65, 1.10),
) -> TrajectoryContext:
    """Build planner context from a recorded embodied-state trajectory.

    The synthetic reference camera is a stable shoulder-like fallback for
    platforms that do not expose an official follow camera.
    """
    body = np.asarray(body_anchor, dtype=np.float64)
    action = np.asarray(action_anchor, dtype=np.float64)
    if body.shape != action.shape or body.ndim != 2 or body.shape[1] != 3:
        raise ValueError("body_anchor and action_anchor must both have shape (N, 3)")
    subject = 0.45 * body + 0.55 * action
    subject[:, 2] += 0.12
    reference_positions = body + np.asarray(reference_offset, dtype=np.float64)
    reference = camera_to_world_from_look_at(reference_positions, subject)
    return TrajectoryContext(
        body_anchor=body,
        action_anchor=action,
        subject_focus=subject,
        reference_camera=reference,
    )


def make_stereo_poses(center_camera_to_world: Array, baseline: float) -> tuple[Array, Array]:
    center = np.asarray(center_camera_to_world, dtype=np.float64)
    left = center.copy()
    right = center.copy()
    half_offset = center[:, :3, 0] * (0.5 * float(baseline))
    left[:, :3, 3] -= half_offset
    right[:, :3, 3] += half_offset
    return left, right


def _hann_smooth(values: Array, width: int = 7) -> Array:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    width = min(int(width), len(values) if len(values) % 2 else len(values) - 1)
    width = max(width, 3)
    if width % 2 == 0:
        width -= 1
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])], axis=1)


def _natural_cubic_interpolate(knot_t: Array, knot_v: Array, sample_t: Array) -> Array:
    """Evaluate a one-dimensional natural cubic spline without SciPy."""
    knot_t = np.asarray(knot_t, dtype=np.float64)
    knot_v = np.asarray(knot_v, dtype=np.float64)
    count = len(knot_t)
    if count < 3:
        return np.interp(sample_t, knot_t, knot_v)
    widths = np.diff(knot_t)
    system = np.zeros((count - 2, count - 2), dtype=np.float64)
    rhs = np.zeros(count - 2, dtype=np.float64)
    for row in range(count - 2):
        if row:
            system[row, row - 1] = widths[row]
        system[row, row] = 2.0 * (widths[row] + widths[row + 1])
        if row + 1 < count - 2:
            system[row, row + 1] = widths[row + 1]
        rhs[row] = 6.0 * (
            (knot_v[row + 2] - knot_v[row + 1]) / widths[row + 1]
            - (knot_v[row + 1] - knot_v[row]) / widths[row]
        )
    # The matrix is tridiagonal. Thomas elimination avoids launching a
    # heavyweight BLAS solve for the tiny spline systems used per trajectory.
    interior = count - 2
    diagonal = np.diag(system).copy()
    upper = np.diag(system, 1).copy()
    lower = np.diag(system, -1).copy()
    solution = rhs.copy()
    for index in range(1, interior):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        solution[index] -= factor * solution[index - 1]
    solution[-1] /= diagonal[-1]
    for index in range(interior - 2, -1, -1):
        solution[index] = (solution[index] - upper[index] * solution[index + 1]) / diagonal[index]

    second = np.zeros(count, dtype=np.float64)
    second[1:-1] = solution
    segment = np.clip(np.searchsorted(knot_t, sample_t, side="right") - 1, 0, count - 2)
    left = knot_t[segment]
    right = knot_t[segment + 1]
    width = right - left
    a = (right - sample_t) / width
    b = (sample_t - left) / width
    return (
        a * knot_v[segment]
        + b * knot_v[segment + 1]
        + ((a**3 - a) * second[segment] + (b**3 - b) * second[segment + 1]) * width**2 / 6.0
    )


def smooth_random_signal(rng: np.random.Generator, frame_count: int, sigma: float, knot_count: int) -> Array:
    knot_count = min(max(2 if frame_count < 4 else 4, int(knot_count)), frame_count)
    knot_t = np.linspace(0.0, frame_count - 1, knot_count)
    knot_v = rng.normal(0.0, sigma, size=(knot_count, 1))
    knot_v[0] *= 0.25
    knot_v[-1] *= 0.25
    values = _natural_cubic_interpolate(knot_t, knot_v[:, 0], np.arange(frame_count))[:, None]
    return _hann_smooth(values, 7)[:, 0]


def _bounded_smooth_signal(
    rng: np.random.Generator,
    frame_count: int,
    bound: float,
    knot_count: int,
) -> Array:
    """Generate a smooth signal whose absolute value never exceeds ``bound``."""
    bound = max(0.0, float(bound))
    if bound == 0.0:
        return np.zeros(frame_count, dtype=np.float64)
    signal = smooth_random_signal(rng, frame_count, 1.0, knot_count)
    low = float(np.min(signal, initial=0.0))
    high = float(np.max(signal, initial=0.0))
    midpoint = 0.5 * (low + high)
    half_span = 0.5 * (high - low)
    if half_span < 1.0e-9:
        # This is only reachable for extremely short/degenerate inputs.
        signal = np.sin(np.linspace(0.0, 2.0 * math.pi, frame_count))
        midpoint = 0.0
        half_span = float(np.max(np.abs(signal), initial=1.0))
    # Normalize by peak-to-peak range rather than only the largest absolute
    # peak. This guarantees a meaningful excursion in both directions for
    # every seed while retaining a hard absolute bound.
    amplitude = bound * float(rng.uniform(0.75, 1.0))
    return (signal - midpoint) * (amplitude / half_span)


def plan_bounded_closeup_trajectory(
    base_position: Array,
    base_target: Array,
    frame_count: int,
    seed: int,
    *,
    action_anchor: Array | None = None,
    config: CloseupMotionConfig = CloseupMotionConfig(),
) -> CameraTrajectory:
    """Plan a deterministic, smooth close-up camera trajectory.

    Position and look-at coordinates remain inside hard axis-aligned bounds
    around the supplied, visually validated base pose.  A small centered term
    from the end-effector trajectory lets the view follow the manipulation
    without turning camera03 into a wrist camera or exposing much more room.
    """
    if int(frame_count) < 2:
        raise ValueError("frame_count must be at least 2")
    frame_count = int(frame_count)
    base_position = np.asarray(base_position, dtype=np.float64)
    base_target = np.asarray(base_target, dtype=np.float64)
    if base_position.shape != (3,) or base_target.shape != (3,):
        raise ValueError("base_position and base_target must each have shape (3,)")
    if not np.all(np.isfinite(base_position)) or not np.all(np.isfinite(base_target)):
        raise ValueError("base_position and base_target must be finite")

    position_bounds = np.maximum(np.asarray(config.position_jitter, dtype=np.float64), 0.0)
    target_bounds = np.maximum(np.asarray(config.target_jitter, dtype=np.float64), 0.0)
    if position_bounds.shape != (3,) or target_bounds.shape != (3,):
        raise ValueError("close-up position and target jitter must each have three values")

    rng = np.random.default_rng(int(seed))
    knots = min(max(2, int(config.knot_count)), frame_count)
    position_offsets = np.column_stack(
        [_bounded_smooth_signal(rng, frame_count, bound, knots) for bound in position_bounds]
    )
    target_random_scales = (0.55, 0.55, 0.0)
    target_offsets = np.column_stack([
        _bounded_smooth_signal(rng, frame_count, bound * scale, knots)
        for bound, scale in zip(target_bounds, target_random_scales)
    ])

    action = None
    follow_applied = action_anchor is not None and float(config.action_follow_gain) > 0.0
    if follow_applied:
        action = _as_track(action_anchor, frame_count, "action_anchor")
        centered_action = action - np.median(action, axis=0, keepdims=True)
        action_follow = float(config.action_follow_gain) * centered_action
        target_offsets[:, :2] += action_follow[:, :2]

    spherical_view = (
        config.view_distance_range is not None
        and config.view_elevation_range_deg is not None
        and config.view_azimuth_range_deg is not None
    )
    pitch_coupling = max(0.0, float(config.vertical_pitch_coupling))
    if spherical_view:
        distance_range = np.sort(np.asarray(config.view_distance_range, dtype=np.float64))
        elevation_range = np.sort(np.asarray(config.view_elevation_range_deg, dtype=np.float64))
        azimuth_range = np.sort(np.asarray(config.view_azimuth_range_deg, dtype=np.float64))
        if distance_range.shape != (2,) or distance_range[0] <= 0.0:
            raise ValueError("close-up view distance range must contain two positive values")
        if elevation_range.shape != (2,) or elevation_range[0] <= 0.0 or elevation_range[1] >= 90.0:
            raise ValueError("close-up elevation range must lie strictly between 0 and 90 degrees")
        if azimuth_range.shape != (2,):
            raise ValueError("close-up azimuth range must contain two values")

        def full_range_signal(low: float, high: float) -> Array:
            signal = _bounded_smooth_signal(rng, frame_count, 1.0, knots)
            signal = _hann_smooth(
                signal[:, None], max(3, int(config.smoothing_window))
            )[:, 0]
            span = float(np.ptp(signal))
            if span < 1.0e-9:
                return np.full(frame_count, 0.5 * (low + high), dtype=np.float64)
            return low + (signal - float(np.min(signal))) * ((high - low) / span)

        # Keep the operation point on the tabletop while independently varying
        # viewing distance, elevation and azimuth. Every trajectory reaches the
        # requested near/far and low/high endpoints, making the framing change
        # visible for every random seed.
        target_offsets[:, 2] = 0.0
        target_offsets = _hann_smooth(
            target_offsets, max(3, int(config.smoothing_window))
        )
        target_offsets = np.clip(target_offsets, -target_bounds, target_bounds)
        targets = base_target + target_offsets
        if config.view_target_z is not None:
            targets[:, 2] = float(config.view_target_z)
        distances = full_range_signal(float(distance_range[0]), float(distance_range[1]))
        elevation_degrees = full_range_signal(
            float(elevation_range[0]), float(elevation_range[1])
        )

        # Near views and overhead views are each useful, but their combination
        # lets one tabletop object dominate almost the entire image.  Blend in
        # a distance floor before reaching the overhead threshold.  A smooth
        # maximum avoids introducing another hard turn where the two curves
        # meet; only the unsafe high-elevation/near-distance combination is
        # removed.
        overhead_threshold = float(config.overhead_elevation_threshold_deg)
        overhead_transition = max(1.0e-6, float(config.overhead_transition_deg))
        overhead_min_distance = max(
            float(distance_range[0]), float(config.overhead_min_distance)
        )
        planning_overhead_min_distance = max(
            float(distance_range[1]), overhead_min_distance + 0.10
        )
        overhead_blend = _smoothstep(
            (elevation_degrees - (overhead_threshold - overhead_transition))
            / overhead_transition
        )
        overhead_floor = (
            float(distance_range[0])
            + overhead_blend
            * (planning_overhead_min_distance - float(distance_range[0]))
        )
        requested_distances = distances.copy()
        smooth_max_epsilon = 5.0e-3
        distances = 0.5 * (
            distances
            + overhead_floor
            + np.sqrt((distances - overhead_floor) ** 2 + smooth_max_epsilon**2)
        )
        overhead_adjusted_frames = int(np.count_nonzero(
            requested_distances < overhead_floor
        ))
        elevations = np.deg2rad(elevation_degrees)
        base_azimuth = math.atan2(base_position[1] - base_target[1], base_position[0] - base_target[0])
        azimuths = base_azimuth + np.deg2rad(
            full_range_signal(float(azimuth_range[0]), float(azimuth_range[1]))
        )
        horizontal = distances * np.cos(elevations)
        positions = targets + np.column_stack(
            (horizontal * np.cos(azimuths), horizontal * np.sin(azimuths), distances * np.sin(elevations))
        )
        positions = _limit_vector_track(
            positions,
            config.motion_limits.max_translation_step,
            config.motion_limits.max_translation_accel,
        )

        # A close operation view can put the lens almost on top of the moving
        # gripper/forearm, allowing it to fill most of the image.  Keep the
        # sampled azimuth, elevation, target and intrinsics unchanged; only
        # retreat along the same target-to-camera ray when the camera center is
        # inside a moving exclusion sphere around the end effector.
        arm_clearance = max(0.0, float(config.min_action_camera_distance))
        clearance_adjusted_frames = 0
        if action_anchor is not None and arm_clearance > 0.0:
            if action is None:
                action = _as_track(action_anchor, frame_count, "action_anchor")
            requested_positions = positions.copy()
            # Plan against a slightly larger sphere. Interpolating/limiting a
            # curved path with straight chords can otherwise shave a fraction
            # of a millimeter off the requested physical clearance.
            planning_clearance = arm_clearance + 0.02
            for _ in range(12):
                camera_to_action = np.linalg.norm(positions - action, axis=1)
                unsafe = camera_to_action < planning_clearance
                if not np.any(unsafe):
                    break
                view_rays = positions - targets
                view_rays /= np.maximum(
                    np.linalg.norm(view_rays, axis=1, keepdims=True), 1.0e-9
                )
                target_to_action = targets - action
                linear = np.sum(target_to_action * view_rays, axis=1)
                constant = (
                    np.sum(target_to_action * target_to_action, axis=1)
                    - planning_clearance**2
                )
                farther_root = -linear + np.sqrt(np.maximum(linear * linear - constant, 0.0))
                current_distance = np.linalg.norm(positions - targets, axis=1)
                # One centimeter of geometric margin prevents the subsequent
                # acceleration limiter from landing numerically just inside
                # the exclusion sphere.
                safe_distance = np.maximum(current_distance, farther_root + 1.0e-2)
                positions[unsafe] = (
                    targets[unsafe]
                    + safe_distance[unsafe, None] * view_rays[unsafe]
                )
                positions = _limit_vector_track(
                    positions,
                    config.motion_limits.max_translation_step,
                    config.motion_limits.max_translation_accel,
                )
            clearance_adjusted_frames = int(np.count_nonzero(
                np.linalg.norm(requested_positions - action, axis=1) < planning_clearance
            ))
            achieved_clearance = float(np.min(np.linalg.norm(positions - action, axis=1)))
            if achieved_clearance < arm_clearance - 1.0e-4:
                raise RuntimeError(
                    "unable to satisfy camera03 arm clearance while preserving smooth motion: "
                    f"required={arm_clearance:.4f}m achieved={achieved_clearance:.4f}m"
                )
        else:
            achieved_clearance = None
        for _ in range(12):
            actual_view = positions - targets
            actual_distance = np.linalg.norm(actual_view, axis=1)
            actual_elevation = np.degrees(np.arctan2(
                actual_view[:, 2], np.linalg.norm(actual_view[:, :2], axis=1)
            ))
            overhead_buffer_unsafe = (
                (actual_elevation >= overhead_threshold)
                & (actual_distance < overhead_min_distance + 0.06)
            )
            if not np.any(overhead_buffer_unsafe):
                break
            unit_view = actual_view / np.maximum(actual_distance[:, None], 1.0e-9)
            positions[overhead_buffer_unsafe] = (
                targets[overhead_buffer_unsafe]
                + planning_overhead_min_distance
                * unit_view[overhead_buffer_unsafe]
            )
            positions = _limit_vector_track(
                positions,
                config.motion_limits.max_translation_step,
                config.motion_limits.max_translation_accel,
            )
        actual_view = positions - targets
        actual_distance = np.linalg.norm(actual_view, axis=1)
        actual_elevation = np.degrees(np.arctan2(
            actual_view[:, 2], np.linalg.norm(actual_view[:, :2], axis=1)
        ))
        forbidden_overhead_near = (
            (actual_elevation >= overhead_threshold)
            & (actual_distance < overhead_min_distance - 1.0e-4)
        )
        if np.any(forbidden_overhead_near):
            raise RuntimeError(
                "camera03 entered forbidden overhead/near region: "
                f"frames={int(np.count_nonzero(forbidden_overhead_near))} "
                f"threshold={overhead_threshold:.1f}deg "
                f"minimum_distance={overhead_min_distance:.3f}m"
            )
        position_bounds = np.maximum(np.abs(positions - base_position).max(axis=0), position_bounds)
    else:
        # Legacy hard-bounded Cartesian motion remains available to direct
        # planner callers; production camera03 uses the spherical view ranges.
        positions = base_position + position_offsets
        positions = _limit_vector_track(
            positions,
            config.motion_limits.max_translation_step,
            config.motion_limits.max_translation_accel,
        )
        positions = np.clip(positions, base_position - position_bounds, base_position + position_bounds)
        target_offsets[:, 2] = -pitch_coupling * (positions[:, 2] - base_position[2])
        target_offsets = np.clip(target_offsets, -target_bounds, target_bounds)
        targets = base_target + target_offsets
        arm_clearance = 0.0
        clearance_adjusted_frames = 0
        achieved_clearance = None
        overhead_threshold = None
        overhead_transition = None
        overhead_min_distance = None
        overhead_adjusted_frames = 0

    targets = _limit_vector_track(
        targets,
        config.motion_limits.max_focus_step,
        config.motion_limits.max_focus_accel,
    )
    # The spherical close-up mode intentionally replaces the legacy base target
    # height (which may sit above the work surface).  Re-applying the Cartesian
    # jitter box here would silently clamp that target back into empty space.
    if not spherical_view:
        targets = np.clip(targets, base_target - target_bounds, base_target + target_bounds)
    else:
        # The causal safety/acceleration projections above can leave small
        # velocity-direction corners even though every individual step passes
        # the motion gate. A zero-phase Hann pass removes those visible turns
        # without adding temporal lag. Re-check both safety contracts after
        # smoothing rather than assuming their margins were sufficient.
        final_smoothing_window = max(3, int(config.smoothing_window))
        positions = _hann_smooth(positions, final_smoothing_window)
        targets = _hann_smooth(targets, final_smoothing_window)
        final_view = positions - targets
        final_distance = np.linalg.norm(final_view, axis=1)
        final_elevation = np.degrees(np.arctan2(
            final_view[:, 2], np.linalg.norm(final_view[:, :2], axis=1)
        ))
        final_overhead_near = (
            (final_elevation >= float(overhead_threshold))
            & (final_distance < float(overhead_min_distance) - 1.0e-4)
        )
        if np.any(final_overhead_near):
            raise RuntimeError(
                "camera03 smoothing entered forbidden overhead/near region: "
                f"frames={int(np.count_nonzero(final_overhead_near))}"
            )
        if action is not None and arm_clearance > 0.0:
            achieved_clearance = float(np.min(np.linalg.norm(positions - action, axis=1)))
            if achieved_clearance < arm_clearance - 1.0e-4:
                raise RuntimeError(
                    "camera03 smoothing violated arm clearance: "
                    f"required={arm_clearance:.4f}m achieved={achieved_clearance:.4f}m"
                )
    undamped_positions = positions.copy()
    undamped_targets = targets.copy()
    damping = 1.0
    poses = camera_to_world_from_look_at(positions, targets)
    trajectory = CameraTrajectory(
        center_camera_to_world=poses,
        actual_look_at_points=targets,
        composition_offsets=np.zeros((frame_count, 2), dtype=np.float64),
        strategy_id_per_frame=np.zeros(frame_count, dtype=np.int32),
        strategy_names=["bounded_closeup_random"],
        segments=[{"start": 0, "end": frame_count, "strategy": "bounded_closeup_random"}],
        composition_profile="closeup_operation_centered",
        trajectory_seed=int(seed),
        strategy_seeds=[int(seed)],
        strategy_metadata=[{
            "position_jitter_m": position_bounds.tolist(),
            "target_jitter_m": target_bounds.tolist(),
            "action_follow_gain": float(config.action_follow_gain),
            "action_follow_applied": bool(follow_applied),
            "vertical_pitch_coupling": pitch_coupling,
            "spherical_view": bool(spherical_view),
            "view_distance_range_m": None if config.view_distance_range is None else list(config.view_distance_range),
            "view_elevation_range_deg": None if config.view_elevation_range_deg is None else list(config.view_elevation_range_deg),
            "view_azimuth_range_deg": None if config.view_azimuth_range_deg is None else list(config.view_azimuth_range_deg),
            "view_target_z_m": config.view_target_z,
            "min_action_camera_distance_m": arm_clearance,
            "action_clearance_adjusted_frames": clearance_adjusted_frames,
            "achieved_action_camera_distance_m": achieved_clearance,
            "smoothing_window_frames": int(config.smoothing_window),
            "overhead_elevation_threshold_deg": overhead_threshold,
            "overhead_transition_deg": overhead_transition,
            "overhead_min_distance_m": overhead_min_distance,
            "overhead_distance_adjusted_frames": overhead_adjusted_frames,
            "knot_count": knots,
        }],
    )
    ok, reason = motion_gate(trajectory, config.motion_limits)
    # Position and target motion can each be within their own limits while the
    # combined look direction is slightly too fast. Uniform damping preserves
    # the exact base framing and hard spatial bounds while guaranteeing a
    # conservative angular trajectory.
    for _ in range(8):
        if ok:
            break
        damping *= 0.75
        positions = base_position + (undamped_positions - base_position) * damping
        targets = base_target + (undamped_targets - base_target) * damping
        trajectory.center_camera_to_world = camera_to_world_from_look_at(positions, targets)
        trajectory.actual_look_at_points = targets
        ok, reason = motion_gate(trajectory, config.motion_limits)
    if not ok:
        raise RuntimeError(
            f"bounded close-up trajectory violates {reason}: "
            f"{trajectory_motion_metrics(trajectory)}"
        )
    trajectory.diagnostics = {
        "hard_position_bounds_m": position_bounds.tolist(),
        "hard_target_bounds_m": target_bounds.tolist(),
        "motion_metrics": trajectory_motion_metrics(trajectory),
        "motion_gate": reason,
        "motion_damping": damping,
        "min_action_camera_distance_m": arm_clearance,
        "action_clearance_adjusted_frames": clearance_adjusted_frames,
        "achieved_action_camera_distance_m": achieved_clearance,
        "smoothing_window_frames": int(config.smoothing_window),
        "overhead_elevation_threshold_deg": overhead_threshold,
        "overhead_transition_deg": overhead_transition,
        "overhead_min_distance_m": overhead_min_distance,
        "overhead_distance_adjusted_frames": overhead_adjusted_frames,
    }
    return trajectory


def _smoothstep(value: Array) -> Array:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _limit_vector_track(desired: Array, max_step: float, max_accel: float) -> Array:
    desired = np.asarray(desired, dtype=np.float64)
    if len(desired) < 2:
        return desired.copy()
    limited = np.empty_like(desired)
    limited[0] = desired[0]
    previous_delta = np.zeros(desired.shape[1], dtype=np.float64)
    for index in range(1, len(desired)):
        delta = desired[index] - limited[index - 1]
        accel = delta - previous_delta
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > max_accel:
            delta = previous_delta + accel * (max_accel / accel_norm)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step:
            delta *= max_step / delta_norm
        limited[index] = limited[index - 1] + delta
        previous_delta = delta
    return limited


def _slerp_direction(first: Array, second: Array, fraction: float) -> Array:
    first = _normalize(first)
    second = _normalize(second)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-8:
        return second
    sine = math.sin(angle)
    return _normalize(
        math.sin((1.0 - fraction) * angle) / sine * first
        + math.sin(fraction * angle) / sine * second
    )


def enforce_motion_limits(
    positions: Array,
    targets: Array,
    limits: MotionLimits,
) -> tuple[Array, Array, dict]:
    positions = _limit_vector_track(positions, limits.max_translation_step, limits.max_translation_accel)
    targets = _limit_vector_track(targets, limits.max_focus_step, limits.max_focus_accel)
    desired_forward = np.stack([_normalize(t - p, np.array([-1.0, 0.0, -0.2])) for p, t in zip(positions, targets)])
    limited_forward = np.empty_like(desired_forward)
    limited_forward[0] = desired_forward[0]
    previous_step = 0.0
    max_step = math.radians(limits.max_rotation_step_deg)
    max_accel = math.radians(limits.max_rotation_accel_deg)
    angular_steps = []
    for index in range(1, len(desired_forward)):
        angle = math.acos(float(np.clip(np.dot(limited_forward[index - 1], desired_forward[index]), -1.0, 1.0)))
        allowed = min(max_step, previous_step + max_accel)
        used = min(angle, allowed)
        fraction = 1.0 if angle < 1.0e-9 else used / angle
        limited_forward[index] = _slerp_direction(limited_forward[index - 1], desired_forward[index], fraction)
        angular_steps.append(used)
        previous_step = used
    focus_distances = np.linalg.norm(targets - positions, axis=1)
    targets = positions + limited_forward * focus_distances[:, None]
    return positions, targets, {
        "limited": True,
        "max_translation_step": float(limits.max_translation_step),
        "max_translation_accel": float(limits.max_translation_accel),
        "max_rotation_step_deg": float(limits.max_rotation_step_deg),
        "max_rotation_accel_deg": float(limits.max_rotation_accel_deg),
        "observed_limiter_rotation_step_deg": float(np.degrees(max(angular_steps, default=0.0))),
    }


def trajectory_motion_metrics(trajectory: CameraTrajectory) -> dict[str, float]:
    poses = trajectory.center_camera_to_world
    positions = poses[:, :3, 3]
    translation_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    translation_changes = np.abs(np.diff(translation_steps))
    rotation_steps = []
    for first, second in zip(poses[:-1, :3, :3], poses[1:, :3, :3]):
        relative = first.T @ second
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        rotation_steps.append(math.degrees(math.acos(cosine)))
    rotation_steps = np.asarray(rotation_steps, dtype=np.float64)
    return {
        "path_length": float(translation_steps.sum()),
        "position_span": float(np.linalg.norm(np.ptp(positions, axis=0))),
        "max_translation_step": float(translation_steps.max(initial=0.0)),
        "max_translation_step_change": float(translation_changes.max(initial=0.0)),
        "max_rotation_step_deg": float(rotation_steps.max(initial=0.0)),
        "max_rotation_step_change_deg": float(np.abs(np.diff(rotation_steps)).max(initial=0.0)),
    }


def motion_gate(trajectory: CameraTrajectory, limits: MotionLimits, tolerance: float = 1.05) -> tuple[bool, str]:
    metrics = trajectory_motion_metrics(trajectory)
    if metrics["max_translation_step"] > limits.max_translation_step * tolerance:
        return False, "translation_step"
    if metrics["max_translation_step_change"] > limits.max_translation_accel * tolerance:
        return False, "translation_accel"
    if metrics["max_rotation_step_deg"] > limits.max_rotation_step_deg * tolerance:
        return False, "rotation_step"
    if metrics["max_rotation_step_change_deg"] > limits.max_rotation_accel_deg * 2.0 * tolerance:
        return False, "rotation_accel"
    return True, "ok"


_COMPOSITION_PROFILES = {
    "centered": (0.28, 0.20, 0.20, 0.22),
    "balanced": (0.48, 0.32, 0.55, 0.38),
    "off_center": (0.65, 0.42, 0.80, 0.55),
}


def _composition_track(rng: np.random.Generator, frame_count: int, profile: str, knot_count: int) -> Array:
    max_x, max_y, off_center_probability, target_offset = _COMPOSITION_PROFILES[profile]
    x_center = 0.0
    if rng.random() < off_center_probability:
        x_center = rng.choice((-1.0, 1.0)) * target_offset
    x = x_center + smooth_random_signal(rng, frame_count, max_x * 0.22, knot_count)
    y = smooth_random_signal(rng, frame_count, max_y * 0.25, knot_count)
    return np.column_stack((np.clip(x, -max_x, max_x), np.clip(y, -max_y, max_y)))


def _apply_composition_offsets(
    positions: Array,
    semantic_focus: Array,
    offsets: Array,
    vertical_fov_deg: float,
    aspect_ratio: float,
) -> Array:
    actual = np.empty_like(semantic_focus)
    half_v = math.tan(math.radians(vertical_fov_deg) * 0.5)
    half_h = half_v * float(aspect_ratio)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for index, (position, focus, offset) in enumerate(zip(positions, semantic_focus, offsets)):
        forward = _normalize(focus - position, np.array([-1.0, 0.0, -0.2]))
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1.0e-7:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = _normalize(right)
        up = _normalize(np.cross(right, forward))
        distance = float(np.linalg.norm(focus - position))
        actual[index] = (
            focus
            - right * float(offset[0]) * distance * half_h
            + up * float(offset[1]) * distance * half_v
        )
    return actual


def _mixed_focus(context: TrajectoryContext, rng: np.random.Generator, random_walk: bool) -> Array:
    body_target = context.body_anchor.copy()
    body_target[:, 2] += 0.55
    regular = 0.55 * context.subject_focus + 0.45 * body_target
    regular[:, 2] += 0.10
    random_context = 0.58 * context.subject_focus + 0.42 * body_target
    random_context[:, 2] += 0.08
    if random_walk:
        focus = 0.50 * context.subject_focus + 0.30 * random_context + 0.20 * context.action_anchor
        sigma = np.array([0.055, 0.055, 0.030])
    else:
        choices = (context.subject_focus, context.action_anchor, body_target, regular)
        focus = choices[int(rng.choice(4, p=(0.40, 0.25, 0.15, 0.20)))].copy()
        if context.frame_count >= 12 and rng.random() < 0.45:
            second = choices[int(rng.choice(4, p=(0.40, 0.25, 0.15, 0.20)))]
            blend = _smoothstep(np.linspace(0.0, 1.0, context.frame_count))[:, None]
            focus = focus * (1.0 - blend) + second * blend
        sigma = np.array([0.045, 0.045, 0.025])
    for axis in range(3):
        focus[:, axis] += smooth_random_signal(rng, context.frame_count, sigma[axis], max(4, context.frame_count // 14))
    if random_walk and context.frame_count >= 12 and rng.random() < 0.65:
        progress = np.arange(context.frame_count, dtype=np.float64)
        for _ in range(int(rng.integers(1, 3))):
            center = float(rng.uniform(0.20, 0.80) * (context.frame_count - 1))
            half_width = float(rng.uniform(context.frame_count / 14.0, context.frame_count / 6.0))
            peak = float(rng.uniform(0.20, 0.45))
            weight = peak * np.exp(-0.5 * ((progress - center) / max(half_width, 1.0)) ** 2)
            weight = _limit_vector_track(weight[:, None], 0.10, 0.05)[:, 0]
            focus = focus * (1.0 - weight[:, None]) + context.action_anchor * weight[:, None]
    return _hann_smooth(focus, 9)


def _reference_azimuth(context: TrajectoryContext) -> Array:
    if context.reference_camera is not None:
        delta = context.reference_camera[:, :2, 3] - context.subject_focus[:, :2]
    else:
        delta = context.body_anchor[:, :2] - context.subject_focus[:, :2]
    degenerate = np.linalg.norm(delta, axis=1) < 1.0e-6
    delta[degenerate] = np.array([1.0, 0.0])
    azimuth = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    return _hann_smooth(azimuth[:, None], 7)[:, 0]


def _positions_from_cylindrical(focus: Array, azimuth: Array, radius: Array, height: Array) -> Array:
    return focus + np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))


def _generate_strategy(
    context: TrajectoryContext,
    strategy: str,
    rng: np.random.Generator,
    config: PlannerConfig,
) -> tuple[Array, Array, dict]:
    frame_count = context.frame_count
    progress = np.linspace(0.0, 1.0, frame_count)
    reference = _reference_azimuth(context)
    radius_lo, radius_hi = config.radius_range
    elevation_lo, elevation_hi = np.radians(config.elevation_range_deg)
    metadata: dict = {"strategy": strategy}

    if strategy == "agent_follow" and context.reference_camera is not None:
        focus = _mixed_focus(context, rng, False)
        reference_pose = context.reference_camera
        positions = reference_pose[:, :3, 3].copy()
        local_noise = np.column_stack(
            [smooth_random_signal(rng, frame_count, sigma, config.knot_count) for sigma in (0.08, 0.05, 0.05)]
        )
        positions += np.einsum("nij,nj->ni", reference_pose[:, :3, :3], local_noise)
        return positions, focus, metadata
    if strategy == "agent_follow":
        strategy = "focus_follow"
        metadata["fallback"] = "focus_follow"

    random_walk = strategy == "random_walk"
    focus = _mixed_focus(context, rng, random_walk)
    az_min, az_max = np.radians(config.azimuth_range_deg)
    base_azimuth = float(rng.uniform(az_min, az_max))

    if strategy == "orbit":
        sweep = math.radians(float(rng.uniform(120.0, 300.0) if rng.random() < 0.8 else rng.uniform(300.0, 360.0)))
        sweep *= rng.choice((-1.0, 1.0))
        azimuth = reference + base_azimuth + sweep * progress
        azimuth += smooth_random_signal(rng, frame_count, math.radians(8.0), config.knot_count)
        radius = rng.uniform(radius_lo, radius_hi) + smooth_random_signal(rng, frame_count, 0.10, config.knot_count)
        elevation = rng.uniform(elevation_lo, elevation_hi) + smooth_random_signal(rng, frame_count, math.radians(4.0), config.knot_count)
        height = radius * np.tan(elevation)
        metadata["sweep_deg"] = math.degrees(sweep)
    elif strategy == "focus_follow":
        knots = min(max(4, math.ceil(frame_count / 10)), 7, frame_count)
        knot_t = np.linspace(0.0, frame_count - 1, knots)
        knot_azimuth = np.unwrap(rng.uniform(az_min, az_max, knots))
        azimuth = reference + np.interp(np.arange(frame_count), knot_t, knot_azimuth)
        azimuth += smooth_random_signal(rng, frame_count, math.radians(8.0), knots)
        radius = rng.uniform(radius_lo, radius_hi) + smooth_random_signal(rng, frame_count, 0.10, knots)
        elevation = rng.uniform(elevation_lo, elevation_hi) + smooth_random_signal(rng, frame_count, math.radians(4.0), knots)
        height = radius * np.tan(elevation)
    else:
        mode = str(rng.choice(("arc_jitter", "lateral_truck", "dolly", "spiral", "focus_shift"), p=(0.30, 0.25, 0.20, 0.15, 0.10)))
        metadata["random_walk_mode"] = mode
        radius_start = float(rng.uniform(radius_lo, radius_hi))
        elevation_start = float(rng.uniform(elevation_lo, elevation_hi))
        azimuth = np.full(frame_count, base_azimuth)
        radius = np.full(frame_count, radius_start)
        height = np.full(frame_count, radius_start * math.tan(elevation_start))
        lateral = np.zeros(frame_count)
        if mode == "arc_jitter":
            sweep = math.radians(float(rng.uniform(70.0, 150.0))) * rng.choice((-1.0, 1.0))
            azimuth += sweep * progress + smooth_random_signal(rng, frame_count, math.radians(7.0), config.knot_count)
            radius += smooth_random_signal(rng, frame_count, 0.12, config.knot_count)
            height += smooth_random_signal(rng, frame_count, 0.10, config.knot_count)
        elif mode == "lateral_truck":
            azimuth += smooth_random_signal(rng, frame_count, math.radians(9.0), config.knot_count)
            lateral = np.linspace(-1.0, 1.0, frame_count) * float(rng.uniform(0.28, 0.48))
            lateral += smooth_random_signal(rng, frame_count, 0.035, config.knot_count)
        elif mode == "dolly":
            azimuth += smooth_random_signal(rng, frame_count, math.radians(7.0), config.knot_count)
            change = float(rng.uniform(0.35, 0.75)) * rng.choice((-1.0, 1.0))
            radius = np.clip(radius_start + change * _smoothstep(progress), max(0.35, radius_lo), radius_hi + 0.55)
        elif mode == "spiral":
            sweep = math.radians(float(rng.uniform(80.0, 165.0))) * rng.choice((-1.0, 1.0))
            azimuth += sweep * progress
            end_radius = float(rng.uniform(max(radius_start, radius_lo + 0.25), radius_hi + 0.45))
            radius = radius_start + (end_radius - radius_start) * _smoothstep(progress)
            if rng.random() < 0.5:
                radius = radius[::-1].copy()
            height_end = height[0] + float(rng.uniform(0.25, 0.60))
            height = height[0] + (height_end - height[0]) * _smoothstep(progress)
            if rng.random() < 0.5:
                height = height[::-1].copy()
        else:
            azimuth += math.radians(float(rng.uniform(-75.0, 75.0))) * _smoothstep(progress)
            start = context.subject_focus if rng.random() < 0.5 else context.action_anchor
            end = context.action_anchor if rng.random() < 0.6 else context.subject_focus
            blend = _smoothstep(progress)[:, None]
            focus = start * (1.0 - blend) + end * blend
        positions = _positions_from_cylindrical(focus, azimuth, radius, height)
        tangent = np.column_stack((-np.sin(azimuth), np.cos(azimuth), np.zeros(frame_count)))
        positions += tangent * lateral[:, None]
        return positions, focus, metadata

    return _positions_from_cylindrical(focus, azimuth, np.clip(radius, 0.35, None), height), focus, metadata


def _sample_strategy_names(
    rng: np.random.Generator,
    strategies: tuple[str, ...],
    weights: Array,
    frame_count: int,
) -> list[str]:
    if frame_count < 6:
        return ["focus_follow"]
    if rng.random() < 0.5:
        return [str(rng.choice(strategies, p=weights))]
    count = 2 if frame_count < 15 or rng.random() < 0.65 else 3
    count = min(count, len(strategies))
    selected: list[str] = []
    available = list(strategies)
    available_weights = weights.copy()
    for _ in range(count):
        probabilities = available_weights / available_weights.sum()
        index = int(rng.choice(len(available), p=probabilities))
        selected.append(available.pop(index))
        available_weights = np.delete(available_weights, index)
    return selected


def _segment_boundaries(frame_count: int, segment_count: int, rng: np.random.Generator) -> list[int]:
    if segment_count == 1:
        return [0, frame_count]
    minimum = max(1, int(math.ceil(frame_count * 0.20)))
    maximum = max(minimum, int(math.floor(frame_count * 0.60)))
    for _ in range(64):
        cuts = sorted(rng.choice(np.arange(1, frame_count), segment_count - 1, replace=False).tolist())
        boundaries = [0, *cuts, frame_count]
        lengths = np.diff(boundaries)
        if np.all(lengths >= minimum) and np.all(lengths <= maximum):
            return boundaries
    # Deterministic fallback for short or awkward episode lengths.
    return np.rint(np.linspace(0, frame_count, segment_count + 1)).astype(int).tolist()


def _blend_strategy_tracks(
    position_tracks: list[Array],
    focus_tracks: list[Array],
    boundaries: list[int],
) -> tuple[Array, Array, Array, list[dict]]:
    frame_count = len(position_tracks[0])
    strategy_ids = np.empty(frame_count, dtype=np.int32)
    positions = np.empty_like(position_tracks[0])
    focus = np.empty_like(focus_tracks[0])
    segments: list[dict] = []
    for strategy_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        positions[start:end] = position_tracks[strategy_id][start:end]
        focus[start:end] = focus_tracks[strategy_id][start:end]
        strategy_ids[start:end] = strategy_id
        segments.append({"start": start, "end": end})

    aligned_positions = [track.copy() for track in position_tracks]
    aligned_focus = [track.copy() for track in focus_tracks]
    for boundary_index, boundary in enumerate(boundaries[1:-1]):
        previous = aligned_positions[boundary_index]
        following = aligned_positions[boundary_index + 1]
        previous_focus = aligned_focus[boundary_index]
        following_focus = aligned_focus[boundary_index + 1]
        position_gap = previous[boundary - 1] - following[boundary]
        focus_gap = previous_focus[boundary - 1] - following_focus[boundary]
        decay_end = boundaries[boundary_index + 2]
        decay = (1.0 - _smoothstep(np.linspace(0.0, 1.0, decay_end - boundary)))[:, None]
        following[boundary:decay_end] += position_gap * decay
        following_focus[boundary:decay_end] += focus_gap * decay
        positions[boundary:decay_end] = following[boundary:decay_end]
        focus[boundary:decay_end] = following_focus[boundary:decay_end]

        half_width = min(max(3, frame_count // 16), boundary, frame_count - boundary)
        start = boundary - half_width
        end = boundary + half_width
        blend = _smoothstep(np.linspace(0.0, 1.0, end - start))[:, None]
        positions[start:end] = (
            aligned_positions[boundary_index][start:end] * (1.0 - blend)
            + aligned_positions[boundary_index + 1][start:end] * blend
        )
        focus[start:end] = (
            aligned_focus[boundary_index][start:end] * (1.0 - blend)
            + aligned_focus[boundary_index + 1][start:end] * blend
        )
        segments[boundary_index]["transition_out"] = [start, end]
        segments[boundary_index + 1]["transition_in"] = [start, end]
        segments[boundary_index + 1]["boundary_alignment_m"] = float(np.linalg.norm(position_gap))
    return positions, focus, strategy_ids, segments


def _diversity_score(trajectory: CameraTrajectory) -> float:
    positions = trajectory.positions
    targets = trajectory.actual_look_at_points
    radius = np.linalg.norm(positions[:, :2] - targets[:, :2], axis=1)
    azimuth = np.unwrap(np.arctan2(positions[:, 1] - targets[:, 1], positions[:, 0] - targets[:, 0]))
    position = min(float(np.linalg.norm(np.ptp(positions, axis=0))) / 0.90, 1.0)
    radius_score = min(float(np.ptp(radius)) / 0.55, 1.0)
    height = min(float(np.ptp(positions[:, 2])) / 0.40, 1.0)
    compose = min(float(np.ptp(trajectory.composition_offsets[:, 0])) / 0.35, 1.0)
    azimuth_coverage = min(float(np.ptp(azimuth)) / math.radians(120.0), 1.0)
    return 0.30 * position + 0.18 * max(radius_score, height) + 0.17 * compose + 0.20 * azimuth_coverage + 0.15 * min(trajectory_motion_metrics(trajectory)["path_length"] / 1.5, 1.0)


def _smoothness_penalty(metrics: dict[str, float]) -> float:
    return (
        metrics["max_translation_step"]
        + metrics["max_rotation_step_deg"] / 90.0
        + metrics["max_translation_step_change"]
        + metrics["max_rotation_step_change_deg"] / 90.0
    )


def _pose_hash(poses: Array) -> bytes:
    return np.round(poses, 5).tobytes()


def solve_episode(
    context: TrajectoryContext,
    episode_seed: int,
    *,
    config: PlannerConfig = PlannerConfig(),
    profile: str = "balanced",
    projector: TrajectoryProjector | None = None,
    validator: TrajectoryValidator | None = None,
) -> CameraTrajectory:
    """Generate a candidate pool and return its best valid complete trajectory."""
    if profile not in _COMPOSITION_PROFILES:
        raise ValueError(f"unknown composition profile: {profile}")
    frame_count = context.frame_count
    if frame_count < 2:
        raise ValueError("at least two frames are required")
    candidate_count = max(1, int(config.candidate_count))
    strategies, weights = zip(*config.strategy_weights)
    weights_array = np.asarray(weights, dtype=np.float64)
    weights_array /= weights_array.sum()
    accepted: list[tuple[float, CameraTrajectory]] = []
    seen = set()
    rejected: dict[str, int] = {}

    for attempt in range(candidate_count):
        seed = stable_seed(episode_seed, "camera_candidate", attempt, profile)
        rng = np.random.default_rng(seed)
        strategy_names = _sample_strategy_names(rng, strategies, weights_array, frame_count)
        strategy_seeds = [
            stable_seed(seed, "strategy", profile, index, strategy)
            for index, strategy in enumerate(strategy_names)
        ]
        generated = [
            _generate_strategy(context, strategy, np.random.default_rng(strategy_seed), config)
            for strategy, strategy_seed in zip(strategy_names, strategy_seeds)
        ]
        boundaries = _segment_boundaries(frame_count, len(strategy_names), rng)
        positions, semantic_focus, strategy_ids, segments = _blend_strategy_tracks(
            [item[0] for item in generated],
            [item[1] for item in generated],
            boundaries,
        )
        if rng.random() < 0.50:
            perturbation = np.column_stack(
                [smooth_random_signal(rng, frame_count, sigma, config.knot_count) for sigma in (0.05, 0.05, 0.035)]
            )
            positions += perturbation
            semantic_focus += 0.35 * perturbation
        strategy_metadata = [item[2] for item in generated]
        for index, segment in enumerate(segments):
            segment.update(
                {
                    "strategy": strategy_names[index],
                    "seed": strategy_seeds[index],
                }
            )
        composition = _composition_track(rng, frame_count, profile, config.knot_count)
        targets = _apply_composition_offsets(
            positions,
            semantic_focus,
            composition,
            config.vertical_fov_deg,
            config.aspect_ratio,
        )
        projection_stats = {}
        if projector is not None:
            projected = projector(positions, targets)
            positions, targets = projected[:2]
            if len(projected) > 2:
                projection_stats = dict(projected[2])
        desired_positions = positions.copy()
        desired_targets = targets.copy()
        positions, targets, limiter_stats = enforce_motion_limits(positions, targets, config.motion_limits)
        poses = camera_to_world_from_look_at(positions, targets)
        trajectory = CameraTrajectory(
            center_camera_to_world=poses,
            actual_look_at_points=targets,
            composition_offsets=composition,
            strategy_id_per_frame=strategy_ids,
            strategy_names=strategy_names,
            segments=segments,
            composition_profile=profile,
            trajectory_seed=seed,
            strategy_seeds=strategy_seeds,
            strategy_metadata=strategy_metadata,
            clearance_projection_stats={**projection_stats, "motion_limiter": limiter_stats},
        )
        ok, reason = motion_gate(trajectory, config.motion_limits)
        if not ok and reason in {"rotation_step", "rotation_accel"}:
            repair_limits = replace(
                config.motion_limits,
                # A world-up look-at frame can rotate around its forward axis
                # faster than the forward vector itself near high elevations.
                max_rotation_step_deg=config.motion_limits.max_rotation_step_deg * 0.15,
                max_rotation_accel_deg=config.motion_limits.max_rotation_accel_deg * 0.15,
            )
            positions, targets, repair_stats = enforce_motion_limits(
                desired_positions,
                desired_targets,
                repair_limits,
            )
            poses = camera_to_world_from_look_at(positions, targets)
            trajectory.center_camera_to_world = poses
            trajectory.actual_look_at_points = targets
            trajectory.clearance_projection_stats.update(
                {
                    "motion_repaired": True,
                    "repair_rotation_step_deg": repair_limits.max_rotation_step_deg,
                    "repair_rotation_accel_deg": repair_limits.max_rotation_accel_deg,
                    "repair_limiter": repair_stats,
                }
            )
            ok, reason = motion_gate(trajectory, config.motion_limits)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        digest = _pose_hash(poses)
        if digest in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        seen.add(digest)
        external_score = 0.0
        if validator is not None:
            ok, external_score, reason = validator(trajectory)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
        metrics = trajectory_motion_metrics(trajectory)
        diversity = _diversity_score(trajectory)
        score = float(external_score + 1.5 * diversity - 2.0 * _smoothness_penalty(metrics))
        trajectory.diagnostics.update({
            "motion": metrics,
            "diversity_score": diversity,
            "selection_score": score,
            "attempt": attempt,
        })
        accepted.append((score, trajectory))

    if not accepted:
        raise RuntimeError(f"no valid camera trajectory; rejected={rejected}")
    trajectory_scores = [(trajectory, score) for score, trajectory in accepted]
    shortlist_limit = min(max(1, 8), len(trajectory_scores))
    shortlist = strategy_aware_shortlist(
        trajectory_scores,
        shortlist_limit,
        config.strategy_weights,
    )
    selected_set = greedy_quality_diversity_selection(
        shortlist,
        requested_count=1,
        strategy_weights=config.strategy_weights,
    )
    if not selected_set:
        raise RuntimeError(
            f"no diverse camera trajectory survived selection; rejected={rejected}"
        )
    selected = selected_set[0]
    selected.diagnostics.update(
        {
            "candidate_funnel": {
                "sampled": candidate_count,
                "motion_and_geometry_accepted": len(accepted),
                "shortlisted": len(shortlist),
                "final_selected": 1,
            },
            "candidate_count": candidate_count,
            "accepted_count": len(accepted),
            "rejected": rejected,
        }
    )
    return selected


@dataclass(slots=True)
class AABBObstacle:
    """Axis-aligned obstacle bounds, static or varying once per trajectory frame."""

    bounds_min: Array
    bounds_max: Array
    kind: str = "scene"
    name: str = "obstacle"

    def __post_init__(self) -> None:
        self.bounds_min = np.asarray(self.bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(self.bounds_max, dtype=np.float64)
        if self.bounds_min.shape != self.bounds_max.shape:
            raise ValueError("obstacle min/max shapes must match")
        if self.bounds_min.shape not in {(3,)} and not (
            self.bounds_min.ndim == 2 and self.bounds_min.shape[1] == 3
        ):
            raise ValueError("obstacle bounds must have shape (3,) or (N, 3)")
        if np.any(self.bounds_max < self.bounds_min):
            raise ValueError("obstacle bounds_max must be >= bounds_min")
        if self.kind not in {"robot", "scene"}:
            raise ValueError("obstacle kind must be 'robot' or 'scene'")

    def at(self, frame_index: int) -> tuple[Array, Array]:
        if self.bounds_min.ndim == 1:
            return self.bounds_min, self.bounds_max
        index = min(max(int(frame_index), 0), len(self.bounds_min) - 1)
        return self.bounds_min[index], self.bounds_max[index]


@dataclass(slots=True, frozen=True)
class GeometryGateConfig:
    baseline: float = 0.16
    vertical_fov_deg: float = 75.0
    aspect_ratio: float = 16.0 / 9.0
    frame_margin: float = 1.15
    min_forward_distance: float = 0.35
    closeup_horizontal_distance: float = 0.75
    closeup_min_forward_distance: float = 0.15
    scene_clearance: float = 0.06
    robot_clearance: float = 0.02
    projection_margin: float = 0.01
    projection_max_displacement: float = 0.12
    projection_iterations: int = 3
    max_bad_frame_ratio: float = 0.10
    max_consecutive_bad_frames: int = 1
    los_target_tolerance: float = 0.04


@dataclass(slots=True, frozen=True)
class GeometryGateResult:
    ok: bool
    reason: str
    bad_frame_ratio: float
    max_consecutive_bad_frames: int
    diagnostics: dict


def _point_aabb_delta(point: Array, bounds_min: Array, bounds_max: Array) -> tuple[float, Array]:
    """Return point-to-box distance and outward unit direction.

    For an interior point, distance is zero and the direction exits through the
    nearest face. For an exterior point, the direction follows the exact closest
    point vector.
    """
    point = np.asarray(point, dtype=np.float64)
    closest = np.clip(point, bounds_min, bounds_max)
    delta = point - closest
    distance = float(np.linalg.norm(delta))
    if distance > 1.0e-10:
        return distance, delta / distance
    face_distances = np.concatenate((point - bounds_min, bounds_max - point))
    face = int(np.argmin(face_distances))
    direction = np.zeros(3, dtype=np.float64)
    direction[face % 3] = -1.0 if face < 3 else 1.0
    return 0.0, direction


def _segment_aabb_hit_distance(origin: Array, target: Array, bounds_min: Array, bounds_max: Array) -> float | None:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-10:
        return None
    unit = direction / length
    t_min, t_max = 0.0, length
    for axis in range(3):
        if abs(unit[axis]) < 1.0e-12:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return None
            continue
        first = (bounds_min[axis] - origin[axis]) / unit[axis]
        second = (bounds_max[axis] - origin[axis]) / unit[axis]
        near, far = min(first, second), max(first, second)
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return None
    return t_min if 0.0 <= t_min <= length else None


def project_stereo_clearance(
    positions: Array,
    targets: Array,
    obstacles: list[AABBObstacle],
    config: GeometryGateConfig,
) -> tuple[Array, Array, dict]:
    """Smoothly project center poses so center and both stereo eyes have clearance."""
    positions = np.asarray(positions, dtype=np.float64).copy()
    targets = np.asarray(targets, dtype=np.float64).copy()
    original = positions.copy()
    iteration_stats = []
    for iteration in range(max(0, int(config.projection_iterations))):
        poses = camera_to_world_from_look_at(positions, targets)
        left, right = make_stereo_poses(poses, config.baseline)
        correction = np.zeros_like(positions)
        violations = 0
        for frame in range(len(positions)):
            eyes = (positions[frame], left[frame, :3, 3], right[frame, :3, 3])
            for obstacle in obstacles:
                bounds_min, bounds_max = obstacle.at(frame)
                clearance = config.robot_clearance if obstacle.kind == "robot" else config.scene_clearance
                required = clearance + config.projection_margin
                for eye in eyes:
                    distance, outward = _point_aabb_delta(eye, bounds_min, bounds_max)
                    if distance < required:
                        correction[frame] += outward * (required - distance)
                        violations += 1
        if violations == 0:
            iteration_stats.append({"iteration": iteration, "violations": 0})
            break
        correction = _hann_smooth(correction, 5)
        desired = positions + correction
        displacement = desired - original
        norms = np.linalg.norm(displacement, axis=1)
        over = norms > config.projection_max_displacement
        displacement[over] *= config.projection_max_displacement / norms[over, None]
        positions = original + displacement
        iteration_stats.append({
            "iteration": iteration,
            "violations": violations,
            "max_cumulative_displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
        })
    return positions, targets, {
        "enabled": bool(obstacles),
        "obstacle_count": len(obstacles),
        "iterations": iteration_stats,
        "max_displacement_m": float(np.max(np.linalg.norm(positions - original, axis=1), initial=0.0)),
    }


def _target_in_frame(pose: Array, target: Array, config: GeometryGateConfig) -> tuple[bool, float]:
    camera_point = pose[:3, :3].T @ (target - pose[:3, 3])
    depth = float(camera_point[2])
    horizontal = float(np.linalg.norm((target - pose[:3, 3])[:2]))
    minimum = (
        config.closeup_min_forward_distance
        if horizontal < config.closeup_horizontal_distance
        else config.min_forward_distance
    )
    if depth < minimum:
        return False, depth
    half_v = math.tan(math.radians(config.vertical_fov_deg) * 0.5) * config.frame_margin
    half_h = half_v * config.aspect_ratio
    return (
        abs(float(camera_point[0]) / depth) <= half_h
        and abs(float(camera_point[1]) / depth) <= half_v,
        depth,
    )


def _line_of_sight_clear(
    eye: Array,
    target: Array,
    frame: int,
    obstacles: list[AABBObstacle],
    tolerance: float,
) -> bool:
    target_distance = float(np.linalg.norm(target - eye))
    for obstacle in obstacles:
        bounds_min, bounds_max = obstacle.at(frame)
        hit = _segment_aabb_hit_distance(eye, target, bounds_min, bounds_max)
        if hit is None:
            continue
        if obstacle.kind == "robot" or hit >= target_distance - tolerance:
            continue
        return False
    return True


def geometry_gate(
    trajectory: CameraTrajectory,
    subject_targets: Array,
    obstacles: list[AABBObstacle],
    config: GeometryGateConfig,
) -> GeometryGateResult:
    """Validate frustum, LOS, and per-eye clearance over every trajectory frame."""
    target_array = np.asarray(subject_targets, dtype=np.float64)
    frame_count = len(trajectory.center_camera_to_world)
    if target_array.ndim == 2 and target_array.shape == (frame_count, 3):
        target_array = target_array[:, None, :]
    if target_array.ndim != 3 or target_array.shape[0] != frame_count or target_array.shape[2] != 3:
        raise ValueError("subject_targets must have shape (N, 3) or (N, K, 3)")

    left, right = make_stereo_poses(trajectory.center_camera_to_world, config.baseline)
    bad = np.zeros(frame_count, dtype=bool)
    reasons: dict[str, int] = {}
    min_clearance = {"robot": math.inf, "scene": math.inf}
    for frame in range(frame_count):
        eyes = (left[frame], right[frame])
        frame_reason = None
        for eye_pose in eyes:
            eye = eye_pose[:3, 3]
            for obstacle in obstacles:
                bounds_min, bounds_max = obstacle.at(frame)
                distance, _ = _point_aabb_delta(eye, bounds_min, bounds_max)
                min_clearance[obstacle.kind] = min(min_clearance[obstacle.kind], distance)
                required = config.robot_clearance if obstacle.kind == "robot" else config.scene_clearance
                if distance < required:
                    frame_reason = f"{obstacle.kind}_clearance"
                    break
            if frame_reason:
                break
            visible = False
            framed = False
            for target in target_array[frame]:
                target_framed, _ = _target_in_frame(eye_pose, target, config)
                framed = framed or target_framed
                if target_framed and _line_of_sight_clear(
                    eye, target, frame, obstacles, config.los_target_tolerance
                ):
                    visible = True
                    break
            if not framed:
                frame_reason = "framing"
                break
            if not visible:
                frame_reason = "line_of_sight"
                break
        if frame_reason:
            bad[frame] = True
            reasons[frame_reason] = reasons.get(frame_reason, 0) + 1

    runs = []
    current = 0
    for value in bad:
        current = current + 1 if value else 0
        runs.append(current)
    max_run = max(runs, default=0)
    ratio = float(np.mean(bad))
    hard_clearance_failure = any(key.endswith("_clearance") for key in reasons)
    ok = (
        not hard_clearance_failure
        and ratio <= config.max_bad_frame_ratio
        and max_run <= config.max_consecutive_bad_frames
    )
    if hard_clearance_failure:
        reason = next(key for key in reasons if key.endswith("_clearance"))
    elif ratio > config.max_bad_frame_ratio:
        reason = "bad_frame_ratio"
    elif max_run > config.max_consecutive_bad_frames:
        reason = "consecutive_bad_frames"
    else:
        reason = "ok"
    return GeometryGateResult(
        ok=ok,
        reason=reason,
        bad_frame_ratio=ratio,
        max_consecutive_bad_frames=max_run,
        diagnostics={
            "frame_count": frame_count,
            "bad_frame_count": int(np.sum(bad)),
            "reason_counts": reasons,
            "minimum_clearance_m": {
                key: (None if not np.isfinite(value) else float(value))
                for key, value in min_clearance.items()
            },
            "stereo_checked": True,
        },
    )


@dataclass(slots=True, frozen=True)
class VisualQualityConfig:
    max_weak_background_ratio: float = 0.70
    # Whole-frame gradients are diluted by large, valid low-texture surfaces
    # such as tabletops, walls, and cabinet panels. 0.07 keeps those views
    # while still rejecting genuinely textureless renders.
    min_texture_score: float = 0.07
    max_bad_frame_ratio: float = 0.10
    max_consecutive_bad_frames: int = 1


@dataclass(slots=True, frozen=True)
class VisualQualityResult:
    ok: bool
    score: float
    reason: str
    diagnostics: dict


def sparse_quality_frame_indices(trajectory: CameraTrajectory, uniform_count: int = 9) -> Array:
    """Uniform frames plus azimuth/radius/height view extremes."""
    positions = trajectory.positions
    targets = trajectory.actual_look_at_points
    frame_count = len(positions)
    uniform = np.rint(np.linspace(0, frame_count - 1, min(uniform_count, frame_count))).astype(int)
    delta = positions - targets
    azimuth = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    radius = np.linalg.norm(delta[:, :2], axis=1)
    extremes = np.asarray(
        [np.argmin(azimuth), np.argmax(azimuth), np.argmin(radius), np.argmin(positions[:, 2])],
        dtype=int,
    )
    return np.unique(np.concatenate((uniform, extremes)))


def stereo_frame_quality(
    left_rgb: Array,
    right_rgb: Array,
    left_depth: Array,
    right_depth: Array,
    *,
    left_subject_mask: Array | None = None,
    right_subject_mask: Array | None = None,
) -> dict[str, float | None | bool]:
    """Compute conservative stereo image/depth metrics for one synchronized frame."""
    eye_metrics = []
    for rgb, depth, subject in (
        (left_rgb, left_depth, left_subject_mask),
        (right_rgb, right_depth, right_subject_mask),
    ):
        image = np.asarray(rgb, dtype=np.float32)
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        depth_array = np.asarray(depth, dtype=np.float32).squeeze()
        valid_depth = np.isfinite(depth_array) & (depth_array > 0)
        gray = np.mean(image[..., :3], axis=-1)
        vertical = float(np.mean(np.abs(np.diff(gray, axis=0)))) if gray.shape[0] > 1 else 0.0
        horizontal = float(np.mean(np.abs(np.diff(gray, axis=1)))) if gray.shape[1] > 1 else 0.0
        texture = float(np.clip(0.5 * (vertical + horizontal) / 24.0, 0.0, 1.0))
        subject_ratio = None
        subject_offset = None
        if subject is not None:
            mask = np.asarray(subject, dtype=bool)
            subject_ratio = float(np.mean(mask))
            rows, columns = np.nonzero(mask)
            if len(rows):
                cx = 2.0 * ((float(np.mean(columns)) + 0.5) / mask.shape[1]) - 1.0
                cy = 2.0 * ((float(np.mean(rows)) + 0.5) / mask.shape[0]) - 1.0
                subject_offset = float(math.hypot(cx, cy))
        eye_metrics.append({
            "weak_background_ratio": 1.0 - float(np.mean(valid_depth)),
            "texture_score": texture,
            "subject_pixel_ratio": subject_ratio,
            "subject_center_offset": subject_offset,
        })
    available_subject = all(item["subject_pixel_ratio"] is not None for item in eye_metrics)
    offsets = [item["subject_center_offset"] for item in eye_metrics if item["subject_center_offset"] is not None]
    return {
        "weak_background_ratio": max(float(item["weak_background_ratio"]) for item in eye_metrics),
        "texture_score": min(float(item["texture_score"]) for item in eye_metrics),
        "subject_pixel_ratio": (
            min(float(item["subject_pixel_ratio"]) for item in eye_metrics)
            if available_subject else None
        ),
        "subject_center_offset": min(float(value) for value in offsets) if offsets else None,
        "subject_segmentation_available": available_subject,
        "stereo_conservative_aggregation": True,
    }


def stereo_depth_frame_quality(left_depth: Array, right_depth: Array) -> dict[str, float | bool]:
    """Compute depth-only coverage using the worse synchronized stereo eye."""
    valid_ratios = []
    for depth in (left_depth, right_depth):
        depth_array = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_array.ndim != 2 or depth_array.size == 0:
            valid_ratios.append(0.0)
            continue
        valid = np.isfinite(depth_array) & (depth_array > 0.0)
        valid_ratios.append(float(np.mean(valid)))
    conservative_valid = min(valid_ratios)
    return {
        "weak_background_ratio": 1.0 - conservative_valid,
        "valid_depth_ratio": conservative_valid,
        "left_valid_depth_ratio": valid_ratios[0],
        "right_valid_depth_ratio": valid_ratios[1],
        "stereo_conservative_aggregation": True,
    }


def aggregate_visual_quality(
    frame_metrics: list[dict],
    config: VisualQualityConfig,
    *,
    profile: str = "balanced",
    planned_composition_offsets: Array | None = None,
) -> VisualQualityResult:
    if not frame_metrics:
        return VisualQualityResult(False, -math.inf, "no_quality_frames", {"frame_count": 0})
    weak = np.asarray([item["weak_background_ratio"] for item in frame_metrics], dtype=np.float64)
    texture = np.asarray([item["texture_score"] for item in frame_metrics], dtype=np.float64)
    bad = (weak > config.max_weak_background_ratio) | (texture < config.min_texture_score)
    current = 0
    max_run = 0
    for value in bad:
        current = current + 1 if value else 0
        max_run = max(max_run, current)
    bad_ratio = float(np.mean(bad))
    if planned_composition_offsets is not None and len(planned_composition_offsets):
        center_offset = float(np.mean(np.linalg.norm(planned_composition_offsets, axis=1)))
    else:
        measured = [item.get("subject_center_offset") for item in frame_metrics]
        measured = [float(value) for value in measured if value is not None]
        center_offset = float(np.mean(measured)) if measured else 0.0
    profile_target = _COMPOSITION_PROFILES.get(profile, _COMPOSITION_PROFILES["balanced"])[3]
    composition_fit = 1.0 - abs(center_offset - profile_target)
    valid_ratio = 1.0 - bad_ratio
    score = (
        10.0 * valid_ratio
        + composition_fit
        + float(np.mean(texture))
        - float(np.mean(weak))
    )
    ok = bad_ratio <= config.max_bad_frame_ratio and max_run <= config.max_consecutive_bad_frames
    reason = "ok"
    if bad_ratio > config.max_bad_frame_ratio:
        reason = "visual_bad_frame_ratio"
    elif max_run > config.max_consecutive_bad_frames:
        reason = "visual_consecutive_bad_frames"
    return VisualQualityResult(ok, score, reason, {
        "frame_count": len(frame_metrics),
        "valid_rendered_frame_ratio": valid_ratio,
        "bad_frame_ratio": bad_ratio,
        "max_consecutive_bad_frames": max_run,
        "mean_weak_background_ratio": float(np.mean(weak)),
        "max_weak_background_ratio": float(np.max(weak)),
        "mean_texture_score": float(np.mean(texture)),
        "min_texture_score": float(np.min(texture)),
        "composition_fit": float(composition_fit),
        "mean_composition_offset": center_offset,
        "subject_segmentation_available": all(
            bool(item.get("subject_segmentation_available", False)) for item in frame_metrics
        ),
        "stereo_conservative_aggregation": True,
    })


def aggregate_depth_coverage(
    frame_metrics: list[dict],
    min_valid_ratio: float = 0.90,
    max_bad_frame_ratio: float = 0.10,
) -> dict:
    """Evaluate stereo depth coverage independently of texture quality."""
    if not frame_metrics:
        return {
            "ok": False,
            "reason": "no_depth_frames",
            "frame_count": 0,
        }
    minimum = float(np.clip(min_valid_ratio, 0.0, 1.0))
    maximum_bad = float(np.clip(max_bad_frame_ratio, 0.0, 1.0))
    valid = 1.0 - np.asarray(
        [item["weak_background_ratio"] for item in frame_metrics],
        dtype=np.float64,
    )
    bad = valid < minimum
    bad_ratio = float(np.mean(bad))
    return {
        "ok": bad_ratio <= maximum_bad,
        "reason": "ok" if bad_ratio <= maximum_bad else "depth_bad_frame_ratio",
        "frame_count": len(frame_metrics),
        "min_required_valid_depth_ratio": minimum,
        "max_allowed_bad_frame_ratio": maximum_bad,
        "bad_frame_ratio": bad_ratio,
        "mean_valid_depth_ratio": float(np.mean(valid)),
        "min_valid_depth_ratio": float(np.min(valid)),
    }


def aggregate_multi_camera_depth_coverage(
    metrics_by_camera: dict[str, list[dict]],
    min_valid_ratio: float = 0.90,
    max_bad_frame_ratio: float = 0.10,
) -> dict:
    """Require every recorded stereo rig to satisfy the depth coverage gate."""
    if not metrics_by_camera:
        return {
            "ok": False,
            "reason": "no_depth_cameras",
            "camera_count": 0,
            "failed_cameras": [],
            "cameras": {},
        }
    cameras = {
        camera_name: aggregate_depth_coverage(
            metrics,
            min_valid_ratio=min_valid_ratio,
            max_bad_frame_ratio=max_bad_frame_ratio,
        )
        for camera_name, metrics in metrics_by_camera.items()
    }
    failed = [name for name, diagnostics in cameras.items() if not diagnostics["ok"]]
    worst = min(
        cameras,
        key=lambda name: cameras[name].get("mean_valid_depth_ratio", -1.0),
    )
    result = dict(cameras[worst])
    result.update(
        {
            "ok": not failed,
            "reason": "ok" if not failed else "camera_depth_coverage_failed",
            "camera_count": len(cameras),
            "worst_camera": worst,
            "failed_cameras": failed,
            "cameras": cameras,
        }
    )
    return result


def strategy_aware_shortlist(
    candidates: list[tuple[CameraTrajectory, float]],
    limit: int,
    strategy_weights: tuple[tuple[str, float], ...] = PlannerConfig().strategy_weights,
) -> list[tuple[CameraTrajectory, float]]:
    """Protect strategy representatives and random-walk before score fill."""
    if limit <= 0:
        return []
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    selected: list[tuple[CameraTrajectory, float]] = []
    selected_ids: set[int] = set()

    def add(item):
        identity = id(item[0])
        if identity not in selected_ids and len(selected) < limit:
            selected.append(item)
            selected_ids.add(identity)

    for strategy, _ in sorted(strategy_weights, key=lambda item: item[1], reverse=True):
        representative = next((item for item in ranked if strategy in item[0].strategy_names), None)
        if representative is not None:
            add(representative)
    random_target = min(limit, int(math.ceil(0.40 * limit)))
    random_ranked = sorted(
        (item for item in ranked if "random_walk" in item[0].strategy_names),
        key=lambda item: item[1] + 0.75 * _diversity_score(item[0]),
        reverse=True,
    )
    for item in random_ranked:
        if sum("random_walk" in selected_item[0].strategy_names for selected_item in selected) >= random_target:
            break
        add(item)
    for item in ranked:
        add(item)
    return selected


def trajectory_overlap_ratio(first: CameraTrajectory, second: CameraTrajectory) -> float:
    frame_count = min(len(first.positions), len(second.positions))
    position_close = np.linalg.norm(first.positions[:frame_count] - second.positions[:frame_count], axis=1) < 0.12
    rotation_close = []
    for left, right in zip(
        first.center_camera_to_world[:frame_count, :3, :3],
        second.center_camera_to_world[:frame_count, :3, :3],
    ):
        cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
        rotation_close.append(math.degrees(math.acos(cosine)) < 8.0)
    return float(np.mean(position_close & np.asarray(rotation_close, dtype=bool)))


def _view_bucket(trajectory: CameraTrajectory) -> tuple[int, int, int, int]:
    delta = trajectory.positions - trajectory.actual_look_at_points
    azimuth = float(np.median(np.degrees(np.unwrap(np.arctan2(delta[:, 1], delta[:, 0])))))
    radius = float(np.median(np.linalg.norm(delta[:, :2], axis=1)))
    height = float(np.median(trajectory.positions[:, 2]))
    compose_x = float(np.median(trajectory.composition_offsets[:, 0]))
    return (
        int(math.floor((azimuth + 180.0) / 45.0)),
        int(np.digitize(radius, [0.95, 1.25, 1.60])),
        int(np.digitize(height, [0.55, 0.95, 1.30])),
        int(np.digitize(compose_x, [-0.30, 0.30])),
    )


def greedy_quality_diversity_selection(
    candidates: list[tuple[CameraTrajectory, float]],
    requested_count: int,
    strategy_weights: tuple[tuple[str, float], ...] = PlannerConfig().strategy_weights,
) -> list[CameraTrajectory]:
    remaining = list(candidates)
    selected: list[CameraTrajectory] = []
    covered = [set() for _ in range(4)]
    target = dict(strategy_weights)
    selected_strategy_counts = {name: 0 for name in target}
    selected_frames = 0
    while remaining and len(selected) < requested_count:
        scored = []
        for trajectory, quality in remaining:
            overlaps = [trajectory_overlap_ratio(trajectory, prior) for prior in selected]
            maximum_overlap = max(overlaps, default=0.0)
            if maximum_overlap > 0.35:
                continue
            bucket = _view_bucket(trajectory)
            view_alignment = 1.0 if not selected else float(np.mean([
                value not in covered[index] for index, value in enumerate(bucket)
            ]))
            before_total = max(selected_frames, 1)
            before_error = sum(abs(selected_strategy_counts[name] / before_total - weight) for name, weight in target.items())
            added = {name: int(np.sum([
                trajectory.strategy_names[int(strategy_id)] == name
                for strategy_id in trajectory.strategy_id_per_frame
            ])) for name in target}
            after_total = selected_frames + len(trajectory.strategy_id_per_frame)
            after_error = sum(abs((selected_strategy_counts[name] + added[name]) / after_total - weight) for name, weight in target.items())
            strategy_alignment = before_error - after_error
            metrics = trajectory_motion_metrics(trajectory)
            random_bonus = _diversity_score(trajectory) if "random_walk" in trajectory.strategy_names else 0.0
            score = (
                quality + 4.0 * strategy_alignment + 2.0 * view_alignment
                + 1.5 * random_bonus - 2.0 * _smoothness_penalty(metrics)
                - 3.0 * maximum_overlap
            )
            scored.append((score, trajectory, bucket, added))
        if not scored:
            break
        _, chosen, bucket, added = max(scored, key=lambda item: item[0])
        selected.append(chosen)
        for index, value in enumerate(bucket):
            covered[index].add(value)
        for name, count in added.items():
            selected_strategy_counts[name] += count
        selected_frames += len(chosen.strategy_id_per_frame)
        remaining = [item for item in remaining if item[0] is not chosen]
    return selected
