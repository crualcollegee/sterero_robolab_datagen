"""Small, dependency-light diagnostics for manipulation policy rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def summarize_policy_interaction(
    actions,
    end_effector_positions,
    object_positions: Mapping[str, np.ndarray],
    *,
    actual_gripper_positions=None,
    object_gripper_contacts: Mapping[str, np.ndarray] | None = None,
    near_distance_m: float = 0.15,
    moving_step_m: float = 0.003,
) -> dict[str, object]:
    """Summarize whether a rollout contains plausible gripper/object interaction."""
    actions = np.asarray(actions, dtype=np.float64)
    ee = np.asarray(end_effector_positions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 1:
        raise ValueError(f"actions must have shape [T, D], got {actions.shape}")
    if ee.shape != (actions.shape[0], 3):
        raise ValueError(f"end_effector_positions must have shape {(actions.shape[0], 3)}, got {ee.shape}")

    close = actions[:, -1] > 0.5
    close_starts = np.flatnonzero(close & ~np.r_[False, close[:-1]])
    close_ends = np.flatnonzero(close & ~np.r_[close[1:], False])
    close_lengths = close_ends - close_starts + 1
    per_object = {}
    near_closed_steps = 0
    near_closed_moving_steps = 0
    for name, values in sorted(object_positions.items()):
        positions = np.asarray(values, dtype=np.float64)
        if positions.shape != ee.shape:
            raise ValueError(f"object {name!r} must have shape {ee.shape}, got {positions.shape}")
        step_motion = np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1)]
        distance = np.linalg.norm(positions - ee, axis=1)
        near_closed = close & (distance <= float(near_distance_m))
        moving_near_closed = near_closed & (step_motion >= float(moving_step_m))
        near_closed_steps += int(np.count_nonzero(near_closed))
        near_closed_moving_steps += int(np.count_nonzero(moving_near_closed))
        per_object[name] = {
            "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "path_length_m": float(np.sum(step_motion)),
            "max_step_motion_m": float(np.max(step_motion)),
            "min_ee_center_distance_m": float(np.min(distance)),
            "near_closed_steps": int(np.count_nonzero(near_closed)),
            "near_closed_moving_steps": int(np.count_nonzero(moving_near_closed)),
        }

    contact_steps = 0
    closed_contact_steps = 0
    contact_by_object = {}
    if object_gripper_contacts is not None:
        for name, values in sorted(object_gripper_contacts.items()):
            contact = np.asarray(values, dtype=np.bool_).reshape(-1)
            if contact.shape[0] != actions.shape[0]:
                raise ValueError(f"object contact {name!r} must have {actions.shape[0]} values")
            closed_contact = contact & close
            object_contact_steps = int(np.count_nonzero(contact))
            object_closed_contact_steps = int(np.count_nonzero(closed_contact))
            contact_steps += object_contact_steps
            closed_contact_steps += object_closed_contact_steps
            contact_by_object[name] = {
                "contact_steps": object_contact_steps,
                "closed_contact_steps": object_closed_contact_steps,
            }

    result: dict[str, object] = {
        "frame_count": int(actions.shape[0]),
        "gripper_close_steps": int(np.count_nonzero(close)),
        "gripper_close_fraction": float(np.mean(close)),
        "gripper_close_intervals": int(close_starts.size),
        "longest_gripper_close_steps": int(np.max(close_lengths)) if close_lengths.size else 0,
        "near_closed_object_steps": near_closed_steps,
        "near_closed_moving_object_steps": near_closed_moving_steps,
        "gripper_object_contact_steps": contact_steps,
        "closed_gripper_object_contact_steps": closed_contact_steps,
        "verified_gripper_object_contact": bool(contact_steps > 0),
        "verified_closed_gripper_object_contact": bool(closed_contact_steps > 0),
        "plausible_object_interaction": bool(closed_contact_steps > 0 or near_closed_moving_steps > 0),
        "near_distance_m": float(near_distance_m),
        "moving_step_m": float(moving_step_m),
        "per_object": per_object,
        "contact_by_object": contact_by_object,
    }
    if actual_gripper_positions is not None:
        actual = np.asarray(actual_gripper_positions, dtype=np.float64).reshape(-1)
        if actual.shape[0] != actions.shape[0]:
            raise ValueError("actual_gripper_positions must have one value per action")
        result["actual_gripper_position_min"] = float(np.min(actual))
        result["actual_gripper_position_max"] = float(np.max(actual))
    return result
