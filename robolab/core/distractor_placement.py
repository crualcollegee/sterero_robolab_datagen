"""Pure helpers for safe tabletop distractor placement and settling checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def aabb_is_clear(
    candidate_min,
    candidate_max,
    forbidden_bounds: Iterable[tuple[np.ndarray, np.ndarray]],
    clearance: float = 0.0,
) -> bool:
    """Return whether a candidate AABB is separated from every forbidden AABB."""
    candidate_min = np.asarray(candidate_min, dtype=np.float64)
    candidate_max = np.asarray(candidate_max, dtype=np.float64)
    if candidate_min.shape != (3,) or candidate_max.shape != (3,):
        raise ValueError("AABB endpoints must have shape (3,)")
    if not np.isfinite(candidate_min).all() or not np.isfinite(candidate_max).all():
        return False
    margin = max(0.0, float(clearance))
    for other_min, other_max in forbidden_bounds:
        other_min = np.asarray(other_min, dtype=np.float64)
        other_max = np.asarray(other_max, dtype=np.float64)
        separated = np.any(
            (candidate_max + margin <= other_min)
            | (candidate_min - margin >= other_max)
        )
        if not separated:
            return False
    return True


def settling_motion_metrics(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Summarize per-object displacement across an injected-scene settling pass."""
    names = sorted(set(before) & set(after))
    displacement = {
        name: float(
            np.linalg.norm(
                np.asarray(after[name], dtype=np.float64)
                - np.asarray(before[name], dtype=np.float64)
            )
        )
        for name in names
    }
    return {
        "object_count": len(names),
        "max_displacement_m": max(displacement.values(), default=0.0),
        "per_object_displacement_m": displacement,
    }
