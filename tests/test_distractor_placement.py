import numpy as np

from robolab.core.distractor_placement import aabb_is_clear, settling_motion_metrics


def test_aabb_is_clear_rejects_overlap_and_clearance_violation():
    forbidden = [(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))]
    assert not aabb_is_clear([0.5, 0.5, 0.5], [1.5, 1.5, 1.5], forbidden)
    assert aabb_is_clear([1.1, 0.0, 0.0], [2.0, 1.0, 1.0], forbidden)
    assert not aabb_is_clear(
        [1.1, 0.0, 0.0], [2.0, 1.0, 1.0], forbidden, clearance=0.11
    )


def test_aabb_is_clear_treats_touching_faces_as_separate():
    forbidden = [(np.zeros(3), np.ones(3))]
    assert aabb_is_clear([1.0, 0.2, 0.2], [2.0, 0.8, 0.8], forbidden)


def test_aabb_is_clear_rejects_nonfinite_bounds():
    assert not aabb_is_clear([0.0, np.nan, 0.0], [1.0, 1.0, 1.0], [])


def test_settling_motion_metrics_reports_largest_object_motion():
    result = settling_motion_metrics(
        {"apple": np.array([0.0, 0.0, 0.0]), "bin": np.array([1.0, 0.0, 0.0])},
        {"apple": np.array([0.03, 0.04, 0.0]), "bin": np.array([1.0, 0.0, 0.0])},
    )
    assert result["object_count"] == 2
    assert np.isclose(result["max_displacement_m"], 0.05)
    assert np.isclose(result["per_object_displacement_m"]["apple"], 0.05)
