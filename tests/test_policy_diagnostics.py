import numpy as np

from robolab.core.policy_diagnostics import summarize_policy_interaction


def test_policy_diagnostics_detects_closed_gripper_object_motion():
    actions = np.zeros((5, 8), dtype=np.float32)
    actions[2:4, -1] = 1.0
    ee = np.zeros((5, 3), dtype=np.float32)
    obj = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0.11, 0, 0], [0.11, 0, 0]])
    result = summarize_policy_interaction(actions, ee, {"fruit": obj})
    assert result["gripper_close_intervals"] == 1
    assert result["near_closed_moving_object_steps"] == 1
    assert result["plausible_object_interaction"] is True


def test_policy_diagnostics_rejects_motion_while_open_or_far():
    actions = np.zeros((4, 8), dtype=np.float32)
    ee = np.zeros((4, 3), dtype=np.float32)
    obj = np.array([[1.0, 0, 0], [1.1, 0, 0], [1.2, 0, 0], [1.3, 0, 0]])
    result = summarize_policy_interaction(actions, ee, {"fruit": obj})
    assert result["gripper_close_steps"] == 0
    assert result["plausible_object_interaction"] is False


def test_policy_diagnostics_reports_sensor_verified_closed_contact():
    actions = np.zeros((5, 8), dtype=np.float64)
    actions[2:4, -1] = 1.0
    ee = np.zeros((5, 3), dtype=np.float64)
    obj = np.ones((5, 3), dtype=np.float64)
    result = summarize_policy_interaction(
        actions,
        ee,
        {"fruit": obj},
        object_gripper_contacts={"fruit": np.array([False, False, True, True, False])},
    )
    assert result["gripper_object_contact_steps"] == 2
    assert result["closed_gripper_object_contact_steps"] == 2
    assert result["verified_closed_gripper_object_contact"] is True
    assert result["plausible_object_interaction"] is True


def test_policy_diagnostics_does_not_accept_open_gripper_contact_as_grasp():
    actions = np.zeros((3, 8), dtype=np.float64)
    ee = np.zeros((3, 3), dtype=np.float64)
    obj = np.ones((3, 3), dtype=np.float64)
    result = summarize_policy_interaction(
        actions,
        ee,
        {"fruit": obj},
        object_gripper_contacts={"fruit": np.ones(3, dtype=np.bool_)},
    )
    assert result["verified_gripper_object_contact"] is True
    assert result["verified_closed_gripper_object_contact"] is False
    assert result["plausible_object_interaction"] is False
