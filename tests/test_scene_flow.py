import numpy as np

from robolab.core.scene_flow import (
    backward_to_forward_flow,
    compose_backward_motion_vectors,
    instance_id_path_mask,
    remove_camera_motion,
    sanitize_native_forward_flow,
    static_camera_forward_flow,
)


def test_instance_id_path_mask_selects_background_descendants() -> None:
    ids = np.array([[0, 3, 7], [8, 9, 3]], dtype=np.int32)[..., None]
    info = {
        "idToLabels": {
            "0": "BACKGROUND",
            "3": "/World/envs/env_0/scene/asset_backgrounds/room/asset/wall",
            "7": {"primPath": "/World/envs/env_0/scene/robot/link"},
            "8": ["/World/envs/env_0/scene/geometric_background/floor"],
            "9": "/World/envs/env_0/scene/asset_backgrounds/roommate",
            "10": [
                "/World/envs/env_0/scene/asset_backgrounds/room/asset/chair",
                "/World/envs/env_0/scene/robot/link",
            ],
        }
    }

    mask = instance_id_path_mask(
        ids,
        info,
        [
            "/World/envs/env_0/scene/asset_backgrounds/room",
            "/World/envs/env_0/scene/geometric_background/floor",
        ],
    )

    np.testing.assert_array_equal(
        mask,
        np.array([[False, True, False], [True, False, True]]),
    )

    mixed = instance_id_path_mask(
        np.array([[10]], dtype=np.int32),
        info,
        ["/World/envs/env_0/scene/asset_backgrounds/room"],
    )
    assert not mixed.any()


def test_instance_id_path_mask_accepts_json_mapping() -> None:
    ids = np.array([[1, 2]], dtype=np.int32)
    info = {
        "idToLabels": (
            '{"1": "/World/background/mesh", '
            '"2": "/World/dynamic/mesh"}'
        )
    }
    mask = instance_id_path_mask(ids, info, ["/World/background"])
    np.testing.assert_array_equal(mask, np.array([[True, False]]))


def test_compose_backward_motion_vectors() -> None:
    fields = [np.zeros((8, 10, 2), np.float32) for _ in range(2)]
    fields[0][..., 0] = -1.0
    fields[1][..., 0] = -1.0
    combined, valid = compose_backward_motion_vectors(fields)
    np.testing.assert_allclose(combined[:, 2:, 0], -2.0)
    assert valid[:, 2:].all()
    assert not valid[:, :2].any()


def test_backward_to_forward_flow() -> None:
    backward = np.zeros((6, 8, 2), np.float32)
    backward[..., 0] = -1.0
    forward, valid = backward_to_forward_flow(backward)
    np.testing.assert_allclose(forward[:, :-1, 0], 1.0)
    assert valid[:, :-1].all()
    assert not valid[:, -1].any()


def test_remove_camera_motion_cancels_static_scene_translation() -> None:
    height, width = 9, 11
    k = np.array([[20.0, 0.0, 5.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]], np.float32)
    depth = np.full((height, width), 2.0, np.float32)
    observed = np.zeros((height, width, 2), np.float32)
    observed[..., 0] = -1.0
    c2w = np.eye(4, dtype=np.float32)
    c2w_next = np.eye(4, dtype=np.float32)
    c2w_next[0, 3] = 0.1

    residual, camera_flow, valid = remove_camera_motion(
        depth, observed, k, c2w, c2w_next
    )

    np.testing.assert_allclose(camera_flow[..., 0][valid], -1.0, atol=1.0e-5)
    np.testing.assert_allclose(residual[valid], 0.0, atol=1.0e-5)
    assert valid[:, 1:].all()
    assert not valid[:, 0].any()


def test_remove_camera_motion_preserves_object_motion() -> None:
    height, width = 7, 9
    k = np.array([[20.0, 0.0, 4.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]], np.float32)
    depth = np.full((height, width), 2.0, np.float32)
    observed = np.zeros((height, width, 2), np.float32)
    observed[..., 0] = 1.0
    c2w = np.eye(4, dtype=np.float32)

    residual, camera_flow, valid = remove_camera_motion(
        depth, observed, k, c2w, c2w
    )

    np.testing.assert_allclose(camera_flow[valid], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(residual[..., 0][valid], 1.0, atol=1.0e-6)
    assert valid[:, :-1].all()
    assert not valid[:, -1].any()


def test_native_forward_invalid_mask_does_not_include_out_of_frame() -> None:
    depth = np.ones((4, 6), np.float32)
    forward = np.zeros((4, 6, 2), np.float32)
    forward[..., 0] = 20.0  # Every endpoint leaves the image.
    flow, invalid = sanitize_native_forward_flow(forward, depth)
    np.testing.assert_array_equal(flow, forward)
    assert invalid.dtype == np.bool_
    assert not invalid.any()


def test_native_forward_invalid_mask_uses_depth_finite_flow_and_native_sentinel() -> None:
    depth = np.ones((3, 5), np.float32)
    depth[0, 0] = 0.0
    forward = np.ones((3, 5, 2), np.float32)
    forward[1, 2, 0] = np.nan
    forward[2, 4] = (-1.0, -1.0)
    flow, invalid = sanitize_native_forward_flow(forward, depth)
    assert invalid[0, 0] and invalid[1, 2] and invalid[2, 4]
    assert int(invalid.sum()) == 3
    assert np.all(flow[invalid] == 0.0)


def test_native_forward_keeps_non_sentinel_negative_flow_valid() -> None:
    depth = np.ones((2, 3), np.float32)
    forward = np.full((2, 3, 2), (-1.0, -1.0), np.float32)
    forward[0, 0] = (-1.0, -0.999)
    forward[0, 1] = (-2.0, -1.0)
    flow, invalid = sanitize_native_forward_flow(forward, depth)
    assert not invalid[0, 0]
    assert not invalid[0, 1]
    assert invalid[0, 2]
    np.testing.assert_array_equal(flow[0, :2], forward[0, :2])


def test_static_camera_forward_flow_keeps_out_of_frame_endpoints_valid() -> None:
    height, width = 5, 7
    k = np.array([[20.0, 0.0, 3.0], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]], np.float32)
    depth = np.full((height, width), 2.0, np.float32)
    depth[0, 0] = 0.0
    c2w = np.eye(4, dtype=np.float32)
    c2w_next = np.eye(4, dtype=np.float32)
    c2w_next[0, 3] = 0.1

    flow, invalid = static_camera_forward_flow(depth, k, c2w, c2w_next)

    np.testing.assert_allclose(flow[..., 0][~invalid], -1.0, atol=1.0e-5)
    np.testing.assert_allclose(flow[..., 1][~invalid], 0.0, atol=1.0e-5)
    assert invalid[0, 0]
    assert int(invalid.sum()) == 1
    assert not invalid[:, 1:].any()
