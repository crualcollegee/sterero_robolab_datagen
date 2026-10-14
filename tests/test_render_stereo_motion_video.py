import importlib.util
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_stereo_motion_video.py"
SPEC = importlib.util.spec_from_file_location("render_stereo_motion_video", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def test_extract_cameras_reads_multiple_views_in_one_archive(tmp_path):
    archive_path = tmp_path / "scene.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_member(archive, "scene/camera00/metadata.json", b'{"view": 0}')
        _write_member(archive, "scene/camera01/metadata.json", b'{"view": 1}')
        _write_member(archive, "scene/camera02/ignored.txt", b"ignored")

    roots = MODULE.extract_cameras(
        archive_path, ("camera00", "camera01"), tmp_path / "extract"
    )

    assert (roots["camera00"] / "metadata.json").read_bytes() == b'{"view": 0}'
    assert (roots["camera01"] / "metadata.json").read_bytes() == b'{"view": 1}'
    assert not (tmp_path / "extract" / "camera02").exists()


def test_extract_cameras_rejects_missing_requested_view(tmp_path):
    archive_path = tmp_path / "scene.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_member(archive, "scene/camera00/metadata.json", b"{}")

    with pytest.raises(ValueError, match="camera04"):
        MODULE.extract_cameras(
            archive_path, ("camera00", "camera04"), tmp_path / "extract"
        )


def test_camera_has_flow_accepts_optional_flow_pair(tmp_path):
    camera = tmp_path / "camera00"
    camera.mkdir()
    assert MODULE.camera_has_flow(camera) is False

    (camera / "flow").mkdir()
    with pytest.raises(ValueError, match="flow and flow_masks"):
        MODULE.camera_has_flow(camera)

    (camera / "flow_masks").mkdir()
    assert MODULE.camera_has_flow(camera) is True


def test_resize_rgbd_flow_does_not_blend_across_depth_edges():
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    depth = np.array([[1, 1, 9, 9], [1, 1, 9, 9]], dtype=np.float32)
    flow = np.zeros((2, 4, 2), dtype=np.float32)
    flow[:, 2:, 0] = 8.0
    valid = np.ones((2, 4), dtype=bool)
    intrinsics = np.array(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    _, depth_small, flow_small, valid_small, _, scale = MODULE.resize_rgbd_flow(
        rgb, depth, flow, valid, intrinsics, width=2
    )

    assert scale == pytest.approx(0.5)
    np.testing.assert_array_equal(depth_small, np.array([[1.0, 9.0]], np.float32))
    np.testing.assert_array_equal(flow_small[..., 0], np.array([[0.0, 4.0]], np.float32))
    assert valid_small.all()
