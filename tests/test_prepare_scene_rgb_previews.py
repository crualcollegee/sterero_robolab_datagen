import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_scene_rgb_previews.py"
SPEC = importlib.util.spec_from_file_location("prepare_scene_rgb_previews", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_evenly_spaced_indices_cover_sequence():
    assert MODULE.evenly_spaced_indices(11, 6) == [0, 2, 4, 6, 8, 10]
    assert MODULE.evenly_spaced_indices(2, 6) == [0, 1]


def test_prepare_copies_evenly_spaced_frames(tmp_path):
    scene = tmp_path / "scene"
    source = scene / "camera03" / "rgb" / "left"
    source.mkdir(parents=True)
    for index in range(11):
        (source / f"{index:06d}.png").write_bytes(str(index).encode())

    output = tmp_path / "previews" / "scene"
    written = MODULE.prepare(scene, output, "camera03", "left", 3)

    assert [path.name for path in written] == [
        "camera03_left_000000.png",
        "camera03_left_000005.png",
        "camera03_left_000010.png",
    ]
    assert [path.read_bytes() for path in written] == [b"0", b"5", b"10"]
