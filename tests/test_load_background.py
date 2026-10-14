from __future__ import annotations

from robolab.core.scenes.load_background import background_mesh_is_safe_to_hide


def test_structural_background_meshes_are_never_hidden() -> None:
    assert not background_mesh_is_safe_to_hide(
        "/Room/Meshes/wall/wall_001", [0.0, 0.0, 0.0], [0.2, 0.8, 0.8]
    )
    assert not background_mesh_is_safe_to_hide(
        "/Room/Meshes/floor/floor_001", [0.0, 0.0, 0.0], [0.7, 0.7, 0.1]
    )


def test_unknown_or_large_background_meshes_are_preserved() -> None:
    assert not background_mesh_is_safe_to_hide(
        "/Room/Meshes/mesh_0042", [-2.0, -2.0, 0.0], [2.0, 2.0, 3.0]
    )
    assert not background_mesh_is_safe_to_hide(
        "/Room/Props/plant_merged", [0.0, 0.0, 0.0], [2.0, 1.0, 2.0]
    )


def test_only_compact_named_props_may_be_hidden() -> None:
    assert background_mesh_is_safe_to_hide(
        "/Room/Props/plant_01", [0.0, 0.0, 0.0], [0.35, 0.35, 0.9]
    )
    assert background_mesh_is_safe_to_hide(
        "/Room/Decoration/vase_01", [0.0, 0.0, 0.0], [0.2, 0.2, 0.4]
    )
