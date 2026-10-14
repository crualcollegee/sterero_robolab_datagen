"""Load static USD geometry backgrounds for RoboLab data generation.

RoboLab's original ``background`` is a DomeLight HDR/EXR texture and does not
produce depth. This module loads actual USD geometry as a background layer so
background pixels can have depth while staying outside the robot workspace.
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robolab.constants import PACKAGE_DIR


@dataclass(frozen=True)
class BackgroundPreset:
    asset: str
    yaw_deg: float = 0.0
    scale: float = 1.0
    safe_max_x: float = -0.22
    floor_z: float = -0.58
    y_center: float = 0.0


ASSET_BACKGROUND_PRESETS: dict[str, BackgroundPreset] = {
    # Kitchen fixture assets. These are useful replacement backgrounds because
    # they are real USD geometry, not HDRI image backgrounds.
    "asset_kitchen_single_wall": BackgroundPreset("assets/fixtures/kitchen_single_wall.usd", yaw_deg=0.0, scale=0.85),
    "asset_kitchen_l_shaped": BackgroundPreset("assets/fixtures/kitchen_l_shaped_static.usd", yaw_deg=0.0, scale=0.82),
    "asset_kitchen_island": BackgroundPreset("assets/fixtures/kitchen_island.usd", yaw_deg=0.0, scale=0.82),
    "asset_kitchen_peninsula": BackgroundPreset("assets/fixtures/kitchen_peninsula.usd", yaw_deg=0.0, scale=0.78),
    "asset_kitchen_u_shaped": BackgroundPreset("assets/fixtures/kitchen_u_shaped.usd", yaw_deg=0.0, scale=0.76),
    # RoboLab scene USDs reused as far-field static geometry. They are pushed
    # behind the manipulation area so they do not affect task execution.
    "asset_workdesk": BackgroundPreset("assets/scenes/workdesk.usda", scale=0.82),
    "asset_workdesk_bin": BackgroundPreset("assets/scenes/workdesk_bin.usda", scale=0.78),
    "asset_workdesk_snacks": BackgroundPreset("assets/scenes/workdesk_snacks.usda", scale=0.78),
    "asset_shelf_cleaning": BackgroundPreset("assets/scenes/shelf_with_cleaning_products.usda", scale=0.78),
    "asset_shelf_cleaning2": BackgroundPreset("assets/scenes/shelf_with_cleaning_products2.usda", scale=0.78),
    "asset_shelf_mugs": BackgroundPreset("assets/scenes/shelf_mugs_jug_bowl.usda", scale=0.78),
    "asset_wire_shelf": BackgroundPreset("assets/scenes/wire_shelf_mugs_plate_spatula.usda", scale=0.78),
    "asset_tools_container": BackgroundPreset("assets/scenes/tools_container.usda", scale=0.78),
    "asset_tools_picking": BackgroundPreset("assets/scenes/tools_picking.usda", scale=0.78),
    "asset_clutter_shelf": BackgroundPreset("assets/scenes/clutter_shelf.usda", scale=0.82),
    "asset_front_shelf": BackgroundPreset("assets/scenes/front_of_shelf.usda", scale=0.82),
    "asset_cutlery_shelf": BackgroundPreset("assets/scenes/cutlery_shelf.usda", scale=0.82),
    "asset_food_packing_dense": BackgroundPreset("assets/scenes/food_packing_dense.usda", scale=0.78),
    # Isaac Sim 5.1 official environment assets. These paths are absolute on
    # AutoDL because the asset pack is downloaded outside the RoboLab checkout.
    "isaac_grid_default": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd", scale=1.0),
    "isaac_gridroom_black": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/gridroom_black.usd", scale=0.85),
    "isaac_gridroom_curved": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/gridroom_curved.usd", scale=0.85),
    "isaac_simple_room": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd", scale=0.65),
    "isaac_warehouse": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd", scale=0.35),
    "isaac_warehouse_shelves": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd", scale=0.35),
    "isaac_warehouse_forklifts": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd", scale=0.35),
    "isaac_warehouse_extras": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Stage/warehouse_extras.usd", scale=0.35),
    "isaac_full_warehouse": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", scale=0.18),
    "isaac_office": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd", scale=0.28),
    "isaac_hospital": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Hospital/hospital.usd", scale=0.25),
    "isaac_digital_twin_warehouse": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd", scale=0.25),
    "isaac_rivermark": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/rivermark.usd", scale=0.05),
    "isaac_jetracer_track": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Jetracer/jetracer_track_solid.usd", scale=0.55),
    "isaac_terrain_flat": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Terrains/flat_plane.usd", scale=1.0),
    "isaac_terrain_rough": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Terrains/rough_plane.usd", scale=1.0),
    "isaac_terrain_slope": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Terrains/slope.usd", scale=1.0),
    "isaac_terrain_stairs": BackgroundPreset("/autodl-fs/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Terrains/stairs.usd", scale=1.0),
}


INTERIORAGENT_ROOT = Path(os.environ.get("INTERIORAGENT_ROOT", "/autodl-fs/data/spatialverse/InteriorAgent"))
INTERIORAGENT_ARCHIVE_ROOT = Path(
    os.environ.get("INTERIORAGENT_ARCHIVE_ROOT", "/autodl-fs/data/spatialverse/InteriorAgent_archives")
)
INTERIORAGENT_RUNTIME_CACHE = Path(
    os.environ.get("INTERIORAGENT_RUNTIME_CACHE", "/root/.cache/robolab/interioragent")
)
_INTERIORAGENT_ARCHIVE_PREFIX = "interioragent-archive:"
_INTERIORAGENT_PROCESS_CACHE = INTERIORAGENT_RUNTIME_CACHE / str(os.getpid())
_INTERIORAGENT_CACHE_REGISTERED = False


def _discover_interioragent_presets(
    root: Path = INTERIORAGENT_ROOT,
    archive_root: Path = INTERIORAGENT_ARCHIVE_ROOT,
) -> dict[str, BackgroundPreset]:
    """Discover local InteriorAgent room USDs.

    The HuggingFace dataset is organized as ``kujiale_XXXX/kujiale_XXXX.usda``.
    Registering them dynamically keeps the generator usable after more scenes
    are downloaded without requiring a code edit per room.
    """
    presets: dict[str, BackgroundPreset] = {}
    if root.exists():
        for scene_dir in sorted(root.glob("kujiale_*")):
            if not scene_dir.is_dir():
                continue
            usd_path = scene_dir / f"{scene_dir.name}.usda"
            if usd_path.exists() and not _is_git_lfs_pointer(usd_path):
                presets[f"interioragent_{scene_dir.name}"] = BackgroundPreset(str(usd_path), scale=1.0)

    if archive_root.exists():
        for archive_path in sorted(archive_root.glob("kujiale_*.tar.gz")):
            scene_name = archive_path.name.removesuffix(".tar.gz")
            style = f"interioragent_{scene_name}"
            presets.setdefault(
                style,
                BackgroundPreset(f"{_INTERIORAGENT_ARCHIVE_PREFIX}{archive_path}", scale=1.0),
            )
    return presets


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _cleanup_interioragent_process_cache() -> None:
    shutil.rmtree(_INTERIORAGENT_PROCESS_CACHE, ignore_errors=True)


def _materialize_interioragent_archive(archive_path: Path) -> Path:
    """Extract one archived room into a process-local cache on the system disk."""
    global _INTERIORAGENT_CACHE_REGISTERED

    if not archive_path.is_file():
        raise FileNotFoundError(f"InteriorAgent archive not found: {archive_path}")
    scene_name = archive_path.name.removesuffix(".tar.gz")
    target_dir = _INTERIORAGENT_PROCESS_CACHE / scene_name
    usd_path = target_dir / f"{scene_name}.usda"
    if usd_path.is_file() and not _is_git_lfs_pointer(usd_path):
        return usd_path

    if not _INTERIORAGENT_CACHE_REGISTERED:
        atexit.register(_cleanup_interioragent_process_cache)
        _INTERIORAGENT_CACHE_REGISTERED = True

    shutil.rmtree(target_dir, ignore_errors=True)
    _INTERIORAGENT_PROCESS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_root = _INTERIORAGENT_PROCESS_CACHE.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (cache_root / member.name).resolve()
            if cache_root not in destination.parents and destination != cache_root:
                raise ValueError(f"Unsafe path in InteriorAgent archive {archive_path}: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in InteriorAgent archive {archive_path}: {member.name}")
        archive.extractall(cache_root)

    if not usd_path.is_file() or _is_git_lfs_pointer(usd_path):
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ValueError(f"Archive does not contain a valid {scene_name}.usda: {archive_path}")
    print(f"[INFO] Extracted InteriorAgent background archive: {archive_path} -> {target_dir}")
    return usd_path


ASSET_BACKGROUND_PRESETS.update(_discover_interioragent_presets())
ASSET_BACKGROUND_STYLE_NAMES = tuple(ASSET_BACKGROUND_PRESETS.keys())

WORKSPACE_KEEPOUT_MIN = np.asarray([-0.35, -0.85, -0.10], dtype=np.float64)
WORKSPACE_KEEPOUT_MAX = np.asarray([1.25, 0.85, 1.75], dtype=np.float64)


def _clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "background"


def _resolve_asset_path(asset: str) -> str:
    if asset.startswith(_INTERIORAGENT_ARCHIVE_PREFIX):
        archive_path = Path(asset.removeprefix(_INTERIORAGENT_ARCHIVE_PREFIX))
        return str(_materialize_interioragent_archive(archive_path))
    path = Path(asset)
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(PACKAGE_DIR) / asset
    if not resolved.exists():
        raise FileNotFoundError(f"Background USD asset not found: {asset} -> {resolved}")
    if _is_git_lfs_pointer(resolved):
        raise ValueError(f"Background USD asset is a Git LFS pointer, not a downloaded USD: {resolved}")
    return str(resolved)


def _bbox_world(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    min_v = None
    max_v = None
    for child in Usd.PrimRange(prim):
        if not child.IsActive() or not child.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(child)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        bbox = cache.ComputeWorldBound(child).ComputeAlignedBox()
        if bbox.IsEmpty():
            continue
        child_min = np.asarray(bbox.GetMin(), dtype=np.float64)
        child_max = np.asarray(bbox.GetMax(), dtype=np.float64)
        min_v = child_min if min_v is None else np.minimum(min_v, child_min)
        max_v = child_max if max_v is None else np.maximum(max_v, child_max)
    if min_v is None or max_v is None:
        return None
    return min_v, max_v


def _hide_primary_ray_lights(stage, prim_path: str) -> int:
    """Disable lights embedded in background USDs from appearing as image backgrounds.

    Interior room USDs often contain a DomeLight with an HDR texture. That HDR
    renders in RGB but has no geometry depth, which defeats the purpose of using
    the USD as a depth-producing background. The data-generation script supplies
    its own lighting, so embedded background lights are hidden here.
    """
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return 0
    hidden = 0
    for prim in Usd.PrimRange(root):
        type_name = prim.GetTypeName()
        if not prim.IsActive() or not type_name.endswith("Light"):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        visible_attr = prim.GetAttribute("visibleInPrimaryRay")
        if visible_attr:
            visible_attr.Set(False)
        inputs_visible_attr = prim.GetAttribute("inputs:visibleInPrimaryRay")
        if inputs_visible_attr:
            inputs_visible_attr.Set(False)
        hidden += 1
    return hidden


def _strip_physics_subtree(stage, prim_path: str) -> int:
    """Remove physics APIs from visual-only background assets.

    InteriorAgent meshes include rigid-body and collision APIs for furniture,
    doors, windows, etc. RoboLab only needs these assets as static RGB/depth
    geometry. Leaving physics schemas active makes Isaac/PhysX cook hundreds of
    unrelated meshes and can trigger nested-rigid-body errors.
    """
    from pxr import Usd

    try:
        from pxr import UsdPhysics
    except Exception:
        UsdPhysics = None
    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return 0

    api_classes = []
    if UsdPhysics is not None:
        for name in ("RigidBodyAPI", "CollisionAPI", "MeshCollisionAPI", "MassAPI"):
            api_cls = getattr(UsdPhysics, name, None)
            if api_cls is not None:
                api_classes.append(api_cls)
    if PhysxSchema is not None:
        for name in (
            "PhysxRigidBodyAPI",
            "PhysxCollisionAPI",
            "PhysxContactReportAPI",
            "PhysxMassAPI",
        ):
            api_cls = getattr(PhysxSchema, name, None)
            if api_cls is not None:
                api_classes.append(api_cls)

    removed = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsActive():
            continue
        for api_cls in api_classes:
            try:
                if prim.HasAPI(api_cls):
                    prim.RemoveAPI(api_cls)
                    removed += 1
            except Exception:
                continue
        for attr_name in (
            "physics:collisionEnabled",
            "physics:kinematicEnabled",
            "physics:rigidBodyEnabled",
        ):
            attr = prim.GetAttribute(attr_name)
            if attr:
                attr.Set(False)
    return removed


def _deinstance_subtree(stage, prim_path: str) -> int:
    """Turn USD instance prims into ordinary prims so camera depth sees their meshes.

    Several Isaac Sim environment assets use native USD instanceable Xforms for
    repeated furniture. RGB renders those instances, but RTX/Replicator depth can
    omit them in some headless TiledCamera paths. Authoring instanceable=false on
    the loaded reference keeps the visual layout while exposing prototype meshes
    as normal scene geometry.
    """
    from pxr import Usd

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return 0
    changed_total = 0
    # Deinstancing one level can expose nested instance prims, so repeat until stable.
    for _ in range(8):
        changed = 0
        for prim in Usd.PrimRange(root):
            if prim.IsInstance():
                prim.SetInstanceable(False)
                changed += 1
        changed_total += changed
        if changed == 0:
            break
    return changed_total


def _is_interioragent_floor_mesh_path(prim_path: str) -> bool:
    """Return whether a mesh belongs to InteriorAgent's explicit floor scope."""
    parts = tuple(part.lower() for part in str(prim_path).split("/") if part)
    return any(parts[index : index + 2] == ("meshes", "floor") for index in range(len(parts) - 1))


_STRUCTURAL_BACKGROUND_TOKENS = frozenset(
    {
        "architecture",
        "ceiling",
        "column",
        "door",
        "floor",
        "foundation",
        "ground",
        "pillar",
        "roof",
        "shell",
        "slab",
        "stair",
        "stairs",
        "structure",
        "wall",
        "window",
    }
)
_REMOVABLE_BACKGROUND_TOKENS = frozenset(
    {
        "basket",
        "bin",
        "chair",
        "clutter",
        "decor",
        "decoration",
        "lamp",
        "plant",
        "prop",
        "stool",
        "trash",
        "vase",
    }
)


def background_mesh_is_safe_to_hide(
    prim_path: str,
    bounds_min,
    bounds_max,
) -> bool:
    """Return whether a background Gprim may be removed as a small prop.

    InteriorAgent rooms vary substantially in USD granularity.  One Gprim can
    represent a vase in one room and an entire wall/floor shell in another.
    The previous safety pass hid every AABB intersection, so one conservative
    AABB hit could erase most of a room.  Unknown and structural meshes now
    default to preservation; only explicitly prop-like, compact meshes may be
    hidden to clear the robot workspace or a camera sightline.
    """
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(prim_path).lower())
        if token
    }
    if tokens & _STRUCTURAL_BACKGROUND_TOKENS:
        return False
    if not tokens & _REMOVABLE_BACKGROUND_TOKENS:
        return False

    minimum = np.asarray(bounds_min, dtype=np.float64)
    maximum = np.asarray(bounds_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        return False
    extent = maximum - minimum
    if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
        return False

    # A chair/plant-sized object may be removed.  Large furniture and merged
    # meshes are retained even if their names happen to contain a prop token.
    return bool(float(np.max(extent)) <= 1.25 and float(np.prod(extent)) <= 0.45)


def _make_interioragent_floors_double_sided(stage, prim_path: str) -> int:
    """Make only source room floor meshes visible to rays from the room side.

    InteriorAgent floor assets are authored as single-sided meshes. Several
    rooms expose their back face to the imported RoboLab workspace, so RTX RGB
    can shade the floor while the depth annotator returns no hit. Restricting
    this override to the dataset's explicit ``Meshes/floor`` scope avoids the
    wall, furniture, and generic-name overrides that previously distorted room
    depth.
    """
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return 0
    changed = 0
    for prim in Usd.PrimRange(root):
        if (
            not prim.IsActive()
            or not prim.IsA(UsdGeom.Mesh)
            or not _is_interioragent_floor_mesh_path(str(prim.GetPath()))
        ):
            continue
        attr = UsdGeom.Mesh(prim).GetDoubleSidedAttr()
        if attr.Get() is not True:
            attr.Set(True)
            changed += 1
    return changed


_FALLBACK_MATERIAL_PALETTE = (
    (0.72, 0.28, 0.20),  # terracotta
    (0.18, 0.42, 0.46),  # muted teal
    (0.55, 0.62, 0.38),  # sage
    (0.78, 0.57, 0.18),  # mustard
    (0.30, 0.43, 0.62),  # dusty blue
    (0.50, 0.22, 0.28),  # burgundy
    (0.70, 0.48, 0.34),  # warm clay
    (0.34, 0.52, 0.42),  # desaturated green
)


def _repair_background_material_assets(stage, prim_path: str, asset_path: str, seed: int) -> dict[str, Any]:
    """Repair broken room texture paths and color genuinely missing materials.

    Some InteriorAgent MDL/USD materials reference a texture using a path that
    is relative to a different layer even though the texture exists in the
    downloaded room. Other source rooms contain references to files that are
    absent from the dataset. The first case is re-anchored to the real file;
    the second receives a deterministic, muted interior-design color instead
    of silently rendering as black or gray.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return {"repaired_texture_paths": 0, "fallback_materials": 0, "fallback_bound_prims": 0}

    room_root = Path(asset_path).resolve().parent
    basename_index: dict[str, list[Path]] | None = None

    def find_asset(asset_name: str) -> Path | None:
        nonlocal basename_index
        basename = Path(asset_name).name
        preferred = room_root / "Materials" / "Textures" / basename
        if preferred.is_file():
            return preferred
        if basename_index is None:
            basename_index = {}
            for candidate in room_root.rglob("*"):
                if candidate.is_file():
                    basename_index.setdefault(candidate.name, []).append(candidate)
        matches = basename_index.get(basename, [])
        return matches[0] if len(matches) == 1 else None

    def containing_material(prim):
        current = prim
        while current and current.GetPath().HasPrefix(root.GetPath()):
            if current.IsA(UsdShade.Material):
                return current
            current = current.GetParent()
        return None

    def is_base_color_asset(attribute_name: str) -> bool:
        name = attribute_name.lower()
        if any(token in name for token in ("normal", "rough", "metal", "specular", "opacity", "reflection")):
            return False
        return any(token in name for token in ("basecolor", "base_color", "diffuse", "albedo", "color_tex"))

    repaired = 0
    broken_material_paths: set[str] = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsActive():
            continue
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            if value.resolvedPath and Path(value.resolvedPath).is_file():
                continue
            if Path(value.path).is_absolute() and Path(value.path).is_file():
                continue
            replacement = find_asset(value.path)
            if replacement is not None:
                attr.Set(Sdf.AssetPath(str(replacement.resolve())))
                repaired += 1
                continue
            material = containing_material(prim)
            if material is not None and is_base_color_asset(attr.GetName()):
                broken_material_paths.add(str(material.GetPath()))

    fallback_materials: dict[str, Any] = {}
    fallback_bound_prims = 0
    fallback_root = f"{prim_path}/_robolab_fallback_materials"
    for gprim in Usd.PrimRange(root):
        if not gprim.IsActive() or not gprim.IsA(UsdGeom.Gprim):
            continue
        try:
            bound_material, _ = UsdShade.MaterialBindingAPI(gprim).ComputeBoundMaterial()
        except Exception:
            continue
        if not bound_material or str(bound_material.GetPath()) not in broken_material_paths:
            continue

        source_path = str(bound_material.GetPath())
        fallback = fallback_materials.get(source_path)
        if fallback is None:
            digest = hashlib.sha256(f"{seed}:{source_path}".encode("utf-8")).digest()
            color = _FALLBACK_MATERIAL_PALETTE[int.from_bytes(digest[:2], "big") % len(_FALLBACK_MATERIAL_PALETTE)]
            material_name = f"material_{len(fallback_materials):04d}"
            fallback = UsdShade.Material.Define(stage, f"{fallback_root}/{material_name}")
            shader = UsdShade.Shader.Define(stage, f"{fallback_root}/{material_name}/preview_surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.68)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            fallback.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            fallback_materials[source_path] = fallback
        UsdShade.MaterialBindingAPI.Apply(gprim).Bind(fallback)
        fallback_bound_prims += 1

    result = {
        "repaired_texture_paths": int(repaired),
        "unresolved_materials": int(len(broken_material_paths)),
        "fallback_materials": int(len(fallback_materials)),
        "fallback_bound_prims": int(fallback_bound_prims),
    }
    if repaired or fallback_materials:
        print(
            "[INFO] Background material repair: "
            f"textures={repaired}, fallback_materials={len(fallback_materials)}, "
            f"bound_prims={fallback_bound_prims}"
        )
    return result


def _clear_workspace_keepout(stage, prim_path: str) -> int:
    """Hide background meshes that intrude into the robot workspace.

    Isaac environment USDs are complete rooms/warehouses, so random surround
    placement can put shelves, forklifts, chairs, or props through the task table.
    Only compact, explicitly prop-like meshes may be hidden. Structural,
    unknown, or merged room meshes are preserved because hiding one such Gprim
    can remove an entire wall, floor, or room shell and create invalid depth.
    """
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return 0

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    hidden = 0
    protected = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
            continue
        if _is_interioragent_floor_mesh_path(str(prim.GetPath())):
            continue
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if bbox.IsEmpty():
            continue
        min_v = np.asarray(bbox.GetMin(), dtype=np.float64)
        max_v = np.asarray(bbox.GetMax(), dtype=np.float64)
        if np.any(max_v < WORKSPACE_KEEPOUT_MIN) or np.any(min_v > WORKSPACE_KEEPOUT_MAX):
            continue

        if not background_mesh_is_safe_to_hide(str(prim.GetPath()), min_v, max_v):
            protected += 1
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden += 1

    if protected:
        print(
            "[INFO] Preserved background structure in workspace keep-out: "
            f"protected={protected}, hidden_small_props={hidden}"
        )
    return hidden


def _sample_safe_surround_offset(
    stage,
    prim_path: str,
    rng,
    *,
    candidate_count: int,
    max_shift: float,
    room_margin: float = 0.20,
) -> dict[str, Any]:
    """Choose a varied room offset with minimal workspace intersections."""
    from pxr import Usd, UsdGeom

    root_bbox = _bbox_world(stage, prim_path)
    if root_bbox is None:
        return {
            "offset": [0.0, 0.0, 0.0],
            "candidate_count": 1,
            "best_candidate_count": 1,
            "workspace_intersections": 0,
            "reason": "background has no valid bounds",
        }

    root_min, root_max = root_bbox
    limit = max(float(max_shift), 0.0)
    allowed_x = (
        max(float(WORKSPACE_KEEPOUT_MAX[0] + room_margin - root_max[0]), -limit),
        min(float(WORKSPACE_KEEPOUT_MIN[0] - room_margin - root_min[0]), limit),
    )
    allowed_y = (
        max(float(WORKSPACE_KEEPOUT_MAX[1] + room_margin - root_max[1]), -limit),
        min(float(WORKSPACE_KEEPOUT_MIN[1] - room_margin - root_min[1]), limit),
    )
    if allowed_x[0] > allowed_x[1]:
        allowed_x = (0.0, 0.0)
    if allowed_y[0] > allowed_y[1]:
        allowed_y = (0.0, 0.0)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    geometry_bounds = []
    root = stage.GetPrimAtPath(prim_path)
    if root:
        for prim in Usd.PrimRange(root):
            if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                continue
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if bbox.IsEmpty():
                continue
            geometry_bounds.append(
                (
                    np.asarray(bbox.GetMin(), dtype=np.float64),
                    np.asarray(bbox.GetMax(), dtype=np.float64),
                )
            )

    count = max(int(candidate_count), 1)
    candidates = [np.zeros(3, dtype=np.float64)]
    for _ in range(count - 1):
        candidates.append(
            np.asarray(
                [rng.uniform(*allowed_x), rng.uniform(*allowed_y), 0.0],
                dtype=np.float64,
            )
        )

    scored = []
    for offset in candidates:
        intersections = 0
        overlap_volume = 0.0
        for bbox_min, bbox_max in geometry_bounds:
            shifted_min = bbox_min + offset
            shifted_max = bbox_max + offset
            overlap = np.minimum(shifted_max, WORKSPACE_KEEPOUT_MAX) - np.maximum(
                shifted_min,
                WORKSPACE_KEEPOUT_MIN,
            )
            if np.all(overlap >= 0.0):
                intersections += 1
                overlap_volume += float(np.prod(np.maximum(overlap, 1.0e-4)))
        scored.append((intersections, overlap_volume, offset))

    best_score = min((item[0], item[1]) for item in scored)
    best = [item for item in scored if (item[0], item[1]) == best_score]
    chosen = best[int(rng.integers(0, len(best)))]
    return {
        "offset": chosen[2].round(6).tolist(),
        "candidate_count": int(len(candidates)),
        "best_candidate_count": int(len(best)),
        "workspace_intersections": int(chosen[0]),
        "workspace_overlap_volume_m3": float(chosen[1]),
        "allowed_x_range": [float(v) for v in allowed_x],
        "allowed_y_range": [float(v) for v in allowed_y],
    }


def _segment_intersects_aabb(start, end, bbox_min, bbox_max) -> bool:
    """Return whether a finite line segment intersects an axis-aligned box."""
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1.0e-9:
            if start[axis] < bbox_min[axis] or start[axis] > bbox_max[axis]:
                return False
            continue
        inv = 1.0 / float(direction[axis])
        near = float((bbox_min[axis] - start[axis]) * inv)
        far = float((bbox_max[axis] - start[axis]) * inv)
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return False
    return True


def clear_background_visibility_corridors(
    background_entries: list[dict[str, Any]],
    camera_centers,
    targets,
    *,
    clearance: float = 0.08,
    sample_count: int = 32,
    stage=None,
) -> dict[str, Any]:
    """Hide loaded background meshes blocking camera-to-workspace segments.

    Only geometry under USD asset background roots is considered. Visibility is
    authored once before recording, so this safety pass cannot cause temporal
    flicker. AABBs are deliberately conservative, but only compact named props
    may be hidden; structural, unknown, and merged meshes are preserved so a
    coarse AABB cannot erase a room surface and invalidate depth.
    """
    from pxr import Usd, UsdGeom

    centers = np.asarray(camera_centers, dtype=np.float64)
    target_points = np.asarray(targets, dtype=np.float64)
    result = {
        "supported": False,
        "sampled_poses": 0,
        "target_points": int(len(target_points)) if target_points.ndim == 2 else 0,
        "tested_prims": 0,
        "hidden_occluder_prims": 0,
        "protected_occluder_prims": 0,
        "clearance_m": float(clearance),
    }
    if centers.ndim != 2 or centers.shape[1:] != (3,) or len(centers) == 0:
        result["reason"] = "camera trajectory has no valid centers"
        return result
    if target_points.ndim != 2 or target_points.shape[1:] != (3,) or len(target_points) == 0:
        result["reason"] = "visibility check has no valid targets"
        return result

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    count = min(max(int(sample_count), 2), len(centers))
    sample_indices = np.unique(np.linspace(0, len(centers) - 1, count).round().astype(int))
    sampled_centers = centers[sample_indices]
    padding = max(float(clearance), 0.0)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    hidden_paths = []
    protected_paths = []
    tested = 0

    for entry in background_entries:
        if entry.get("mode") != "usd_asset" or not entry.get("path"):
            continue
        root = stage.GetPrimAtPath(entry["path"])
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue
            if _is_interioragent_floor_mesh_path(str(prim.GetPath())):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible":
                continue
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if bbox.IsEmpty():
                continue
            tested += 1
            bbox_min = np.asarray(bbox.GetMin(), dtype=np.float64) - padding
            bbox_max = np.asarray(bbox.GetMax(), dtype=np.float64) + padding
            if any(
                _segment_intersects_aabb(center, target, bbox_min, bbox_max)
                for center in sampled_centers
                for target in target_points
            ):
                if not background_mesh_is_safe_to_hide(
                    str(prim.GetPath()), bbox_min + padding, bbox_max - padding
                ):
                    protected_paths.append(str(prim.GetPath()))
                    continue
                imageable.MakeInvisible()
                hidden_paths.append(str(prim.GetPath()))

    result.update(
        {
            "supported": True,
            "sampled_poses": int(len(sample_indices)),
            "sample_indices": sample_indices.tolist(),
            "target_points": int(len(target_points)),
            "tested_prims": int(tested),
            "hidden_occluder_prims": int(len(hidden_paths)),
            "hidden_occluder_paths": hidden_paths,
            "protected_occluder_prims": int(len(protected_paths)),
            "protected_occluder_paths": protected_paths,
        }
    )
    return result


def _define_reference_background(
    stage,
    style: str,
    preset: BackgroundPreset,
    seed: int,
    scale_multiplier: float,
    root_path: str,
    placement: str,
    clear_workspace: bool,
    randomize_surround_position: bool,
    surround_position_candidates: int,
    surround_max_shift: float,
) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    rng = np.random.default_rng(seed)
    asset_path = _resolve_asset_path(preset.asset)
    style_name = _clean_name(style)
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetMetadata("comment", "Static geometry backgrounds loaded by robolab.core.scenes.load_background")

    prim_path = f"{root_path}/{style_name}"
    parent = UsdGeom.Xform.Define(stage, prim_path)
    parent_prim = parent.GetPrim()
    translate_op = parent.AddTranslateOp()
    if placement == "surround":
        yaw_deg = preset.yaw_deg + float(rng.choice([0.0, 90.0, 180.0, 270.0])) + float(rng.uniform(-8.0, 8.0))
    else:
        yaw_deg = preset.yaw_deg + float(rng.uniform(-4.0, 4.0))
    parent.AddRotateZOp().Set(yaw_deg)
    parent.AddScaleOp().Set(Gf.Vec3f(*(float(max(scale_multiplier * preset.scale, 0.05)),) * 3))

    asset_child_path = f"{prim_path}/asset"
    asset_prim = UsdGeom.Xform.Define(stage, asset_child_path).GetPrim()
    asset_prim.GetReferences().AddReference(asset_path)
    deinstanced_count = _deinstance_subtree(stage, asset_child_path)
    double_sided_floor_meshes = 0
    if style.startswith("interioragent_"):
        double_sided_floor_meshes = _make_interioragent_floors_double_sided(stage, asset_child_path)
    material_repair = _repair_background_material_assets(stage, asset_child_path, asset_path, seed)
    hidden_light_prims = _hide_primary_ray_lights(stage, asset_child_path)
    stripped_physics_apis = _strip_physics_subtree(stage, asset_child_path)

    # Initial placement. In ``behind`` mode the asset stays behind the robot
    # workspace. In ``surround`` mode the operation area is placed inside the
    # USD environment so visible background is real scene geometry, not an HDRI.
    workspace_center = np.asarray([0.35, preset.y_center, preset.floor_z], dtype=np.float64)
    current = np.asarray([preset.safe_max_x - 0.55, preset.y_center, preset.floor_z], dtype=np.float64)
    translate_op.Set(Gf.Vec3d(*current.tolist()))

    bbox = _bbox_world(stage, prim_path)
    if bbox is not None:
        min_v, max_v = bbox
        center = (min_v + max_v) * 0.5
        delta = np.zeros(3, dtype=np.float64)
        if placement == "surround":
            delta[0] = workspace_center[0] - center[0]
            delta[1] = workspace_center[1] - center[1]
        else:
            if max_v[0] > preset.safe_max_x:
                delta[0] = preset.safe_max_x - max_v[0]
            delta[1] = preset.y_center - center[1]
        delta[2] = preset.floor_z - min_v[2]
        current = current + delta
        translate_op.Set(Gf.Vec3d(*current.tolist()))
        bbox = _bbox_world(stage, prim_path)

    surround_position = {
        "offset": [0.0, 0.0, 0.0],
        "candidate_count": 1,
        "best_candidate_count": 1,
        "workspace_intersections": 0,
        "reason": "position randomization disabled",
    }
    if placement == "surround" and randomize_surround_position:
        surround_position = _sample_safe_surround_offset(
            stage,
            asset_child_path,
            rng,
            candidate_count=surround_position_candidates,
            max_shift=surround_max_shift,
        )
        position_offset = np.asarray(surround_position["offset"], dtype=np.float64)
        current = current + position_offset
        translate_op.Set(Gf.Vec3d(*current.tolist()))

    hidden_keepout_prims = 0
    if placement == "surround" and clear_workspace:
        hidden_keepout_prims = _clear_workspace_keepout(stage, asset_child_path)
        bbox = _bbox_world(stage, prim_path)

    result = {
        "mode": "usd_asset",
        "style": style,
        "path": prim_path,
        "asset": asset_path,
        "translation": current.round(6).tolist(),
        "yaw_deg": float(yaw_deg),
        "placement": placement,
        "workspace_keepout_enabled": bool(clear_workspace),
        "surround_position_randomization": bool(randomize_surround_position),
        "surround_position_selection": surround_position,
        "workspace_center": workspace_center.round(6).tolist(),
        "scale": float(scale_multiplier * preset.scale),
        "safe_max_x": preset.safe_max_x,
        "floor_z": preset.floor_z,
        "deinstanced_prims": int(deinstanced_count),
        "double_sided_floor_meshes": int(double_sided_floor_meshes),
        "material_repair": material_repair,
        "hidden_light_prims": int(hidden_light_prims),
        "stripped_physics_apis": int(stripped_physics_apis),
        "hidden_workspace_keepout_prims": int(hidden_keepout_prims),
    }
    if bbox is not None:
        min_v, max_v = bbox
        result["bbox_min"] = min_v.round(6).tolist()
        result["bbox_max"] = max_v.round(6).tolist()
    return result


def load_background(
    style: str | None,
    *,
    seed: int = 0,
    scale: float = 1.0,
    placement: str = "behind",
    clear_workspace: bool = True,
    randomize_surround_position: bool = True,
    surround_position_candidates: int = 64,
    surround_max_shift: float = 2.0,
    root_path: str = "/World/envs/env_0/scene/asset_backgrounds",
) -> list[dict[str, Any]]:
    """Load a static USD geometry background into the active USD stage.

    Args:
        style: Background style name. Recognized values are keys in
            ``ASSET_BACKGROUND_PRESETS``. ``asset_random`` selects one preset
            deterministically from ``seed``. ``asset:/abs/or/relative.usd`` loads
            a specific asset path.
        seed: Random seed for deterministic asset selection and small yaw jitter.
        scale: Extra scale multiplier applied on top of the preset scale.
        placement: ``behind`` keeps the asset behind the workspace; ``surround``
            centers the workspace inside the USD environment.
        clear_workspace: Hide any loaded background geometry intersecting the
            robot/table keep-out volume when using ``surround`` placement.
        randomize_surround_position: Vary where the fixed robot workspace appears
            inside the room while keeping task and policy coordinates unchanged.
        surround_position_candidates: Number of deterministic seeded placement
            candidates scored for workspace intersections.
        surround_max_shift: Maximum room translation in x/y from centered placement.
        root_path: USD prim path where backgrounds are inserted.

    Returns:
        A metadata list. Empty list means the style is not handled by this module
        and callers may fall back to procedural backgrounds.
    """
    if not style or style in {"none", "off"}:
        return []
    if placement not in {"behind", "surround"}:
        raise ValueError(f"Unsupported background placement: {placement}")

    if style == "asset_random":
        rng = np.random.default_rng(seed)
        style = str(rng.choice(ASSET_BACKGROUND_STYLE_NAMES))

    if style.startswith("asset:"):
        asset = style.split(":", 1)[1]
        preset = BackgroundPreset(asset=asset, scale=1.0)
    elif style in ASSET_BACKGROUND_PRESETS:
        preset = ASSET_BACKGROUND_PRESETS[style]
    else:
        return []

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    created = _define_reference_background(
        stage,
        style,
        preset,
        seed,
        scale,
        root_path,
        placement,
        clear_workspace,
        randomize_surround_position,
        surround_position_candidates,
        surround_max_shift,
    )
    if "bbox_min" not in created or "bbox_max" not in created:
        raise RuntimeError(
            "Background USD reference produced no renderable geometry bounds: "
            f"style={style} asset={created.get('asset')}"
        )
    extent = np.asarray(created["bbox_max"], dtype=np.float64) - np.asarray(
        created["bbox_min"], dtype=np.float64
    )
    if not np.all(np.isfinite(extent)) or np.any(extent <= 1.0e-4):
        raise RuntimeError(
            "Background USD reference produced invalid geometry bounds: "
            f"style={style} extent={extent.tolist()}"
        )
    print(f"[INFO] Loaded USD asset background: {created['style']} asset={created['asset']}")
    return [created]


def collect_background_gprim_bounds(
    prim_paths: list[str],
    *,
    workspace_center: tuple[float, float, float] = (0.45, 0.0, 0.75),
    max_obstacles: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect leaf render-geometry world AABBs for camera safety checks.

    A single AABB for a concave room encloses its empty interior and would mark
    every in-room camera as colliding. This intentionally returns one bound per
    Gprim and skips aggregate meshes whose box encloses the manipulation center.
    """
    from pxr import Usd, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True
    )
    center = np.asarray(workspace_center, dtype=np.float64)
    candidates: list[tuple[float, dict[str, Any]]] = []
    seen: set[tuple[float, ...]] = set()
    skipped_enclosing = 0
    skipped_invalid = 0
    for root_path in prim_paths:
        root = stage.GetPrimAtPath(str(root_path))
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue
            try:
                imageable = UsdGeom.Imageable(prim)
                if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue
                aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                if aligned.IsEmpty():
                    continue
                bounds_min = np.asarray(aligned.GetMin(), dtype=np.float64)
                bounds_max = np.asarray(aligned.GetMax(), dtype=np.float64)
            except Exception:
                skipped_invalid += 1
                continue
            extent = bounds_max - bounds_min
            if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)) or np.any(extent <= 1.0e-5):
                skipped_invalid += 1
                continue
            encloses_center = bool(np.all(bounds_min < center) and np.all(bounds_max > center))
            if encloses_center and int(np.sum(extent > 1.0)) >= 2:
                skipped_enclosing += 1
                continue
            key = tuple(np.round(np.concatenate((bounds_min, bounds_max)), 4))
            if key in seen:
                continue
            seen.add(key)
            closest = np.clip(center, bounds_min, bounds_max)
            distance_to_workspace = float(np.linalg.norm(center - closest))
            candidates.append((distance_to_workspace, {
                "name": str(prim.GetPath()),
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "kind": "scene",
            }))
    candidates.sort(key=lambda item: item[0])
    obstacles = [item[1] for item in candidates[:max(1, int(max_obstacles))]]
    diagnostics = {
        "root_prim_count": len(prim_paths),
        "gprim_obstacle_count": len(obstacles),
        "skipped_enclosing_aggregate": skipped_enclosing,
        "skipped_invalid": skipped_invalid,
        "candidate_gprim_count": len(candidates),
        "selection": "nearest_to_workspace",
        "truncated": len(candidates) > max(1, int(max_obstacles)),
    }
    return obstacles, diagnostics
