"""Dense 2D optical-flow conversion utilities for stereo datasets."""

from __future__ import annotations

import json

import cv2
import numpy as np


def compose_backward_motion_vectors(fields: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compose Isaac current-to-previous pixel maps over rendered frames."""
    if not fields:
        raise ValueError("At least one motion-vector field is required")
    arrays = [np.asarray(field, dtype=np.float32) for field in fields]
    if any(field.ndim != 3 or field.shape[-1] != 2 for field in arrays):
        raise ValueError("Motion-vector fields must have shape [H, W, 2]")
    if any(field.shape != arrays[0].shape for field in arrays[1:]):
        raise ValueError("All motion-vector fields must have the same shape")

    height, width = arrays[0].shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x.copy()
    map_y = grid_y.copy()
    valid = np.ones((height, width), dtype=bool)
    for field in reversed(arrays):
        sampled_x = cv2.remap(
            field[..., 0], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_y = cv2.remap(
            field[..., 1], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        valid &= np.isfinite(sampled_x) & np.isfinite(sampled_y)
        map_x += np.nan_to_num(sampled_x)
        map_y += np.nan_to_num(sampled_y)
        valid &= (map_x >= 0.0) & (map_x <= width - 1.0)
        valid &= (map_y >= 0.0) & (map_y <= height - 1.0)

    backward = np.stack((map_x - grid_x, map_y - grid_y), axis=-1).astype(np.float32)
    backward[~valid] = 0.0
    return backward, valid


def backward_to_forward_flow(
    backward: np.ndarray, backward_valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a current-to-previous map by splatting into the previous frame."""
    backward = np.asarray(backward, dtype=np.float32)
    if backward.ndim != 3 or backward.shape[-1] != 2:
        raise ValueError("backward must have shape [H, W, 2]")
    height, width = backward.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    previous_ix = np.rint(grid_x + backward[..., 0]).astype(np.int32)
    previous_iy = np.rint(grid_y + backward[..., 1]).astype(np.int32)
    valid = np.isfinite(backward).all(axis=-1)
    valid &= (previous_ix >= 0) & (previous_ix < width)
    valid &= (previous_iy >= 0) & (previous_iy < height)
    if backward_valid is not None:
        valid &= np.asarray(backward_valid, dtype=bool)

    source_x = grid_x[valid]
    source_y = grid_y[valid]
    target_x = previous_ix[valid]
    target_y = previous_iy[valid]
    forward = np.zeros_like(backward)
    forward_valid = np.zeros((height, width), dtype=bool)
    forward[target_y, target_x, 0] = source_x - target_x
    forward[target_y, target_x, 1] = source_y - target_y
    forward_valid[target_y, target_x] = True
    return forward, forward_valid


def sanitize_native_forward_flow(
    forward: np.ndarray, depth_source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-grid forward flow and its specification-level invalid mask.

    Endpoint visibility is deliberately not tested: out-of-frame and occluded
    correspondences remain valid when the renderer provides a finite vector.
    """
    flow = np.asarray(forward, dtype=np.float32)
    depth = np.asarray(depth_source, dtype=np.float32).squeeze()
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("forward must have shape [H, W, 2]")
    if depth.shape != flow.shape[:2]:
        raise ValueError("depth_source must have shape [H, W]")

    # RTX motion vectors use the exact vector (-1, -1) when no temporal
    # correspondence is available (commonly at disocclusion/image borders).
    # It is finite, so a finite-only check would silently turn the sentinel
    # into a plausible 1.414-pixel flow vector.
    native_invalid = np.all(flow == np.float32(-1.0), axis=-1)
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(flow).all(axis=-1)
        & (~native_invalid)
    )
    clean = np.where(valid[..., None], flow, 0.0).astype(np.float32)
    return clean, ~valid


def instance_id_path_mask(
    instance_ids: np.ndarray,
    annotator_info: dict,
    root_paths: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Select instance pixels whose reported USD path is below a root path.

    Replicator's uncolorized ``instance_id_segmentation_fast`` annotator
    returns integer IDs plus an ``idToLabels`` mapping.  Path-based selection
    is more reliable for referenced USD backgrounds than inherited semantic
    labels, which can be lost on nested renderable prims.
    """
    ids = np.asarray(instance_ids)
    if ids.ndim == 3 and ids.shape[-1] == 1:
        ids = ids[..., 0]
    if ids.ndim != 2:
        raise ValueError("instance_ids must have shape [H, W] or [H, W, 1]")
    roots = tuple(str(path).rstrip("/") for path in root_paths if path)
    if not roots:
        return np.zeros(ids.shape, dtype=bool)

    mapping = (annotator_info or {}).get("idToLabels", {})
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError as exc:
            raise ValueError("instance annotator idToLabels is not valid JSON") from exc
    if not isinstance(mapping, dict):
        raise ValueError("instance annotator info must contain an idToLabels mapping")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from strings(child)

    selected_ids = []
    for raw_id, labels in mapping.items():
        paths = tuple(value for value in strings(labels) if value.startswith("/"))
        if not paths or not all(
            any(path == root or path.startswith(root + "/") for root in roots)
            for path in paths
        ):
            continue
        try:
            selected_ids.append(int(raw_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid instance id in idToLabels: {raw_id!r}") from exc

    if not selected_ids:
        return np.zeros(ids.shape, dtype=bool)
    return np.isin(ids, np.asarray(selected_ids, dtype=ids.dtype))


def static_camera_forward_flow(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    camera_to_world_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project static source pixels into the next camera without visibility tests.

    Out-of-frame endpoints remain valid as required by the dataset contract.
    Points that have invalid source depth or lie behind the next camera are
    marked invalid.
    """
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    c2w_next = np.asarray(camera_to_world_next, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if k.shape != (3, 3) or c2w.shape != (4, 4) or c2w_next.shape != (4, 4):
        raise ValueError("Expected intrinsics [3,3] and camera_to_world poses [4,4]")

    height, width = depth.shape
    source_valid = np.isfinite(depth) & (depth > 0.0)
    projection_depth = np.where(source_valid, depth, 0.0)
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    points_camera = np.stack(
        (
            (x - k[0, 2]) * projection_depth / k[0, 0],
            (y - k[1, 2]) * projection_depth / k[1, 1],
            projection_depth,
        ),
        axis=-1,
    )
    points_world = points_camera @ c2w[:3, :3].T + c2w[:3, 3]
    points_next = (points_world - c2w_next[:3, 3]) @ c2w_next[:3, :3]
    z_next = points_next[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        next_x = k[0, 0] * points_next[..., 0] / z_next + k[0, 2]
        next_y = k[1, 1] * points_next[..., 1] / z_next + k[1, 2]
    flow = np.stack((next_x - x, next_y - y), axis=-1).astype(np.float32)
    valid = source_valid
    valid &= np.isfinite(flow).all(axis=-1) & np.isfinite(z_next) & (z_next > 0.0)
    flow[~valid] = 0.0
    return flow, ~valid


def remove_camera_motion(
    depth: np.ndarray,
    observed_flow: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    camera_to_world_next: np.ndarray,
    flow_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract optical flow caused by camera motion under a static-scene model."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    observed = np.asarray(observed_flow, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if observed.shape != depth.shape + (2,):
        raise ValueError("observed_flow must have shape [H, W, 2]")

    height, width = depth.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    k = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    c2w_next = np.asarray(camera_to_world_next, dtype=np.float64)
    if k.shape != (3, 3) or c2w.shape != (4, 4) or c2w_next.shape != (4, 4):
        raise ValueError("Expected intrinsics [3,3] and camera_to_world poses [4,4]")

    points_camera = np.stack(
        (
            (x - k[0, 2]) * depth / k[0, 0],
            (y - k[1, 2]) * depth / k[1, 1],
            depth,
        ),
        axis=-1,
    )
    points_world = points_camera @ c2w[:3, :3].T + c2w[:3, 3]
    points_next = (points_world - c2w_next[:3, 3]) @ c2w_next[:3, :3]
    z_next = points_next[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        next_x = k[0, 0] * points_next[..., 0] / z_next + k[0, 2]
        next_y = k[1, 1] * points_next[..., 1] / z_next + k[1, 2]

    camera_flow = np.stack((next_x - x, next_y - y), axis=-1).astype(np.float32)
    observed_x = x + observed[..., 0]
    observed_y = y + observed[..., 1]
    valid = np.isfinite(depth) & (depth > 0.0)
    valid &= np.isfinite(observed).all(axis=-1)
    valid &= np.isfinite(camera_flow).all(axis=-1) & (z_next > 0.0)
    projection_epsilon = 1.0e-3
    valid &= (next_x >= -projection_epsilon) & (next_x <= width - 1.0 + projection_epsilon)
    valid &= (next_y >= -projection_epsilon) & (next_y <= height - 1.0 + projection_epsilon)
    valid &= (observed_x >= 0.0) & (observed_x <= width - 1.0)
    valid &= (observed_y >= 0.0) & (observed_y <= height - 1.0)
    if flow_valid is not None:
        valid &= np.asarray(flow_valid, dtype=bool)

    residual = (observed - camera_flow).astype(np.float32)
    camera_flow[~valid] = 0.0
    residual[~valid] = 0.0
    return residual, camera_flow, valid
