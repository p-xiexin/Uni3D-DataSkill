from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .sampling import PairSkip, make_positive


def to_numpy(values):
    if isinstance(values, tuple):
        return tuple(to_numpy(value) for value in values)
    if isinstance(values, list):
        return [to_numpy(value) for value in values]
    if isinstance(values, dict):
        return {key: to_numpy(value) for key, value in values.items()}
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return values


def pixel_to_linear(xy: np.ndarray, width: int) -> np.ndarray:
    return xy[:, 0].astype(np.int64) + width * xy[:, 1].astype(np.int64)


def pair_key(corres1: np.ndarray, corres2: np.ndarray, width1: int, width2: int, height2: int) -> np.ndarray:
    return pixel_to_linear(corres1, width1) * np.int64(width2 * height2) + pixel_to_linear(corres2, width2)


def project_pixels_between_views_numpy(
    source_xy: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    pose1_c2w: np.ndarray,
    pose2_c2w: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source_xy_float = np.asarray(source_xy, dtype=np.float64)
    source_xy_round = np.rint(source_xy_float).astype(np.int64)
    h1, w1 = depth1.shape
    h2, w2 = depth2.shape

    source_inside = (
        (source_xy_round[:, 0] >= 0)
        & (source_xy_round[:, 0] < w1)
        & (source_xy_round[:, 1] >= 0)
        & (source_xy_round[:, 1] < h1)
    )
    safe_x1 = np.clip(source_xy_round[:, 0], 0, w1 - 1)
    safe_y1 = np.clip(source_xy_round[:, 1], 0, h1 - 1)
    source_depth = depth1[safe_y1, safe_x1].astype(np.float64)

    x_cam1 = (source_xy_float[:, 0] - k1[0, 2]) / k1[0, 0] * source_depth
    y_cam1 = (source_xy_float[:, 1] - k1[1, 2]) / k1[1, 1] * source_depth
    cam1_h = np.stack((x_cam1, y_cam1, source_depth, np.ones_like(source_depth)), axis=1)
    world_h = (pose1_c2w.astype(np.float64) @ cam1_h.T).T
    cam2 = (np.linalg.inv(pose2_c2w.astype(np.float64)) @ world_h.T).T[:, :3]
    projected_depth = cam2[:, 2]

    target_xy_float = np.empty((len(source_xy_float), 2), dtype=np.float64)
    target_xy_float[:, 0] = k2[0, 0] * cam2[:, 0] / projected_depth + k2[0, 2]
    target_xy_float[:, 1] = k2[1, 1] * cam2[:, 1] / projected_depth + k2[1, 2]
    target_xy_round = np.rint(target_xy_float).astype(np.int64)

    target_inside = (
        (target_xy_round[:, 0] >= 0)
        & (target_xy_round[:, 0] < w2)
        & (target_xy_round[:, 1] >= 0)
        & (target_xy_round[:, 1] < h2)
    )
    safe_x2 = np.clip(target_xy_round[:, 0], 0, w2 - 1)
    safe_y2 = np.clip(target_xy_round[:, 1], 0, h2 - 1)
    target_depth = depth2[safe_y2, safe_x2].astype(np.float64)
    depth_error = np.abs(projected_depth - target_depth)

    source_depth_valid = (
        source_inside
        & np.isfinite(source_depth)
        & (source_depth > args.min_depth)
        & (source_depth <= args.max_depth)
    )
    keep = (
        source_depth_valid
        & np.isfinite(cam2).all(axis=1)
        & np.isfinite(target_xy_float).all(axis=1)
        & np.isfinite(projected_depth)
        & (projected_depth > args.min_depth)
        & (projected_depth <= args.max_depth)
        & target_inside
        & np.isfinite(target_depth)
        & (target_depth > args.min_depth)
        & (target_depth <= args.max_depth)
        & np.isfinite(depth_error)
        & (depth_error <= args.depth_consistency_thresh)
    )
    stats = {
        "source_valid_depth": int(source_depth_valid.sum()),
        "target_inside": int((source_depth_valid & target_inside).sum()),
        "after_filter": int(keep.sum()),
    }
    return source_xy_round, target_xy_round, depth_error, keep, stats


def geometry_positives_numpy(
    depth1: np.ndarray,
    depth2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    pose1: np.ndarray,
    pose2: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    h1, w1 = depth1.shape
    y1, x1 = np.indices((h1, w1), dtype=np.int64)
    if args.geometry_stride > 1:
        y1 = y1[:: args.geometry_stride, :: args.geometry_stride]
        x1 = x1[:: args.geometry_stride, :: args.geometry_stride]

    source_xy = np.stack((x1.reshape(-1), y1.reshape(-1)), axis=1)
    source_xy, target_xy, depth_error, keep, stats = project_pixels_between_views_numpy(source_xy, depth1, depth2, k1, k2, pose1, pose2, args)
    positives = make_positive(source_xy[keep], target_xy[keep], depth_error[keep], "geometry", depth_error=depth_error[keep])
    stats["raw"] = int(len(source_xy))
    return positives, stats


def geometry_positives_torch(
    depth1: np.ndarray,
    depth2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    pose1: np.ndarray,
    pose2: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise PairSkip("torch_required_for_geometry_device") from exc

    device = torch.device(args.geometry_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairSkip(f"cuda_unavailable_for_geometry_device:{args.geometry_device}")

    depth1_t = torch.as_tensor(depth1, dtype=torch.float32, device=device)
    depth2_t = torch.as_tensor(depth2, dtype=torch.float32, device=device)
    k1_t = torch.as_tensor(k1, dtype=torch.float32, device=device)
    k2_t = torch.as_tensor(k2, dtype=torch.float32, device=device)
    pose1_t = torch.as_tensor(pose1, dtype=torch.float32, device=device)
    pose2_inv_t = torch.linalg.inv(torch.as_tensor(pose2, dtype=torch.float32, device=device))

    h1, w1 = depth1_t.shape
    h2, w2 = depth2_t.shape
    ys = torch.arange(0, h1, args.geometry_stride, dtype=torch.float32, device=device)
    xs = torch.arange(0, w1, args.geometry_stride, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    source_depth = depth1_t[y_grid.long(), x_grid.long()].reshape(-1)
    source_x = x_grid.reshape(-1)
    source_y = y_grid.reshape(-1)

    x_cam = (source_x - k1_t[0, 2]) / k1_t[0, 0] * source_depth
    y_cam = (source_y - k1_t[1, 2]) / k1_t[1, 1] * source_depth
    ones = torch.ones_like(source_depth)
    cam1_h = torch.stack((x_cam, y_cam, source_depth, ones), dim=1)
    world_h = (pose1_t @ cam1_h.T).T
    cam2 = (pose2_inv_t @ world_h.T).T[:, :3]
    z2_projected = cam2[:, 2]
    x2 = k2_t[0, 0] * cam2[:, 0] / z2_projected + k2_t[0, 2]
    y2 = k2_t[1, 1] * cam2[:, 1] / z2_projected + k2_t[1, 2]
    target_x = torch.round(x2).long()
    target_y = torch.round(y2).long()

    inside = (target_x >= 0) & (target_x < w2) & (target_y >= 0) & (target_y < h2)
    safe_x2 = target_x.clamp(0, w2 - 1)
    safe_y2 = target_y.clamp(0, h2 - 1)
    target_depth = depth2_t[safe_y2, safe_x2]
    depth_error = torch.abs(z2_projected - target_depth)
    keep = (
        inside
        & torch.isfinite(source_depth)
        & torch.isfinite(target_depth)
        & torch.isfinite(z2_projected)
        & (source_depth > args.min_depth)
        & (target_depth > args.min_depth)
        & (z2_projected > args.min_depth)
        & (source_depth <= args.max_depth)
        & (target_depth <= args.max_depth)
        & (z2_projected <= args.max_depth)
        & (depth_error <= args.depth_consistency_thresh)
    )

    source_xy = torch.stack((source_x, source_y), dim=1).round().long()
    target_xy = torch.stack((target_x, target_y), dim=1)
    positives = make_positive(
        source_xy[keep].detach().cpu().numpy(),
        target_xy[keep].detach().cpu().numpy(),
        depth_error[keep].detach().cpu().numpy(),
        "geometry",
        depth_error=depth_error[keep].detach().cpu().numpy(),
    )
    return positives, {"raw": int(source_xy.shape[0]), "after_filter": int(keep.sum().item()), "device": str(device)}


def geometry_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    view1, view2 = to_numpy((view1, view2))
    depth1 = np.asarray(view1["depthmap"], dtype=np.float32)
    depth2 = np.asarray(view2["depthmap"], dtype=np.float32)
    k1 = np.asarray(view1["camera_intrinsics"], dtype=np.float32)
    k2 = np.asarray(view2["camera_intrinsics"], dtype=np.float32)
    pose1 = np.asarray(view1["camera_pose"], dtype=np.float32)
    pose2 = np.asarray(view2["camera_pose"], dtype=np.float32)
    if args.geometry_device == "cpu":
        positives, stats = geometry_positives_numpy(depth1, depth2, k1, k2, pose1, pose2, args)
        stats["device"] = "cpu"
        return positives, stats
    return geometry_positives_torch(depth1, depth2, k1, k2, pose1, pose2, args)
