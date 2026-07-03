from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .sampling import make_positive


def pixel_to_linear(xy: np.ndarray, width: int) -> np.ndarray:
    return xy[:, 0].astype(np.int64) + width * xy[:, 1].astype(np.int64)


def pair_key(corres1: np.ndarray, corres2: np.ndarray, width1: int, width2: int, height2: int) -> np.ndarray:
    return pixel_to_linear(corres1, width1) * np.int64(width2 * height2) + pixel_to_linear(corres2, width2)


def world_points(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float64)
    z = depth.astype(np.float64)
    xyz = np.stack(
        (
            (x - intrinsics[0, 2]) / intrinsics[0, 0] * z,
            (y - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
            np.ones_like(z),
        ),
        axis=-1,
    )
    points = (pose.astype(np.float64) @ xyz.reshape(-1, 4).T).T[:, :3]
    return points.reshape(height, width, 3).astype(np.float32)


def project_world(points_world: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = points_world.reshape(-1, 3).astype(np.float64)
    homog = np.concatenate((flat, np.ones((len(flat), 1), dtype=np.float64)), axis=1)
    cam = (np.linalg.inv(pose.astype(np.float64)) @ homog.T).T[:, :3]
    z = cam[:, 2]
    xy = np.empty((len(cam), 2), dtype=np.float64)
    xy[:, 0] = intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2]
    xy[:, 1] = intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]
    return xy.reshape(*points_world.shape[:2], 2), z.reshape(points_world.shape[:2])


def geometry_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth1 = np.asarray(view1["depthmap"], dtype=np.float32)
    depth2 = np.asarray(view2["depthmap"], dtype=np.float32)
    k1 = np.asarray(view1["camera_intrinsics"], dtype=np.float32)
    k2 = np.asarray(view2["camera_intrinsics"], dtype=np.float32)
    pose1 = np.asarray(view1["camera_pose"], dtype=np.float32)
    pose2 = np.asarray(view2["camera_pose"], dtype=np.float32)
    h1, w1 = depth1.shape
    h2, w2 = depth2.shape

    points1 = world_points(depth1, k1, pose1)
    xy2_float, z2_projected = project_world(points1, k2, pose2)
    y1, x1 = np.indices((h1, w1), dtype=np.int64)
    source_xy = np.stack((x1.reshape(-1), y1.reshape(-1)), axis=1)
    target_xy = np.rint(xy2_float.reshape(-1, 2)).astype(np.int64)
    z2_projected = z2_projected.reshape(-1)

    inside = (target_xy[:, 0] >= 0) & (target_xy[:, 0] < w2) & (target_xy[:, 1] >= 0) & (target_xy[:, 1] < h2)
    source_depth = depth1.reshape(-1)
    safe_x2 = np.clip(target_xy[:, 0], 0, w2 - 1)
    safe_y2 = np.clip(target_xy[:, 1], 0, h2 - 1)
    target_depth = depth2[safe_y2, safe_x2]
    depth_error = np.abs(z2_projected - target_depth)
    keep = (
        inside
        & np.isfinite(source_depth)
        & np.isfinite(target_depth)
        & np.isfinite(z2_projected)
        & (source_depth > args.min_depth)
        & (target_depth > args.min_depth)
        & (z2_projected > args.min_depth)
        & (source_depth <= args.max_depth)
        & (target_depth <= args.max_depth)
        & (z2_projected <= args.max_depth)
        & (depth_error <= args.depth_consistency_thresh)
    )
    positives = make_positive(source_xy[keep], target_xy[keep], depth_error[keep], "geometry", depth_error=depth_error[keep])
    return positives, {"raw": int(len(source_xy)), "after_filter": int(keep.sum())}

