from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .cropping import extract_correspondences_from_pts3d, to_numpy
from .sampling import make_positive


def camera_points_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 0] = (x - intrinsics[0, 2]) / intrinsics[0, 0] * z
    points[..., 1] = (y - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points[..., 2] = z
    return points


def world_points_from_depth(depth: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    camera_points = camera_points_from_depth(depth, intrinsics).reshape(-1, 3)
    homogeneous = np.concatenate((camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)), axis=1)
    world_points = (camera_pose.astype(np.float64) @ homogeneous.T).T[:, :3]
    return world_points.astype(np.float32).reshape(height, width, 3)


def camera_points_from_world(world_points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    height, width, _ = world_points.shape
    flat = world_points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((flat.shape[0], 1), dtype=np.float32)), axis=1)
    camera_points = (np.linalg.inv(camera_pose.astype(np.float64)) @ homogeneous.T).T[:, :3]
    return camera_points.astype(np.float32).reshape(height, width, 3)


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
    return source_xy_float.astype(np.float32), target_xy_float.astype(np.float32), depth_error, keep, stats


def geometry_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    view1, view2 = to_numpy((view1, view2))
    depth1 = np.asarray(view1["depthmap"], dtype=np.float32)
    depth2 = np.asarray(view2["depthmap"], dtype=np.float32)
    k1 = np.asarray(view1["camera_intrinsics"], dtype=np.float32)
    k2 = np.asarray(view2["camera_intrinsics"], dtype=np.float32)
    pose1 = np.asarray(view1["camera_pose"], dtype=np.float32)
    pose2 = np.asarray(view2["camera_pose"], dtype=np.float32)

    pts1_world = world_points_from_depth(depth1, k1, pose1)
    pts2_world = world_points_from_depth(depth2, k2, pose2)
    crop_view1 = {"pts3d": pts1_world, "camera_intrinsics": k1, "camera_pose": pose1}
    crop_view2 = {"pts3d": pts2_world, "camera_intrinsics": k2, "camera_pose": pose2}
    source_xy, target_xy = extract_correspondences_from_pts3d(crop_view1, crop_view2, target_n_corres=None, ret_xy=True)
    reciprocal_count = int(len(source_xy))

    x1 = source_xy[:, 0].astype(np.int64)
    y1 = source_xy[:, 1].astype(np.int64)
    x2 = target_xy[:, 0].astype(np.int64)
    y2 = target_xy[:, 1].astype(np.int64)
    valid = (
        (x1 >= 0)
        & (x1 < depth1.shape[1])
        & (y1 >= 0)
        & (y1 < depth1.shape[0])
        & (x2 >= 0)
        & (x2 < depth2.shape[1])
        & (y2 >= 0)
        & (y2 < depth2.shape[0])
        & (depth1[y1, x1] > args.min_depth)
        & (depth2[y2, x2] > args.min_depth)
        & (depth1[y1, x1] <= args.max_depth)
        & (depth2[y2, x2] <= args.max_depth)
    )
    source_xy = source_xy[valid]
    target_xy = target_xy[valid]
    x1 = source_xy[:, 0].astype(np.int64)
    y1 = source_xy[:, 1].astype(np.int64)
    x2 = target_xy[:, 0].astype(np.int64)
    y2 = target_xy[:, 1].astype(np.int64)

    pts1_in_cam2 = camera_points_from_world(pts1_world, pose2)[y1, x1]
    pts2_in_cam2 = camera_points_from_world(pts2_world, pose2)[y2, x2]
    distances = np.linalg.norm(pts1_in_cam2 - pts2_in_cam2, axis=1)
    keep = np.isfinite(distances) & (distances <= args.depth_consistency_thresh)
    positives = make_positive(source_xy[keep], target_xy[keep], distances[keep], "geometry", depth_error=distances[keep])
    return positives, {
        "matching_style": "cropping.extract_correspondences_from_pts3d",
        "reciprocal": reciprocal_count,
        "valid_depth": int(valid.sum()),
        "after_filter": int(keep.sum()),
        "dist_thresh": float(args.depth_consistency_thresh),
        "raw": reciprocal_count,
    }
