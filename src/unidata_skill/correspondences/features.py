from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .dataset_views import as_image_array
from .geometry import to_numpy
from .sampling import PairSkip, make_positive


def image_to_tensor(image: np.ndarray, device: str):
    import torch

    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).to(device)


def extract_features(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    method = args.feature_method.lower()
    if method == "sift":
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detector = cv2.SIFT_create(nfeatures=args.max_keypoints)
        keypoints = detector.detect(gray, None)
        if not keypoints:
            raise PairSkip("feature_extractor_empty:sift")
        keypoints = sorted(keypoints, key=lambda item: item.response, reverse=True)[: args.max_keypoints]
        return (
            np.asarray([item.pt for item in keypoints], dtype=np.float32),
            np.asarray([item.response for item in keypoints], dtype=np.float32),
            "opencv_sift",
        )

    import torch
    from lightglue import ALIKED, SIFT, SuperPoint

    if method == "aliked":
        extractor = ALIKED(max_num_keypoints=args.max_keypoints, detection_threshold=args.detection_threshold)
    elif method in {"sp", "superpoint"}:
        extractor = SuperPoint(max_num_keypoints=args.max_keypoints, detection_threshold=args.detection_threshold)
    elif method == "lightglue_sift":
        extractor = SIFT(max_num_keypoints=args.max_keypoints)
    else:
        raise ValueError(f"unsupported feature method: {args.feature_method}")

    extractor = extractor.to(args.device).eval()
    with torch.no_grad():
        feats = extractor.extract(image_to_tensor(image, args.device), invalid_mask=None)
    xy = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    scores = feats.get("keypoint_scores")
    if scores is None:
        scores = feats.get("scores")
    score = np.ones(len(xy), dtype=np.float32) if scores is None else scores[0].detach().cpu().numpy().astype(np.float32)
    if len(xy) == 0:
        raise PairSkip(f"feature_extractor_empty:{method}")
    return xy, score, method


def feature_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    view1, view2 = to_numpy((view1, view2))
    image = as_image_array(view1["img"])
    xy1, score, method = extract_features(image, args)
    depth1 = np.asarray(view1["depthmap"], dtype=np.float32)
    depth2 = np.asarray(view2["depthmap"], dtype=np.float32)
    k1 = np.asarray(view1["camera_intrinsics"], dtype=np.float64)
    k2 = np.asarray(view2["camera_intrinsics"], dtype=np.float64)
    pose1 = np.asarray(view1["camera_pose"], dtype=np.float64)
    pose2 = np.asarray(view2["camera_pose"], dtype=np.float64)
    h1, w1 = depth1.shape
    h2, w2 = depth2.shape

    x_src_round = np.rint(xy1[:, 0]).astype(np.int64)
    y_src_round = np.rint(xy1[:, 1]).astype(np.int64)
    source_inside = (x_src_round >= 0) & (x_src_round < w1) & (y_src_round >= 0) & (y_src_round < h1)
    safe_x_src = np.clip(x_src_round, 0, w1 - 1)
    safe_y_src = np.clip(y_src_round, 0, h1 - 1)
    source_depth = depth1[safe_y_src, safe_x_src].astype(np.float64)
    source_depth_valid = (
        source_inside
        & np.isfinite(source_depth)
        & (source_depth > args.min_depth)
        & (source_depth <= args.max_depth)
    )

    z_src = source_depth
    x_cam_src = (xy1[:, 0].astype(np.float64) - k1[0, 2]) / k1[0, 0] * z_src
    y_cam_src = (xy1[:, 1].astype(np.float64) - k1[1, 2]) / k1[1, 1] * z_src
    points_src = np.stack((x_cam_src, y_cam_src, z_src, np.ones_like(z_src)), axis=1)
    points_world = (pose1 @ points_src.T).T
    points_dst = (np.linalg.inv(pose2) @ points_world.T).T[:, :3]
    projected_depth = points_dst[:, 2]

    target_xy = np.empty((len(xy1), 2), dtype=np.float64)
    target_xy[:, 0] = k2[0, 0] * points_dst[:, 0] / projected_depth + k2[0, 2]
    target_xy[:, 1] = k2[1, 1] * points_dst[:, 1] / projected_depth + k2[1, 2]

    x_dst_round = np.rint(target_xy[:, 0]).astype(np.int64)
    y_dst_round = np.rint(target_xy[:, 1]).astype(np.int64)
    target_inside = (x_dst_round >= 0) & (x_dst_round < w2) & (y_dst_round >= 0) & (y_dst_round < h2)
    safe_x_dst = np.clip(x_dst_round, 0, w2 - 1)
    safe_y_dst = np.clip(y_dst_round, 0, h2 - 1)
    target_depth = depth2[safe_y_dst, safe_x_dst].astype(np.float64)
    depth_error = np.abs(projected_depth - target_depth)

    keep = (
        source_depth_valid
        & np.isfinite(points_dst).all(axis=1)
        & np.isfinite(target_xy).all(axis=1)
        & (projected_depth > args.min_depth)
        & (projected_depth <= args.max_depth)
        & target_inside
        & np.isfinite(target_depth)
        & (target_depth > args.min_depth)
        & (target_depth <= args.max_depth)
        & np.isfinite(depth_error)
        & (depth_error <= args.depth_consistency_thresh)
    )

    positives = make_positive(xy1[keep], target_xy[keep].astype(np.float32), depth_error[keep], "feature", feature_score=score[keep], depth_error=depth_error[keep])
    positives["source_depth_m"] = source_depth[keep].astype(np.float32)
    positives["target_depth_m"] = target_depth[keep].astype(np.float32)
    positives["projected_target_depth_m"] = projected_depth[keep].astype(np.float32)
    positives["source_linear"] = (x_src_round[keep] + w1 * y_src_round[keep]).astype(np.int64)
    return positives, {
        "method": method,
        "projection_filter": "gt_depth_projection",
        "raw": int(len(xy1)),
        "raw_features": int(len(xy1)),
        "source_valid_depth": int(source_depth_valid.sum()),
        "target_inside": int((source_depth_valid & target_inside).sum()),
        "after_filter": int(keep.sum()),
        "after_depth_consistency": int(keep.sum()),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "depth_consistency_thresh": float(args.depth_consistency_thresh),
    }
