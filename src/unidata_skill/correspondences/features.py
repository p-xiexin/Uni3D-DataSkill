from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .dataset_views import as_image_array
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

    source_xy = np.rint(xy1).astype(np.int64)
    inside1 = (source_xy[:, 0] >= 0) & (source_xy[:, 0] < w1) & (source_xy[:, 1] >= 0) & (source_xy[:, 1] < h1)
    safe_x1 = np.clip(source_xy[:, 0], 0, w1 - 1)
    safe_y1 = np.clip(source_xy[:, 1], 0, h1 - 1)
    z1 = depth1[safe_y1, safe_x1].astype(np.float64)
    cam1 = np.stack(((xy1[:, 0] - k1[0, 2]) / k1[0, 0] * z1, (xy1[:, 1] - k1[1, 2]) / k1[1, 1] * z1, z1, np.ones_like(z1)), axis=1)
    world = (pose1 @ cam1.T).T
    cam2 = (np.linalg.inv(pose2) @ world.T).T[:, :3]
    z2 = cam2[:, 2]
    xy2 = np.empty((len(xy1), 2), dtype=np.float64)
    xy2[:, 0] = k2[0, 0] * cam2[:, 0] / z2 + k2[0, 2]
    xy2[:, 1] = k2[1, 1] * cam2[:, 1] / z2 + k2[1, 2]
    target_xy = np.rint(xy2).astype(np.int64)

    inside2 = (target_xy[:, 0] >= 0) & (target_xy[:, 0] < w2) & (target_xy[:, 1] >= 0) & (target_xy[:, 1] < h2)
    safe_x2 = np.clip(target_xy[:, 0], 0, w2 - 1)
    safe_y2 = np.clip(target_xy[:, 1], 0, h2 - 1)
    target_depth = depth2[safe_y2, safe_x2].astype(np.float64)
    depth_error = np.abs(z2 - target_depth)
    keep = (
        inside1
        & inside2
        & np.isfinite(z1)
        & np.isfinite(z2)
        & np.isfinite(target_depth)
        & (z1 > args.min_depth)
        & (z2 > args.min_depth)
        & (target_depth > args.min_depth)
        & (z1 <= args.max_depth)
        & (z2 <= args.max_depth)
        & (target_depth <= args.max_depth)
        & (depth_error <= args.depth_consistency_thresh)
    )
    positives = make_positive(source_xy[keep], target_xy[keep], depth_error[keep], "feature", feature_score=score[keep], depth_error=depth_error[keep])
    return positives, {"method": method, "raw": int(len(xy1)), "after_filter": int(keep.sum())}
