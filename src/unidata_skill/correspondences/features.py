from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .dataset_views import as_image_array
from .geometry import project_pixels_between_views_numpy, to_numpy
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
    source_xy, target_xy, depth_error, keep, projection_stats = project_pixels_between_views_numpy(
        xy1,
        depth1,
        depth2,
        k1,
        k2,
        pose1,
        pose2,
        args,
    )
    positives = make_positive(source_xy[keep], target_xy[keep], depth_error[keep], "feature", feature_score=score[keep], depth_error=depth_error[keep])
    projection_stats.update({"method": method, "raw": int(len(xy1))})
    return positives, projection_stats
