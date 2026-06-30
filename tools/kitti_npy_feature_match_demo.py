from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_npy_match_cropping_demo import (
    load_index,
    read_kitti_depth,
    read_rgb,
    select_frames,
)


def image_to_tensor(image: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def lightglue_feature_name(method: str) -> str:
    method = method.lower()
    if method == "aliked":
        return "aliked"
    if method in {"sp", "superpoint"}:
        return "superpoint"
    if method == "lightglue_sift":
        return "sift"
    raise ValueError(f"unsupported LightGlue extractor: {method}")


def match_lightglue_features(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    method: str,
    max_keypoints: int,
    detection_threshold: float,
    device: str,
) -> dict[str, np.ndarray | str]:
    import torch
    from lightglue import ALIKED, SIFT, LightGlue, SuperPoint

    method = method.lower()
    if method == "aliked":
        extractor = ALIKED(max_num_keypoints=max_keypoints, detection_threshold=detection_threshold)
    elif method in {"sp", "superpoint"}:
        extractor = SuperPoint(max_num_keypoints=max_keypoints, detection_threshold=detection_threshold)
    elif method == "lightglue_sift":
        extractor = SIFT(max_num_keypoints=max_keypoints)
    else:
        raise ValueError(f"unsupported LightGlue extractor: {method}")

    extractor = extractor.to(device).eval()
    matcher = LightGlue(features=lightglue_feature_name(method)).to(device).eval()
    image0 = image_to_tensor(image_src, device)
    image1 = image_to_tensor(image_dst, device)
    with torch.no_grad():
        feats0 = extractor.extract(image0, invalid_mask=None)
        feats1 = extractor.extract(image1, invalid_mask=None)
        pred = matcher({"image0": feats0, "image1": feats1})

    matches = pred["matches"][0].detach().cpu().numpy().astype(np.int64)
    if matches.size == 0:
        raise RuntimeError(f"feature matcher produced no matches: {method}")
    keypoints0 = feats0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    keypoints1 = feats1["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    scores = pred.get("scores")
    if scores is None:
        scores_np = np.ones(len(matches), dtype=np.float32)
    else:
        scores_np = scores[0].detach().cpu().numpy().astype(np.float32)
    return {
        "source_xy": keypoints0[matches[:, 0]],
        "target_xy": keypoints1[matches[:, 1]],
        "score": scores_np,
        "method": method,
    }


def match_sift_features(image_src: np.ndarray, image_dst: np.ndarray, max_keypoints: int) -> dict[str, np.ndarray | str]:
    import cv2

    gray_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2GRAY)
    gray_dst = cv2.cvtColor(image_dst, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints_src, desc_src = sift.detectAndCompute(gray_src, None)
    keypoints_dst, desc_dst = sift.detectAndCompute(gray_dst, None)
    if desc_src is None or desc_dst is None or not keypoints_src or not keypoints_dst:
        raise RuntimeError("SIFT produced no descriptors")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(desc_src, desc_dst, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)
    if not good_matches:
        raise RuntimeError("SIFT produced no ratio-test matches")

    good_matches = sorted(good_matches, key=lambda item: item.distance)[:max_keypoints]
    src_xy = np.asarray([keypoints_src[item.queryIdx].pt for item in good_matches], dtype=np.float32)
    dst_xy = np.asarray([keypoints_dst[item.trainIdx].pt for item in good_matches], dtype=np.float32)
    distance = np.asarray([item.distance for item in good_matches], dtype=np.float32)
    score = 1.0 / (1.0 + distance)
    return {
        "source_xy": src_xy,
        "target_xy": dst_xy,
        "score": score.astype(np.float32),
        "distance": distance,
        "method": "opencv_sift",
    }


def match_features(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    method: str,
    max_keypoints: int,
    detection_threshold: float,
    device: str,
) -> dict[str, np.ndarray | str]:
    method = method.lower()
    if method == "sift":
        return match_sift_features(image_src, image_dst, max_keypoints)
    if method in {"aliked", "sp", "superpoint", "lightglue_sift"}:
        return match_lightglue_features(image_src, image_dst, method, max_keypoints, detection_threshold, device)
    raise ValueError(f"unsupported feature method: {method}")


def filter_match_arrays(feature_matches: dict[str, np.ndarray | str], keep: np.ndarray) -> dict[str, np.ndarray | str]:
    count = len(feature_matches["source_xy"])
    filtered: dict[str, np.ndarray | str] = {}
    for key, value in feature_matches.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == count:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    return filtered


def filter_outliers(
    feature_matches: dict[str, np.ndarray | str],
    method: str,
    threshold: float,
    confidence: float,
    max_iters: int,
) -> tuple[dict[str, np.ndarray | str], dict[str, int | float | str]]:
    method = method.lower()
    if method == "none":
        return feature_matches, {"outlier_filter": "none", "before": int(len(feature_matches["source_xy"])), "after": int(len(feature_matches["source_xy"]))}

    import cv2

    source_xy = np.asarray(feature_matches["source_xy"], dtype=np.float32)
    target_xy = np.asarray(feature_matches["target_xy"], dtype=np.float32)
    min_matches = 8 if method == "fundamental" else 4
    if len(source_xy) < min_matches:
        raise RuntimeError(f"{method} outlier filter needs at least {min_matches} matches, got {len(source_xy)}")

    if method == "fundamental":
        _, mask = cv2.findFundamentalMat(
            source_xy,
            target_xy,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=threshold,
            confidence=confidence,
            maxIters=max_iters,
        )
    elif method == "homography":
        _, mask = cv2.findHomography(
            source_xy,
            target_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            confidence=confidence,
            maxIters=max_iters,
        )
    else:
        raise ValueError(f"unsupported outlier filter: {method}")

    if mask is None:
        raise RuntimeError(f"{method} outlier filter failed")
    keep = mask.reshape(-1).astype(bool)
    if not keep.any():
        raise RuntimeError(f"{method} outlier filter removed every match")
    return filter_match_arrays(feature_matches, keep), {
        "outlier_filter": method,
        "before": int(len(source_xy)),
        "after": int(keep.sum()),
        "threshold_px": float(threshold),
    }


def require_frame_geometry(frame: dict, role: str) -> None:
    for key in ("depth", "camera_intrinsics", "camera_pose"):
        if key not in frame:
            raise KeyError(f"{role} frame is missing required key for depth filtering: {key}")
    if not Path(frame["depth"]).is_file():
        raise FileNotFoundError(f"{role} depth file does not exist: {frame['depth']}")


def filter_by_depth(
    feature_matches: dict[str, np.ndarray | str],
    frame_src: dict,
    frame_dst: dict,
    min_depth: float,
    depth_abs_thresh: float,
    depth_rel_thresh: float,
    reproj_thresh: float,
) -> tuple[dict[str, np.ndarray | str], dict[str, int | float | str]]:
    require_frame_geometry(frame_src, "source")
    require_frame_geometry(frame_dst, "target")

    source_xy = np.asarray(feature_matches["source_xy"], dtype=np.float32)
    target_xy = np.asarray(feature_matches["target_xy"], dtype=np.float32)
    depth_src = read_kitti_depth(Path(frame_src["depth"]))
    depth_dst = read_kitti_depth(Path(frame_dst["depth"]))
    intrinsics_src = np.asarray(frame_src["camera_intrinsics"], dtype=np.float32)
    intrinsics_dst = np.asarray(frame_dst["camera_intrinsics"], dtype=np.float32)
    pose_src = np.asarray(frame_src["camera_pose"], dtype=np.float32)
    pose_dst = np.asarray(frame_dst["camera_pose"], dtype=np.float32)

    h1, w1 = depth_src.shape
    h2, w2 = depth_dst.shape
    x1 = np.rint(source_xy[:, 0]).astype(np.int64)
    y1 = np.rint(source_xy[:, 1]).astype(np.int64)
    x2 = np.rint(target_xy[:, 0]).astype(np.int64)
    y2 = np.rint(target_xy[:, 1]).astype(np.int64)

    inside = (x1 >= 0) & (x1 < w1) & (y1 >= 0) & (y1 < h1) & (x2 >= 0) & (x2 < w2) & (y2 >= 0) & (y2 < h2)
    safe_x1 = np.clip(x1, 0, w1 - 1)
    safe_y1 = np.clip(y1, 0, h1 - 1)
    safe_x2 = np.clip(x2, 0, w2 - 1)
    safe_y2 = np.clip(y2, 0, h2 - 1)
    source_depth = depth_src[safe_y1, safe_x1]
    target_depth = depth_dst[safe_y2, safe_x2]

    ones = np.ones((len(source_xy), 1), dtype=np.float32)
    source_h = np.concatenate((source_xy, ones), axis=1)
    source_cam = (np.linalg.inv(intrinsics_src).astype(np.float32) @ source_h.T).T * source_depth[:, None]
    source_world_h = (pose_src.astype(np.float64) @ np.concatenate((source_cam, ones), axis=1).T).T
    target_cam = (np.linalg.inv(pose_dst.astype(np.float64)) @ source_world_h.T).T[:, :3]
    projected_z = target_cam[:, 2].astype(np.float32)

    projected_h = (intrinsics_dst.astype(np.float64) @ target_cam.T).T
    projected_xy = (projected_h[:, :2] / projected_h[:, 2:3]).astype(np.float32)
    reproj_error = np.linalg.norm(projected_xy - target_xy, axis=1).astype(np.float32)
    depth_error = np.abs(target_depth - projected_z).astype(np.float32)
    depth_limit = np.maximum(depth_abs_thresh, depth_rel_thresh * np.maximum(projected_z, min_depth)).astype(np.float32)

    keep = (
        inside
        & np.isfinite(reproj_error)
        & np.isfinite(depth_error)
        & (source_depth > min_depth)
        & (target_depth > min_depth)
        & (projected_z > min_depth)
        & (reproj_error <= reproj_thresh)
        & (depth_error <= depth_limit)
    )
    if not keep.any():
        raise RuntimeError("depth filter removed every match")

    filtered = filter_match_arrays(feature_matches, keep)
    filtered["depth_reproj_error_px"] = reproj_error[keep]
    filtered["depth_error_m"] = depth_error[keep]
    return filtered, {
        "depth_filter": "enabled",
        "before": int(len(source_xy)),
        "after": int(keep.sum()),
        "min_depth": float(min_depth),
        "depth_abs_thresh": float(depth_abs_thresh),
        "depth_rel_thresh": float(depth_rel_thresh),
        "reproj_thresh_px": float(reproj_thresh),
    }


def select_viz_points(source_xy: np.ndarray, target_xy: np.ndarray, stride: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    source_xy = source_xy[::stride]
    target_xy = target_xy[::stride]
    if len(source_xy) > max_points:
        pick = np.linspace(0, len(source_xy) - 1, max_points).astype(np.int64)
        source_xy = source_xy[pick]
        target_xy = target_xy[pick]
    return source_xy, target_xy


def draw_crosses(axis, xy: np.ndarray, color_values: np.ndarray, size: float, cmap: str) -> None:
    if len(xy) == 0:
        return
    x = xy[:, 0]
    y = xy[:, 1]
    span = size / 2.0
    norm = color_values / max(float(color_values.max()), 1.0)
    import matplotlib.pyplot as plt

    colors = plt.get_cmap(cmap)(norm)
    axis.hlines(y, x - span, x + span, colors=colors, linewidth=0.8)
    axis.vlines(x, y - span, y + span, colors=colors, linewidth=0.8)


def visualize_feature_matches(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    feature_matches: dict[str, np.ndarray | str],
    output_path: Path,
    feature_max_points: int,
    feature_cross_size: float,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_src = np.asarray(feature_matches["source_xy"], dtype=np.float32)
    feat_dst = np.asarray(feature_matches["target_xy"], dtype=np.float32)
    feat_src, feat_dst = select_viz_points(feat_src, feat_dst, 1, feature_max_points)
    feat_color = np.arange(len(feat_src), dtype=np.float32)

    plt.figure("feature_matches", figsize=[5, 6])
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(image_src)
    draw_crosses(ax1, feat_src, feat_color, feature_cross_size, "hsv")
    ax1.tick_params(labelbottom=False, labelleft=False)

    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(image_dst)
    draw_crosses(ax2, feat_dst, feat_color, feature_cross_size, "hsv")
    ax2.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return int(len(feat_src))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI image feature matching demo.")
    parser.add_argument("--index-file", "--table", type=Path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--feature-method", choices=["sift", "aliked", "superpoint", "sp", "lightglue_sift"], default="sift")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--detection-threshold", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outlier-filter", choices=["none", "fundamental", "homography"], default="fundamental")
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iters", type=int, default=10000)
    parser.add_argument("--depth-filter", action="store_true")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--depth-abs-thresh", type=float, default=0.25)
    parser.add_argument("--depth-rel-thresh", type=float, default=0.05)
    parser.add_argument("--depth-reproj-thresh", type=float, default=3.0)
    parser.add_argument("--max-feature-points", type=int, default=1000)
    parser.add_argument("--feature-cross-size", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_npy_feature_match_demo/matches.npy"))
    parser.add_argument("--viz-output", type=Path, default=Path("outputs/kitti_npy_feature_match_demo/matches.jpg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = load_index(args.index_file)
    sequence, frames = select_frames(index, args.sequence)
    frame_src = frames[args.source_frame]
    frame_dst = frames[args.target_frame]

    image_src_path = Path(frame_src["image"])
    image_dst_path = Path(frame_dst["image"])

    image_src = read_rgb(image_src_path)
    image_dst = read_rgb(image_dst_path)

    feature_matches = match_features(
        image_src,
        image_dst,
        args.feature_method,
        args.max_keypoints,
        args.detection_threshold,
        args.device,
    )
    raw_feature_count = int(len(feature_matches["source_xy"]))
    filter_stats: list[dict[str, int | float | str]] = []
    feature_matches, stats = filter_outliers(
        feature_matches,
        args.outlier_filter,
        args.ransac_thresh,
        args.ransac_confidence,
        args.ransac_max_iters,
    )
    filter_stats.append(stats)
    if args.depth_filter:
        feature_matches, stats = filter_by_depth(
            feature_matches,
            frame_src,
            frame_dst,
            args.min_depth,
            args.depth_abs_thresh,
            args.depth_rel_thresh,
            args.depth_reproj_thresh,
        )
        filter_stats.append(stats)

    output = {
        "sequence": sequence,
        "source_frame": frame_src,
        "target_frame": frame_dst,
        "feature_source_xy": np.asarray(feature_matches["source_xy"], dtype=np.float32),
        "feature_target_xy": np.asarray(feature_matches["target_xy"], dtype=np.float32),
        "feature_score": np.asarray(feature_matches["score"], dtype=np.float32),
        "feature_method": feature_matches["method"],
        "raw_feature_count": np.asarray(raw_feature_count, dtype=np.int32),
        "filter_stats": np.asarray(filter_stats, dtype=object),
        "matching_style": "feature_matches",
    }
    if "distance" in feature_matches:
        output["feature_distance"] = np.asarray(feature_matches["distance"], dtype=np.float32)
    if "depth_reproj_error_px" in feature_matches:
        output["depth_reproj_error_px"] = np.asarray(feature_matches["depth_reproj_error_px"], dtype=np.float32)
    if "depth_error_m" in feature_matches:
        output["depth_error_m"] = np.asarray(feature_matches["depth_error_m"], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, output, allow_pickle=True)
    cache_dir = args.viz_output.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    visualized_features = visualize_feature_matches(
        image_src,
        image_dst,
        feature_matches,
        args.viz_output,
        args.max_feature_points,
        args.feature_cross_size,
    )

    print(f"sequence: {sequence}")
    print(f"source: {frame_src.get('frame_id')} {image_src_path}")
    print(f"target: {frame_dst.get('frame_id')} {image_dst_path}")
    print(f"feature method: {feature_matches['method']}")
    print(f"raw feature matches: {raw_feature_count}")
    print(f"feature matches: {len(feature_matches['source_xy'])}")
    print(f"filter stats: {filter_stats}")
    print(f"visualized features: {visualized_features}")
    print(f"saved matches: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
