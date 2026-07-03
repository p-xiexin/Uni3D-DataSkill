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


def extract_lightglue_keypoints(
    image: np.ndarray,
    method: str,
    max_keypoints: int,
    detection_threshold: float,
    device: str,
) -> dict[str, np.ndarray | str]:
    import torch
    from lightglue import ALIKED, SIFT, SuperPoint

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
    with torch.no_grad():
        feats = extractor.extract(image_to_tensor(image, device), invalid_mask=None)

    keypoints = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    scores = feats.get("keypoint_scores")
    if scores is None:
        scores = feats.get("scores")
    if scores is None:
        score_np = np.ones(len(keypoints), dtype=np.float32)
    else:
        score_np = scores[0].detach().cpu().numpy().astype(np.float32)
    if len(keypoints) == 0:
        raise RuntimeError(f"feature extractor produced no keypoints: {method}")
    return {"source_xy": keypoints, "score": score_np, "method": method}


def extract_sift_keypoints(image: np.ndarray, max_keypoints: int) -> dict[str, np.ndarray | str]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints = sift.detect(gray, None)
    if not keypoints:
        raise RuntimeError("SIFT produced no keypoints")

    keypoints = sorted(keypoints, key=lambda item: item.response, reverse=True)[:max_keypoints]
    source_xy = np.asarray([item.pt for item in keypoints], dtype=np.float32)
    score = np.asarray([item.response for item in keypoints], dtype=np.float32)
    return {"source_xy": source_xy, "score": score, "method": "opencv_sift"}


def extract_source_features(
    image_src: np.ndarray,
    method: str,
    max_keypoints: int,
    detection_threshold: float,
    device: str,
) -> dict[str, np.ndarray | str]:
    method = method.lower()
    if method == "sift":
        return extract_sift_keypoints(image_src, max_keypoints)
    if method in {"aliked", "sp", "superpoint", "lightglue_sift"}:
        return extract_lightglue_keypoints(image_src, method, max_keypoints, detection_threshold, device)
    raise ValueError(f"unsupported feature method: {method}")


def filter_feature_arrays(features: dict[str, np.ndarray | str], keep: np.ndarray) -> dict[str, np.ndarray | str]:
    count = len(features["source_xy"])
    filtered: dict[str, np.ndarray | str] = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == count:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    return filtered


def require_geometry_frame(frame: dict, role: str) -> None:
    for key in ("depth", "camera_intrinsics", "camera_pose"):
        if key not in frame:
            raise KeyError(f"{role} frame is missing required key for GT projection: {key}")
    if not Path(frame["depth"]).is_file():
        raise FileNotFoundError(f"{role} depth file does not exist: {frame['depth']}")


def project_source_features_with_gt(
    features: dict[str, np.ndarray | str],
    frame_src: dict,
    frame_dst: dict,
    image_dst_shape: tuple[int, int],
    min_depth: float,
    max_depth: float,
    depth_consistency_thresh: float,
) -> tuple[dict[str, np.ndarray | str], dict[str, int | float | str]]:
    require_geometry_frame(frame_src, "source")
    require_geometry_frame(frame_dst, "target")

    source_xy = np.asarray(features["source_xy"], dtype=np.float32)
    depth_src = read_kitti_depth(Path(frame_src["depth"]))
    depth_dst = read_kitti_depth(Path(frame_dst["depth"]))
    intrinsics_src = np.asarray(frame_src["camera_intrinsics"], dtype=np.float64)
    intrinsics_dst = np.asarray(frame_dst["camera_intrinsics"], dtype=np.float64)
    pose_src = np.asarray(frame_src["camera_pose"], dtype=np.float64)
    pose_dst = np.asarray(frame_dst["camera_pose"], dtype=np.float64)

    if intrinsics_src.shape != (3, 3) or intrinsics_dst.shape != (3, 3):
        raise ValueError("camera_intrinsics must be 3x3")
    if pose_src.shape != (4, 4) or pose_dst.shape != (4, 4):
        raise ValueError("camera_pose must be 4x4")

    h_src, w_src = depth_src.shape
    h_dst, w_dst = depth_dst.shape
    image_h_dst, image_w_dst = image_dst_shape
    if (h_dst, w_dst) != (image_h_dst, image_w_dst):
        raise ValueError(
            f"target depth/image shape mismatch: depth={(h_dst, w_dst)} image={(image_h_dst, image_w_dst)}"
        )

    x_src_round = np.rint(source_xy[:, 0]).astype(np.int64)
    y_src_round = np.rint(source_xy[:, 1]).astype(np.int64)
    source_inside = (x_src_round >= 0) & (x_src_round < w_src) & (y_src_round >= 0) & (y_src_round < h_src)
    safe_x_src = np.clip(x_src_round, 0, w_src - 1)
    safe_y_src = np.clip(y_src_round, 0, h_src - 1)
    source_depth = depth_src[safe_y_src, safe_x_src].astype(np.float64)
    source_depth_valid = (
        source_inside
        & np.isfinite(source_depth)
        & (source_depth > min_depth)
        & (source_depth <= max_depth)
    )

    z_src = source_depth
    x_cam_src = (source_xy[:, 0].astype(np.float64) - intrinsics_src[0, 2]) / intrinsics_src[0, 0] * z_src
    y_cam_src = (source_xy[:, 1].astype(np.float64) - intrinsics_src[1, 2]) / intrinsics_src[1, 1] * z_src
    points_src = np.stack((x_cam_src, y_cam_src, z_src, np.ones_like(z_src)), axis=1)
    points_world = (pose_src @ points_src.T).T
    points_dst = (np.linalg.inv(pose_dst) @ points_world.T).T[:, :3]
    projected_depth = points_dst[:, 2]

    target_xy = np.empty((len(source_xy), 2), dtype=np.float64)
    target_xy[:, 0] = intrinsics_dst[0, 0] * points_dst[:, 0] / projected_depth + intrinsics_dst[0, 2]
    target_xy[:, 1] = intrinsics_dst[1, 1] * points_dst[:, 1] / projected_depth + intrinsics_dst[1, 2]

    x_dst_round = np.rint(target_xy[:, 0]).astype(np.int64)
    y_dst_round = np.rint(target_xy[:, 1]).astype(np.int64)
    target_inside = (x_dst_round >= 0) & (x_dst_round < w_dst) & (y_dst_round >= 0) & (y_dst_round < h_dst)
    safe_x_dst = np.clip(x_dst_round, 0, w_dst - 1)
    safe_y_dst = np.clip(y_dst_round, 0, h_dst - 1)
    target_depth = depth_dst[safe_y_dst, safe_x_dst].astype(np.float64)
    depth_error = np.abs(projected_depth - target_depth)

    keep = (
        source_depth_valid
        & np.isfinite(points_dst).all(axis=1)
        & np.isfinite(target_xy).all(axis=1)
        & (projected_depth > min_depth)
        & (projected_depth <= max_depth)
        & target_inside
        & np.isfinite(target_depth)
        & (target_depth > min_depth)
        & (target_depth <= max_depth)
        & np.isfinite(depth_error)
        & (depth_error <= depth_consistency_thresh)
    )
    if not keep.any():
        raise RuntimeError("GT projection and depth filter removed every feature")

    projected = filter_feature_arrays(features, keep)
    projected["target_xy"] = target_xy[keep].astype(np.float32)
    projected["source_depth_m"] = source_depth[keep].astype(np.float32)
    projected["target_depth_m"] = target_depth[keep].astype(np.float32)
    projected["projected_target_depth_m"] = projected_depth[keep].astype(np.float32)
    projected["target_depth_error_m"] = depth_error[keep].astype(np.float32)
    projected["source_linear"] = (x_src_round[keep] + w_src * y_src_round[keep]).astype(np.int64)
    return projected, {
        "projection_filter": "gt_depth_projection",
        "raw_features": int(len(source_xy)),
        "source_valid_depth": int(source_depth_valid.sum()),
        "target_inside": int((source_depth_valid & target_inside).sum()),
        "after_depth_consistency": int(keep.sum()),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "depth_consistency_thresh": float(depth_consistency_thresh),
    }


def select_viz_points(source_xy: np.ndarray, target_xy: np.ndarray, stride: int, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_xy = source_xy[::stride]
    target_xy = target_xy[::stride]
    color_values = np.arange(len(source_xy), dtype=np.float32)
    if len(source_xy) > max_points:
        pick = np.linspace(0, len(source_xy) - 1, max_points).astype(np.int64)
        source_xy = source_xy[pick]
        target_xy = target_xy[pick]
        color_values = color_values[pick]
    return source_xy, target_xy, color_values


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


def visualize_projected_features(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    features: dict[str, np.ndarray | str],
    output_path: Path,
    max_points: int,
    cross_size: float,
    stride: int,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat_src = np.asarray(features["source_xy"], dtype=np.float32)
    feat_dst = np.asarray(features["target_xy"], dtype=np.float32)
    feat_src, feat_dst, feat_color = select_viz_points(feat_src, feat_dst, stride, max_points)

    plt.figure("feature_gt_projection", figsize=[5, 6])
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(image_src)
    draw_crosses(ax1, feat_src, feat_color, cross_size, "hsv")
    ax1.tick_params(labelbottom=False, labelleft=False)

    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(image_dst)
    draw_crosses(ax2, feat_dst, feat_color, cross_size, "hsv")
    ax2.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return int(len(feat_src))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI source-feature GT projection demo.")
    parser.add_argument("--index-file", "--table", type=Path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--feature-method", choices=["sift", "aliked", "superpoint", "sp", "lightglue_sift"], default="sift")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--detection-threshold", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--depth-consistency-thresh", type=float, default=0.25)
    parser.add_argument("--max-feature-points", type=int, default=1000)
    parser.add_argument("--feature-cross-size", type=float, default=6.0)
    parser.add_argument("--viz-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_npy_feature_match_demo/matches.npy"))
    parser.add_argument("--viz-output", type=Path, default=Path("outputs/kitti_npy_feature_match_demo/matches.jpg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.depth_consistency_thresh <= 0:
        raise ValueError("--depth-consistency-thresh must be positive")
    if args.viz_stride <= 0:
        raise ValueError("--viz-stride must be positive")

    index = load_index(args.index_file)
    sequence, frames = select_frames(index, args.sequence)
    frame_src = frames[args.source_frame]
    frame_dst = frames[args.target_frame]

    image_src_path = Path(frame_src["image"])
    image_dst_path = Path(frame_dst["image"])
    image_src = read_rgb(image_src_path)
    image_dst = read_rgb(image_dst_path)

    source_features = extract_source_features(
        image_src,
        args.feature_method,
        args.max_keypoints,
        args.detection_threshold,
        args.device,
    )
    raw_feature_count = int(len(source_features["source_xy"]))
    projected_features, projection_stats = project_source_features_with_gt(
        source_features,
        frame_src,
        frame_dst,
        image_dst.shape[:2],
        args.min_depth,
        args.max_depth,
        args.depth_consistency_thresh,
    )

    output = {
        "sequence": sequence,
        "source_frame": frame_src,
        "target_frame": frame_dst,
        "feature_source_xy": np.asarray(projected_features["source_xy"], dtype=np.float32),
        "feature_target_xy": np.asarray(projected_features["target_xy"], dtype=np.float32),
        "feature_score": np.asarray(projected_features["score"], dtype=np.float32),
        "feature_method": projected_features["method"],
        "raw_feature_count": np.asarray(raw_feature_count, dtype=np.int32),
        "projection_stats": np.asarray(projection_stats, dtype=object),
        "matching_style": "source_features_gt_depth_projection",
        "source_depth_m": np.asarray(projected_features["source_depth_m"], dtype=np.float32),
        "target_depth_m": np.asarray(projected_features["target_depth_m"], dtype=np.float32),
        "projected_target_depth_m": np.asarray(projected_features["projected_target_depth_m"], dtype=np.float32),
        "target_depth_error_m": np.asarray(projected_features["target_depth_error_m"], dtype=np.float32),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, output, allow_pickle=True)
    cache_dir = args.viz_output.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    visualized_features = visualize_projected_features(
        image_src,
        image_dst,
        projected_features,
        args.viz_output,
        args.max_feature_points,
        args.feature_cross_size,
        args.viz_stride,
    )

    print(f"sequence: {sequence}")
    print(f"source: {frame_src.get('frame_id')} {image_src_path}")
    print(f"target: {frame_dst.get('frame_id')} {image_dst_path}")
    print(f"feature method: {projected_features['method']}")
    print(f"raw source features: {raw_feature_count}")
    print(f"projected features: {len(projected_features['source_xy'])}")
    print(f"projection stats: {projection_stats}")
    print(f"visualized features: {visualized_features}")
    print(f"saved projected features: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
