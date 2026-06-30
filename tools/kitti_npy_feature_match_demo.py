from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_npy_match_cropping_demo import (
    find_cropping_correspondences,
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
    if method == "geometry":
        return {
            "source_xy": np.empty((0, 2), dtype=np.float32),
            "target_xy": np.empty((0, 2), dtype=np.float32),
            "score": np.empty((0,), dtype=np.float32),
            "method": "none",
        }
    if method == "sift":
        return match_sift_features(image_src, image_dst, max_keypoints)
    if method in {"aliked", "sp", "superpoint", "lightglue_sift"}:
        return match_lightglue_features(image_src, image_dst, method, max_keypoints, detection_threshold, device)
    raise ValueError(f"unsupported feature method: {method}")


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


def visualize_overlay(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    geometry_matches: dict[str, np.ndarray],
    feature_matches: dict[str, np.ndarray | str],
    output_path: Path,
    stride: int,
    max_points: int,
    feature_max_points: int,
    feature_cross_size: float,
) -> tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom_src, geom_dst = select_viz_points(
        geometry_matches["source_xy"],
        geometry_matches["target_xy"],
        stride,
        max_points,
    )
    geom_color = geom_src[:, 0].astype(np.float32) + image_src.shape[1] * geom_src[:, 1].astype(np.float32)

    feat_src = np.asarray(feature_matches["source_xy"], dtype=np.float32)
    feat_dst = np.asarray(feature_matches["target_xy"], dtype=np.float32)
    feat_src, feat_dst = select_viz_points(feat_src, feat_dst, 1, feature_max_points)
    feat_color = np.arange(len(feat_src), dtype=np.float32)

    plt.figure("feature_geometry_overlay", figsize=[5, 6])
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(image_src)
    if len(geom_src):
        ax1.scatter(geom_src[:, 0], geom_src[:, 1], s=0.7, c=geom_color, cmap="jet", alpha=0.45)
    draw_crosses(ax1, feat_src, feat_color, feature_cross_size, "hsv")
    ax1.tick_params(labelbottom=False, labelleft=False)

    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(image_dst)
    if len(geom_dst):
        ax2.scatter(geom_dst[:, 0], geom_dst[:, 1], s=0.7, c=geom_color, cmap="jet", alpha=0.45)
    draw_crosses(ax2, feat_dst, feat_color, feature_cross_size, "hsv")
    ax2.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return int(len(geom_src)), int(len(feat_src))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI geometric correspondences overlaid with optional feature matches.")
    parser.add_argument("--index-file", "--table", type=Path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--feature-method", choices=["geometry", "sift", "aliked", "superpoint", "sp", "lightglue_sift"], default="sift")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--detection-threshold", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-thresh", type=float, default=0.25)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--viz-stride", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=3000)
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
    depth_src_path = Path(frame_src["depth"])
    depth_dst_path = Path(frame_dst["depth"])

    image_src = read_rgb(image_src_path)
    image_dst = read_rgb(image_dst_path)
    depth_src = read_kitti_depth(depth_src_path)
    depth_dst = read_kitti_depth(depth_dst_path)
    intrinsics_src = np.asarray(frame_src["camera_intrinsics"], dtype=np.float32)
    intrinsics_dst = np.asarray(frame_dst["camera_intrinsics"], dtype=np.float32)
    pose_src = np.asarray(frame_src["camera_pose"], dtype=np.float32)
    pose_dst = np.asarray(frame_dst["camera_pose"], dtype=np.float32)

    geometry_matches = find_cropping_correspondences(
        depth_src,
        depth_dst,
        intrinsics_src,
        intrinsics_dst,
        pose_src,
        pose_dst,
        args.min_depth,
        args.dist_thresh,
    )

    feature_matches = match_features(
        image_src,
        image_dst,
        args.feature_method,
        args.max_keypoints,
        args.detection_threshold,
        args.device,
    )

    output = {
        "sequence": sequence,
        "source_frame": frame_src,
        "target_frame": frame_dst,
        "geometry_source_xy": geometry_matches["source_xy"],
        "geometry_target_xy": geometry_matches["target_xy"],
        "geometry_source_linear": geometry_matches["source_linear"],
        "geometry_target_linear": geometry_matches["target_linear"],
        "geometry_distance_m": geometry_matches["distance_m"],
        "geometry_stats": geometry_matches["stats"],
        "feature_source_xy": np.asarray(feature_matches["source_xy"], dtype=np.float32),
        "feature_target_xy": np.asarray(feature_matches["target_xy"], dtype=np.float32),
        "feature_score": np.asarray(feature_matches["score"], dtype=np.float32),
        "feature_method": feature_matches["method"],
        "dist_thresh": np.asarray(args.dist_thresh, dtype=np.float32),
        "min_depth": np.asarray(args.min_depth, dtype=np.float32),
        "matching_style": "geometry_correspondences+feature_matches_overlay",
    }
    if "distance" in feature_matches:
        output["feature_distance"] = np.asarray(feature_matches["distance"], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, output, allow_pickle=True)
    cache_dir = args.viz_output.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    visualized_geometry, visualized_features = visualize_overlay(
        image_src,
        image_dst,
        geometry_matches,
        feature_matches,
        args.viz_output,
        args.viz_stride,
        args.max_points,
        args.max_feature_points,
        args.feature_cross_size,
    )

    print(f"sequence: {sequence}")
    print(f"source: {frame_src.get('frame_id')} {image_src_path}")
    print(f"target: {frame_dst.get('frame_id')} {image_dst_path}")
    print(f"feature method: {feature_matches['method']}")
    print(f"geometry matches: {len(geometry_matches['source_xy'])}")
    print(f"feature matches: {len(feature_matches['source_xy'])}")
    print(f"geometry stats: {geometry_matches['stats']}")
    print(f"visualized geometry: {visualized_geometry}")
    print(f"visualized features: {visualized_features}")
    print(f"saved matches: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
