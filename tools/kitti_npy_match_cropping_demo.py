from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cropping import extract_correspondences_from_pts3d


def load_index(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def select_frames(index: dict[str, Any], sequence: str | None) -> tuple[str, list[dict[str, Any]]]:
    records = index.get("sequences", [])
    if sequence is None:
        record = next(item for item in records if item.get("frames"))
    else:
        record = next(item for item in records if item.get("sequence_id") == sequence)
    return str(record["sequence_id"]), list(record["frames"])


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_kitti_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 256.0


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


def pixel_to_lin(xy: np.ndarray, width: int) -> np.ndarray:
    return xy[:, 0].astype(np.int64) + width * xy[:, 1].astype(np.int64)


def filter_correspondences(
    pos1: np.ndarray,
    pos2: np.ndarray,
    view1: dict[str, np.ndarray],
    view2: dict[str, np.ndarray],
    depth1: np.ndarray,
    depth2: np.ndarray,
    min_depth: float,
    dist_thresh: float,
) -> dict[str, np.ndarray]:
    h, w = depth1.shape
    x1 = pos1[:, 0].astype(np.int64)
    y1 = pos1[:, 1].astype(np.int64)
    x2 = pos2[:, 0].astype(np.int64)
    y2 = pos2[:, 1].astype(np.int64)

    valid = (
        (x1 >= 0)
        & (x1 < w)
        & (y1 >= 0)
        & (y1 < h)
        & (x2 >= 0)
        & (x2 < w)
        & (y2 >= 0)
        & (y2 < h)
        & (depth1[y1, x1] > min_depth)
        & (depth2[y2, x2] > min_depth)
    )
    x1 = x1[valid]
    y1 = y1[valid]
    x2 = x2[valid]
    y2 = y2[valid]

    pts1_in_cam2 = camera_points_from_world(view1["pts3d"], view2["camera_pose"])[y1, x1]
    pts2_in_cam2 = camera_points_from_world(view2["pts3d"], view2["camera_pose"])[y2, x2]
    distances = np.linalg.norm(pts1_in_cam2 - pts2_in_cam2, axis=1)
    keep = np.isfinite(distances) & (distances <= dist_thresh)

    source_xy = np.stack((x1[keep], y1[keep]), axis=1).astype(np.int32)
    target_xy = np.stack((x2[keep], y2[keep]), axis=1).astype(np.int32)
    return {
        "source_xy": source_xy,
        "target_xy": target_xy,
        "source_linear": pixel_to_lin(source_xy, w),
        "target_linear": pixel_to_lin(target_xy, w),
        "distance_m": distances[keep].astype(np.float32),
        "image_shape": np.asarray([h, w], dtype=np.int32),
        "stats": {
            "reciprocal": int(len(pos1)),
            "valid_depth": int(valid.sum()),
            "after_distance": int(keep.sum()),
        },
    }


def find_cropping_correspondences(
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intrinsics_src: np.ndarray,
    intrinsics_dst: np.ndarray,
    pose_src: np.ndarray,
    pose_dst: np.ndarray,
    min_depth: float,
    dist_thresh: float,
) -> dict[str, np.ndarray]:
    view_src = {
        "pts3d": world_points_from_depth(depth_src, intrinsics_src, pose_src),
        "camera_intrinsics": intrinsics_src,
        "camera_pose": pose_src,
    }
    view_dst = {
        "pts3d": world_points_from_depth(depth_dst, intrinsics_dst, pose_dst),
        "camera_intrinsics": intrinsics_dst,
        "camera_pose": pose_dst,
    }
    pos_src, pos_dst = extract_correspondences_from_pts3d(view_src, view_dst, target_n_corres=None, ret_xy=True)
    return filter_correspondences(pos_src, pos_dst, view_src, view_dst, depth_src, depth_dst, min_depth, dist_thresh)


def visualize_matches(
    image_src: np.ndarray,
    image_dst: np.ndarray,
    matches: dict[str, np.ndarray],
    output_path: Path,
    stride: int,
    max_points: int,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_xy = matches["source_xy"][::stride]
    target_xy = matches["target_xy"][::stride]
    color_values = matches["source_linear"][::stride]
    if len(source_xy) > max_points:
        pick = np.linspace(0, len(source_xy) - 1, max_points).astype(np.int64)
        source_xy = source_xy[pick]
        target_xy = target_xy[pick]
        color_values = color_values[pick]

    plt.figure("1", figsize=[5, 6])
    plt.subplot(2, 1, 1)
    plt.imshow(image_src)
    plt.scatter(source_xy[:, 0], source_xy[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)

    plt.subplot(2, 1, 2)
    plt.imshow(image_dst)
    plt.scatter(target_xy[:, 0], target_xy[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return len(source_xy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KITTI reciprocal match demo using cropping.py correspondences.")
    parser.add_argument("--index-file", "--table", type=Path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--dist-thresh", type=float, default=0.25)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--viz-stride", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_npy_match_cropping_demo/matches.npy"))
    parser.add_argument("--viz-output", type=Path, default=Path("outputs/kitti_npy_match_cropping_demo/matches.jpg"))
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

    matches = find_cropping_correspondences(
        depth_src,
        depth_dst,
        intrinsics_src,
        intrinsics_dst,
        pose_src,
        pose_dst,
        args.min_depth,
        args.dist_thresh,
    )
    matches.update(
        {
            "sequence": sequence,
            "source_frame": frame_src,
            "target_frame": frame_dst,
            "dist_thresh": np.asarray(args.dist_thresh, dtype=np.float32),
            "min_depth": np.asarray(args.min_depth, dtype=np.float32),
            "matching_style": "cropping.extract_correspondences_from_pts3d",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, matches, allow_pickle=True)
    visualized = visualize_matches(image_src, image_dst, matches, args.viz_output, args.viz_stride, args.max_points)

    print(f"sequence: {sequence}")
    print(f"source: {frame_src.get('frame_id')} {image_src_path}")
    print(f"target: {frame_dst.get('frame_id')} {image_dst_path}")
    print(f"matches: {len(matches['source_xy'])}")
    print(f"stats: {matches['stats']}")
    print(f"visualized: {visualized}")
    print(f"saved matches: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
