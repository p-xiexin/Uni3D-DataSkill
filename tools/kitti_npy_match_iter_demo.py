from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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


def pixel_to_lin(pixel: np.ndarray, width: int, subpixel_factor: int = 1) -> np.ndarray:
    return pixel[..., 0] + (width * subpixel_factor * pixel[..., 1])


def lin_to_pixel(index: np.ndarray, width: int) -> np.ndarray:
    u = index % width
    v = index // width
    return np.stack((u, v), axis=-1)


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 0] = (x - intrinsics[0, 2]) / intrinsics[0, 0] * z
    points[..., 1] = (y - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points[..., 2] = z
    return points


def transform_points(points: np.ndarray, pose_src: np.ndarray, pose_dst: np.ndarray) -> np.ndarray:
    height, width, _ = points.shape
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)), axis=1)
    dst_from_src = np.linalg.inv(pose_dst.astype(np.float64)) @ pose_src.astype(np.float64)
    transformed = (dst_from_src @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32).reshape(height, width, 3)


def project_to_pixels(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[..., 2]
    u = intrinsics[0, 0] * (points[..., 0] / z) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (points[..., 1] / z) + intrinsics[1, 2]
    return u, v


def prep_for_iter_proj(
    source_points: np.ndarray,
    target_points_in_source: np.ndarray,
    source_intrinsics: np.ndarray,
    idx_target_to_source_init: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, _ = source_points.shape
    rays_img = source_points / np.linalg.norm(source_points, axis=-1, keepdims=True)
    pts3d_norm = target_points_in_source.reshape(-1, 3)
    pts3d_norm = pts3d_norm / np.linalg.norm(pts3d_norm, axis=-1, keepdims=True)

    if idx_target_to_source_init is None:
        idx_target_to_source_init = np.arange(height * width, dtype=np.int64)
    p_init = lin_to_pixel(idx_target_to_source_init, width).astype(np.float32)

    u, v = project_to_pixels(target_points_in_source, source_intrinsics)
    p_proj = np.stack((u.reshape(-1), v.reshape(-1)), axis=-1)
    return rays_img, pts3d_norm, np.where(np.isfinite(p_proj), p_proj, p_init)


def match_iterative_proj_without_descriptors(
    source_points: np.ndarray,
    target_points_in_source: np.ndarray,
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    dist_thresh: float,
    min_depth: float,
    idx_target_to_source_init: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    height, width, _ = target_points_in_source.shape
    _, _, p1_float = prep_for_iter_proj(
        source_points,
        target_points_in_source,
        source_intrinsics,
        idx_target_to_source_init,
    )
    p1 = np.rint(p1_float).astype(np.int64)

    src_x = p1[:, 0]
    src_y = p1[:, 1]
    target_linear = np.arange(height * width, dtype=np.int64)
    target_y = target_linear // width
    target_x = target_linear % width

    in_bounds = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)
    target_flat = target_points_in_source.reshape(-1, 3)
    finite_target = np.isfinite(target_flat).all(axis=-1)
    positive_target = (target_depth.reshape(-1) > min_depth) & (target_flat[:, 2] > min_depth)

    valid_proj = in_bounds & finite_target & positive_target
    src_x_valid = src_x[valid_proj]
    src_y_valid = src_y[valid_proj]
    target_x_valid = target_x[valid_proj]
    target_y_valid = target_y[valid_proj]
    target_points_valid = target_flat[valid_proj]

    positive_source = source_depth[src_y_valid, src_x_valid] > min_depth
    src_x_valid = src_x_valid[positive_source]
    src_y_valid = src_y_valid[positive_source]
    target_x_valid = target_x_valid[positive_source]
    target_y_valid = target_y_valid[positive_source]
    target_points_valid = target_points_valid[positive_source]

    source_points_valid = source_points[src_y_valid, src_x_valid]
    distances = np.linalg.norm(source_points_valid - target_points_valid, axis=1)
    valid_dist = np.isfinite(distances) & (distances < dist_thresh)

    src_x_valid = src_x_valid[valid_dist]
    src_y_valid = src_y_valid[valid_dist]
    target_x_valid = target_x_valid[valid_dist]
    target_y_valid = target_y_valid[valid_dist]
    distances = distances[valid_dist]

    source_linear = pixel_to_lin(np.stack((src_x_valid, src_y_valid), axis=1), width)
    target_linear = pixel_to_lin(np.stack((target_x_valid, target_y_valid), axis=1), width)
    return {
        "source_xy": np.stack((src_x_valid, src_y_valid), axis=1).astype(np.int32),
        "target_xy": np.stack((target_x_valid, target_y_valid), axis=1).astype(np.int32),
        "source_linear": source_linear.astype(np.int64),
        "target_linear": target_linear.astype(np.int64),
        "distance_m": distances.astype(np.float32),
        "image_shape": np.asarray([height, width], dtype=np.int32),
    }


def find_depth_matches_matching_style(
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intrinsics_src: np.ndarray,
    intrinsics_dst: np.ndarray,
    pose_src: np.ndarray,
    pose_dst: np.ndarray,
    dist_thresh: float,
    min_depth: float,
) -> dict[str, np.ndarray]:
    source_points = backproject_depth(depth_src, intrinsics_src)
    target_points = backproject_depth(depth_dst, intrinsics_dst)
    target_points_in_source = transform_points(target_points, pose_dst, pose_src)
    return match_iterative_proj_without_descriptors(
        source_points,
        target_points_in_source,
        depth_src,
        depth_dst,
        intrinsics_src,
        dist_thresh,
        min_depth,
    )


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
    color_values = matches["target_linear"][::stride]
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
    parser = argparse.ArgumentParser(description="matching.py-style KITTI depth correspondence demo without descriptors.")
    parser.add_argument("--index-file", "--table", type=Path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--dist-thresh", type=float, default=0.25)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--viz-stride", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_npy_match_iter_demo/matches.npy"))
    parser.add_argument("--viz-output", type=Path, default=Path("outputs/kitti_npy_match_iter_demo/matches.jpg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = load_index(args.index_file)
    sequence, frames = select_frames(index, args.sequence)
    frame_src = frames[args.source_frame]
    frame_dst = frames[args.target_frame]

    image_src_path = Path(frame_src["image_path"])
    image_dst_path = Path(frame_dst["image_path"])
    depth_src_path = Path(frame_src["depth_path"])
    depth_dst_path = Path(frame_dst["depth_path"])

    image_src = read_rgb(image_src_path)
    image_dst = read_rgb(image_dst_path)
    depth_src = read_kitti_depth(depth_src_path)
    depth_dst = read_kitti_depth(depth_dst_path)
    intrinsics_src = np.asarray(frame_src["camera_intrinsics"], dtype=np.float32)
    intrinsics_dst = np.asarray(frame_dst["camera_intrinsics"], dtype=np.float32)
    pose_src = np.asarray(frame_src["camera_pose"], dtype=np.float32)
    pose_dst = np.asarray(frame_dst["camera_pose"], dtype=np.float32)

    matches = find_depth_matches_matching_style(
        depth_src,
        depth_dst,
        intrinsics_src,
        intrinsics_dst,
        pose_src,
        pose_dst,
        args.dist_thresh,
        args.min_depth,
    )
    matches.update(
        {
            "sequence": sequence,
            "source_frame": frame_src,
            "target_frame": frame_dst,
            "dist_thresh": np.asarray(args.dist_thresh, dtype=np.float32),
            "min_depth": np.asarray(args.min_depth, dtype=np.float32),
            "matching_style": "matching.py_without_descriptors",
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
    print(f"visualized: {visualized}")
    print(f"saved matches: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
