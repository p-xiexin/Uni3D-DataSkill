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
    if not records:
        raise ValueError("index contains no sequences")
    if sequence is None:
        record = next((item for item in records if item.get("frames")), records[0])
    else:
        record = next((item for item in records if item.get("sequence_id") == sequence), None)
        if record is None:
            raise ValueError(f"sequence not found: {sequence}")
    frames = list(record.get("frames", []))
    if not frames:
        raise ValueError(f"sequence has no frames: {record.get('sequence_id')}")
    return str(record.get("sequence_id", "")), frames


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_kitti_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 256.0


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 0] = (x - cx) / fx * z
    points[..., 1] = (y - cy) / fy * z
    points[..., 2] = z
    return points


def transform_points(points_src: np.ndarray, pose_src: np.ndarray, pose_dst: np.ndarray) -> np.ndarray:
    height, width, _ = points_src.shape
    flat = points_src.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)), axis=1)
    dst_from_src = np.linalg.inv(pose_dst.astype(np.float64)) @ pose_src.astype(np.float64)
    transformed = (dst_from_src @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32).reshape(height, width, 3)


def project_points(points_dst: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_dst[..., 2]
    u = intrinsics[0, 0] * (points_dst[..., 0] / z) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (points_dst[..., 1] / z) + intrinsics[1, 2]
    return u, v


def find_depth_matches(
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intrinsics_src: np.ndarray,
    intrinsics_dst: np.ndarray,
    pose_src: np.ndarray,
    pose_dst: np.ndarray,
    dist_thresh: float,
    min_depth: float,
) -> dict[str, np.ndarray]:
    if depth_src.shape != depth_dst.shape:
        raise ValueError(f"source and target depth shapes must match, got {depth_src.shape} and {depth_dst.shape}")

    height, width = depth_src.shape
    src_points = backproject_depth(depth_src, intrinsics_src)
    dst_points = backproject_depth(depth_dst, intrinsics_dst)
    src_points_in_dst = transform_points(src_points, pose_src, pose_dst)
    u, v = project_points(src_points_in_dst, intrinsics_dst)
    u_round = np.rint(u).astype(np.int64)
    v_round = np.rint(v).astype(np.int64)

    finite = np.isfinite(src_points_in_dst).all(axis=-1) & np.isfinite(u) & np.isfinite(v)
    in_bounds = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)
    valid_src = depth_src > min_depth
    valid_projected = finite & in_bounds & valid_src & (src_points_in_dst[..., 2] > min_depth)

    src_y, src_x = np.nonzero(valid_projected)
    dst_x = u_round[src_y, src_x]
    dst_y = v_round[src_y, src_x]
    valid_dst = depth_dst[dst_y, dst_x] > min_depth

    src_x = src_x[valid_dst]
    src_y = src_y[valid_dst]
    dst_x = dst_x[valid_dst]
    dst_y = dst_y[valid_dst]

    transformed = src_points_in_dst[src_y, src_x]
    observed = dst_points[dst_y, dst_x]
    distances = np.linalg.norm(transformed - observed, axis=1)
    keep = np.isfinite(distances) & (distances <= dist_thresh)

    src_x = src_x[keep]
    src_y = src_y[keep]
    dst_x = dst_x[keep]
    dst_y = dst_y[keep]
    distances = distances[keep]

    return {
        "source_xy": np.stack((src_x, src_y), axis=1).astype(np.int32),
        "target_xy": np.stack((dst_x, dst_y), axis=1).astype(np.int32),
        "source_linear": (src_y * width + src_x).astype(np.int64),
        "target_linear": (dst_y * width + dst_x).astype(np.int64),
        "distance_m": distances.astype(np.float32),
        "image_shape": np.asarray([height, width], dtype=np.int32),
    }


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

    src_xy = matches["source_xy"][::stride]
    dst_xy = matches["target_xy"][::stride]
    color_values = matches["source_linear"][::stride]
    if len(src_xy) > max_points:
        pick = np.linspace(0, len(src_xy) - 1, max_points).astype(np.int64)
        src_xy = src_xy[pick]
        dst_xy = dst_xy[pick]
        color_values = color_values[pick]

    plt.figure("1", figsize=[5, 6])
    plt.subplot(2, 1, 1)
    plt.imshow(image_src)
    plt.scatter(src_xy[:, 0], src_xy[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)

    plt.subplot(2, 1, 2)
    plt.imshow(image_dst)
    plt.scatter(dst_xy[:, 0], dst_xy[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return len(src_xy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo same-point matching from a KITTI UniData npy index.")
    parser.add_argument("--index-file", "--table", type=Path, required=True, help="KITTI index .npy generated by reindex-dataset.")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame", type=int, default=0, help="Frame ordinal inside the selected sequence.")
    parser.add_argument("--target-frame", type=int, default=1, help="Frame ordinal inside the selected sequence.")
    parser.add_argument("--dist-thresh", type=float, default=0.25, help="3D Euclidean distance threshold in meters.")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--viz-stride", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_npy_match_demo/matches.npy"))
    parser.add_argument("--viz-output", type=Path, default=Path("outputs/kitti_npy_match_demo/matches.jpg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.viz_stride < 1:
        raise ValueError("--viz-stride must be >= 1")
    if args.max_points < 1:
        raise ValueError("--max-points must be >= 1")

    index = load_index(args.index_file)
    sequence, frames = select_frames(index, args.sequence)
    if args.source_frame >= len(frames) or args.target_frame >= len(frames):
        raise IndexError(f"selected sequence has {len(frames)} frames")

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

    matches = find_depth_matches(
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
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, matches, allow_pickle=True)
    drawn = visualize_matches(image_src, image_dst, matches, args.viz_output, args.viz_stride, args.max_points)

    print(f"sequence: {sequence}")
    print(f"source: {frame_src.get('frame_id')} {image_src_path}")
    print(f"target: {frame_dst.get('frame_id')} {image_dst_path}")
    print(f"matches: {len(matches['source_xy'])}")
    print(f"visualized: {drawn}")
    print(f"saved matches: {args.output}")
    print(f"saved visualization: {args.viz_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
