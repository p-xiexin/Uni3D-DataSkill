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


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    out = np.zeros_like(values, dtype=np.float32)
    valid = np.isfinite(values).all(axis=-1, keepdims=True) & (norms > 1e-12)
    np.divide(values, norms, out=out, where=valid)
    return out


def image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gy[1:-1] = 0.5 * (image[2:] - image[:-2])
    gy[0] = image[1] - image[0]
    gy[-1] = image[-1] - image[-2]
    return gx, gy


def bilinear_sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(xy[:, 0], 0.0, width - 1.0)
    y = np.clip(xy[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def iter_project_rays(
    rays_img: np.ndarray,
    pts3d_norm: np.ndarray,
    p_init: np.ndarray,
    max_iter: int,
    lambda_init: float,
    convergence_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rays_img.shape[:2]
    gx_img, gy_img = image_gradient(rays_img)
    p = p_init.astype(np.float32).copy()
    valid = (
        np.isfinite(pts3d_norm).all(axis=-1)
        & (np.linalg.norm(pts3d_norm, axis=-1) > 0.0)
        & np.isfinite(p).all(axis=-1)
    )

    for _ in range(max_iter):
        in_bounds = (p[:, 0] >= 0.0) & (p[:, 0] <= width - 1.0) & (p[:, 1] >= 0.0) & (p[:, 1] <= height - 1.0)
        active = valid & in_bounds
        if not np.any(active):
            break

        rays = bilinear_sample(rays_img, p[active])
        gx = bilinear_sample(gx_img, p[active])
        gy = bilinear_sample(gy_img, p[active])
        residual = rays - pts3d_norm[active]

        jtj_00 = np.sum(gx * gx, axis=1) + lambda_init
        jtj_01 = np.sum(gx * gy, axis=1)
        jtj_11 = np.sum(gy * gy, axis=1) + lambda_init
        jtr_0 = np.sum(gx * residual, axis=1)
        jtr_1 = np.sum(gy * residual, axis=1)
        det = jtj_00 * jtj_11 - jtj_01 * jtj_01
        solvable = np.abs(det) > 1e-12

        delta = np.zeros((int(active.sum()), 2), dtype=np.float32)
        delta[solvable, 0] = (-jtj_11[solvable] * jtr_0[solvable] + jtj_01[solvable] * jtr_1[solvable]) / det[solvable]
        delta[solvable, 1] = (jtj_01[solvable] * jtr_0[solvable] - jtj_00[solvable] * jtr_1[solvable]) / det[solvable]

        active_indices = np.nonzero(active)[0]
        p[active_indices] += delta
        valid[active_indices[~np.isfinite(delta).all(axis=1)]] = False
        if float(np.max(np.linalg.norm(delta[solvable], axis=1), initial=0.0)) < convergence_thresh:
            break

    valid &= (p[:, 0] >= 0.0) & (p[:, 0] <= width - 1.0) & (p[:, 1] >= 0.0) & (p[:, 1] <= height - 1.0)
    return p, valid


def prep_for_iter_proj(
    source_points: np.ndarray,
    target_points_in_source: np.ndarray,
    source_intrinsics: np.ndarray,
    idx_target_to_source_init: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, _ = source_points.shape
    rays_img = normalize_vectors(source_points)
    pts3d_norm = target_points_in_source.reshape(-1, 3)
    pts3d_norm = normalize_vectors(pts3d_norm)

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
    max_iter: int,
    lambda_init: float,
    convergence_thresh: float,
    idx_target_to_source_init: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    height, width, _ = target_points_in_source.shape
    rays_img, pts3d_norm, p_init = prep_for_iter_proj(
        source_points,
        target_points_in_source,
        source_intrinsics,
        idx_target_to_source_init,
    )
    p1_float, valid_ray_proj = iter_project_rays(
        rays_img,
        pts3d_norm,
        p_init,
        max_iter,
        lambda_init,
        convergence_thresh,
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

    valid_proj = valid_ray_proj & in_bounds & finite_target & positive_target
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
        "target_depth_in_source": target_points_valid[valid_dist, 2].astype(np.float32),
        "image_shape": np.asarray([height, width], dtype=np.int32),
    }


def zbuffer_filter(matches: dict[str, np.ndarray], z_key: str = "target_depth_in_source", eps: float = 1e-3) -> dict[str, np.ndarray]:
    source_linear = matches["source_linear"]
    z = matches[z_key]
    if len(source_linear) == 0:
        return matches

    min_z_by_source: dict[int, float] = {}
    for idx, source_idx in enumerate(source_linear):
        source_int = int(source_idx)
        current = min_z_by_source.get(source_int)
        if current is None or float(z[idx]) < current:
            min_z_by_source[source_int] = float(z[idx])

    keep = np.asarray([float(z[idx]) <= min_z_by_source[int(source_idx)] + eps for idx, source_idx in enumerate(source_linear)])
    filtered = matches.copy()
    for key in ("source_xy", "target_xy", "source_linear", "target_linear", "distance_m", "target_depth_in_source"):
        filtered[key] = matches[key][keep]
    return filtered


def bidirectional_filter(forward: dict[str, np.ndarray], reverse: dict[str, np.ndarray], width: int, pixel_tolerance: float) -> dict[str, np.ndarray]:
    reverse_pairs = {int(src): int(dst) for src, dst in zip(reverse["source_linear"], reverse["target_linear"])}
    source_xy = forward["source_xy"].astype(np.float32)
    keep_values = []
    for idx, target_idx in enumerate(forward["target_linear"]):
        reverse_source_idx = reverse_pairs.get(int(target_idx))
        if reverse_source_idx is None:
            keep_values.append(False)
            continue
        reverse_source_xy = lin_to_pixel(np.asarray([reverse_source_idx], dtype=np.int64), width)[0].astype(np.float32)
        keep_values.append(float(np.linalg.norm(source_xy[idx] - reverse_source_xy)) <= pixel_tolerance)
    keep = np.asarray(keep_values, dtype=bool)
    filtered = forward.copy()
    for key in ("source_xy", "target_xy", "source_linear", "target_linear", "distance_m", "target_depth_in_source"):
        filtered[key] = forward[key][keep]
    return filtered


def find_depth_matches_matching_style(
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intrinsics_src: np.ndarray,
    intrinsics_dst: np.ndarray,
    pose_src: np.ndarray,
    pose_dst: np.ndarray,
    dist_thresh: float,
    min_depth: float,
    max_iter: int,
    lambda_init: float,
    convergence_thresh: float,
    bidirectional_px: float,
    zbuffer_eps: float,
) -> dict[str, np.ndarray]:
    source_points = backproject_depth(depth_src, intrinsics_src)
    target_points = backproject_depth(depth_dst, intrinsics_dst)
    target_points_in_source = transform_points(target_points, pose_dst, pose_src)
    forward = match_iterative_proj_without_descriptors(
        source_points,
        target_points_in_source,
        depth_src,
        depth_dst,
        intrinsics_src,
        dist_thresh,
        min_depth,
        max_iter,
        lambda_init,
        convergence_thresh,
    )
    forward = zbuffer_filter(forward, eps=zbuffer_eps)

    source_points_in_target = transform_points(source_points, pose_src, pose_dst)
    reverse = match_iterative_proj_without_descriptors(
        target_points,
        source_points_in_target,
        depth_dst,
        depth_src,
        intrinsics_dst,
        dist_thresh,
        min_depth,
        max_iter,
        lambda_init,
        convergence_thresh,
    )
    reverse = zbuffer_filter(reverse, eps=zbuffer_eps)
    return bidirectional_filter(forward, reverse, depth_src.shape[1], bidirectional_px)


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
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--lambda-init", type=float, default=1e-4)
    parser.add_argument("--convergence-thresh", type=float, default=1e-3)
    parser.add_argument("--bidirectional-px", type=float, default=1.5)
    parser.add_argument("--zbuffer-eps", type=float, default=1e-3)
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
        args.max_iter,
        args.lambda_init,
        args.convergence_thresh,
        args.bidirectional_px,
        args.zbuffer_eps,
    )
    matches.update(
        {
            "sequence": sequence,
            "source_frame": frame_src,
            "target_frame": frame_dst,
            "dist_thresh": np.asarray(args.dist_thresh, dtype=np.float32),
            "min_depth": np.asarray(args.min_depth, dtype=np.float32),
            "max_iter": np.asarray(args.max_iter, dtype=np.int32),
            "lambda_init": np.asarray(args.lambda_init, dtype=np.float32),
            "convergence_thresh": np.asarray(args.convergence_thresh, dtype=np.float32),
            "bidirectional_px": np.asarray(args.bidirectional_px, dtype=np.float32),
            "zbuffer_eps": np.asarray(args.zbuffer_eps, dtype=np.float32),
            "matching_style": "matching.py_iter_proj_without_descriptors_bidirectional_zbuffer",
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
