from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cropping import extract_correspondences_from_pts3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unidata_skill.cli import _coerce_index_kwargs, _loader_spec
from unidata_skill.config import DatasetConfig, load_dataset_configs


PLACEHOLDER_DEPTH_SOURCES = {
    "placeholder_missing_dense_depth",
    "missing",
}


def load_index(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def select_dataset_config(configs: list[DatasetConfig], label: str | None) -> DatasetConfig:
    if label is None:
        if len(configs) != 1:
            labels = ", ".join(config.label for config in configs)
            raise ValueError(f"--label is required when config contains {len(configs)} datasets: {labels}")
        return configs[0]
    for config in configs:
        if config.label == label:
            return config
    raise ValueError(f"label not found in config: {label}")


def resolve_index_file(args: argparse.Namespace) -> Path:
    if args.index_file is not None:
        args.resolved_label = None
        args.resolved_dataset = None
        return args.index_file
    if args.config is None:
        raise ValueError("either --config or --index-file is required")

    config = select_dataset_config(load_dataset_configs(args.config), args.label)
    args.resolved_label = config.label
    args.resolved_dataset = config.dataset
    index_file = config.options.get("index_file")
    if not index_file:
        raise ValueError(f"dataset config '{config.label}' does not define index_file")
    index_path = Path(index_file)
    if index_path.suffix != ".npy":
        raise ValueError(f"index_file must end with .npy: {index_path}")
    if index_path.is_file():
        return index_path

    spec = _loader_spec(config.dataset)
    index_builder = spec.get("index_builder")
    if index_builder is None:
        raise RuntimeError(f"dataset '{config.dataset}' does not define an index builder")
    print(f"index file not found, rebuilding: {index_path}", flush=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_builder(output_path=index_path, **_coerce_index_kwargs(config))
    if not index_path.is_file():
        raise RuntimeError(f"index builder did not create index_file: {index_path}")
    return index_path


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_depth_meters(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key].astype(np.float32)
    else:
        raw_depth = np.asarray(Image.open(path))
        depth = raw_depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if np.issubdtype(raw_depth.dtype, np.integer):
            depth = depth / 256.0
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}: {path}")
    return depth.astype(np.float32)


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


def pixel_to_linear(xy: np.ndarray, width: int) -> np.ndarray:
    return xy[:, 0].astype(np.int64) + width * xy[:, 1].astype(np.int64)


def combined_pair_key(source_linear: np.ndarray, target_linear: np.ndarray, target_size: int) -> np.ndarray:
    return source_linear.astype(np.int64) * np.int64(target_size) + target_linear.astype(np.int64)


def sanitize_path_part(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text).strip("_") or "unknown"


def frame_id(frame: dict[str, Any], fallback: int) -> str:
    for key in ("frame_id", "timestamp", "image_id", "instance"):
        if key in frame:
            return sanitize_path_part(frame[key])
    if "image" in frame:
        return sanitize_path_part(Path(frame["image"]).stem)
    return f"{fallback:08d}"


def is_frame_usable(frame: dict[str, Any]) -> tuple[bool, str | None]:
    depth_source = str(frame.get("depth_source", "")).lower()
    if depth_source in PLACEHOLDER_DEPTH_SOURCES:
        return False, "placeholder_depth"
    for key in ("image", "depth", "camera_intrinsics", "camera_pose"):
        if key not in frame:
            return False, f"missing_{key}"
    if not Path(frame["image"]).is_file():
        return False, "missing_image_file"
    if not Path(frame["depth"]).is_file():
        return False, "missing_depth_file"
    intrinsics = np.asarray(frame["camera_intrinsics"], dtype=np.float32)
    pose = np.asarray(frame["camera_pose"], dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        return False, "invalid_intrinsics"
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False, "invalid_pose"
    return True, None


def find_positive_correspondences(
    depth1: np.ndarray,
    depth2: np.ndarray,
    intrinsics1: np.ndarray,
    intrinsics2: np.ndarray,
    pose1: np.ndarray,
    pose2: np.ndarray,
    min_depth: float,
    dist_thresh: float,
) -> dict[str, np.ndarray]:
    view1 = {
        "pts3d": world_points_from_depth(depth1, intrinsics1, pose1),
        "camera_intrinsics": intrinsics1,
        "camera_pose": pose1,
    }
    view2 = {
        "pts3d": world_points_from_depth(depth2, intrinsics2, pose2),
        "camera_intrinsics": intrinsics2,
        "camera_pose": pose2,
    }
    pos1, pos2 = extract_correspondences_from_pts3d(view1, view2, target_n_corres=None, ret_xy=True)

    h1, w1 = depth1.shape
    h2, w2 = depth2.shape
    x1 = pos1[:, 0].astype(np.int64)
    y1 = pos1[:, 1].astype(np.int64)
    x2 = pos2[:, 0].astype(np.int64)
    y2 = pos2[:, 1].astype(np.int64)

    valid = (
        (x1 >= 0)
        & (x1 < w1)
        & (y1 >= 0)
        & (y1 < h1)
        & (x2 >= 0)
        & (x2 < w2)
        & (y2 >= 0)
        & (y2 < h2)
        & (depth1[y1, x1] > min_depth)
        & (depth2[y2, x2] > min_depth)
    )
    x1 = x1[valid]
    y1 = y1[valid]
    x2 = x2[valid]
    y2 = y2[valid]

    pts1_in_cam2 = camera_points_from_world(view1["pts3d"], pose2)[y1, x1]
    pts2_in_cam2 = camera_points_from_world(view2["pts3d"], pose2)[y2, x2]
    distances = np.linalg.norm(pts1_in_cam2 - pts2_in_cam2, axis=1)
    keep = np.isfinite(distances) & (distances <= dist_thresh)

    corres1 = np.stack((x1[keep], y1[keep]), axis=1).astype(np.int32)
    corres2 = np.stack((x2[keep], y2[keep]), axis=1).astype(np.int32)
    return {
        "corres1": corres1,
        "corres2": corres2,
        "distance_m": distances[keep].astype(np.float32),
        "stats": {
            "reciprocal": int(len(pos1)),
            "valid_depth": int(valid.sum()),
            "after_distance": int(keep.sum()),
        },
    }


def sample_positive_matches(
    positives: dict[str, np.ndarray],
    target_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(positives["corres1"])
    if count < target_count:
        raise ValueError(f"not enough positive matches: {count} < {target_count}")
    indices = rng.choice(count, size=target_count, replace=False)
    return positives["corres1"][indices], positives["corres2"][indices], positives["distance_m"][indices]


def sample_negative_matches(
    depth1: np.ndarray,
    depth2: np.ndarray,
    positive_corres1: np.ndarray,
    positive_corres2: np.ndarray,
    count: int,
    min_depth: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if count == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)

    h1, w1 = depth1.shape
    h2, w2 = depth2.shape
    valid1 = np.argwhere(depth1 > min_depth)
    valid2 = np.argwhere(depth2 > min_depth)
    if len(valid1) == 0 or len(valid2) == 0:
        raise ValueError("no valid pixels for negatives")

    positive_source_linear = pixel_to_linear(positive_corres1, w1)
    positive_target_linear = pixel_to_linear(positive_corres2, w2)
    positive_keys = set(combined_pair_key(positive_source_linear, positive_target_linear, h2 * w2).tolist())

    neg1: list[np.ndarray] = []
    neg2: list[np.ndarray] = []
    attempts = 0
    while sum(len(chunk) for chunk in neg1) < count and attempts < 50:
        need = count - sum(len(chunk) for chunk in neg1)
        draw = max(need * 4, 1024)
        src_yx = valid1[rng.choice(len(valid1), size=draw, replace=True)]
        dst_yx = valid2[rng.choice(len(valid2), size=draw, replace=True)]
        src_xy = src_yx[:, ::-1].astype(np.int32)
        dst_xy = dst_yx[:, ::-1].astype(np.int32)
        src_linear = pixel_to_linear(src_xy, w1)
        dst_linear = pixel_to_linear(dst_xy, w2)
        keys = combined_pair_key(src_linear, dst_linear, h2 * w2)
        keep = np.array([int(key) not in positive_keys for key in keys], dtype=bool)
        if keep.any():
            neg1.append(src_xy[keep][:need])
            neg2.append(dst_xy[keep][:need])
        attempts += 1

    if not neg1:
        raise ValueError("could not sample negatives")
    out1 = np.concatenate(neg1, axis=0)[:count]
    out2 = np.concatenate(neg2, axis=0)[:count]
    if len(out1) < count:
        raise ValueError(f"not enough negative matches: {len(out1)} < {count}")
    return out1.astype(np.int32), out2.astype(np.int32)


def build_mast3r_arrays(
    positives: dict[str, np.ndarray],
    depth1: np.ndarray,
    depth2: np.ndarray,
    n_corres: int,
    nneg: float,
    min_positive: int,
    min_depth: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    target_positive = int(n_corres * (1.0 - nneg))
    target_negative = n_corres - target_positive
    required_positive = max(min_positive, target_positive)
    available_positive = len(positives["corres1"])
    if available_positive < required_positive:
        raise ValueError(f"positive_matches_below_threshold:{available_positive}<{required_positive}")

    pos1, pos2, pos_dist = sample_positive_matches(positives, target_positive, rng)
    neg1, neg2 = sample_negative_matches(depth1, depth2, pos1, pos2, target_negative, min_depth, rng)

    corres1 = np.concatenate((pos1, neg1), axis=0).astype(np.int32)
    corres2 = np.concatenate((pos2, neg2), axis=0).astype(np.int32)
    valid_corres = np.concatenate(
        (np.ones(target_positive, dtype=bool), np.zeros(target_negative, dtype=bool)),
        axis=0,
    )
    distance_m = np.concatenate(
        (pos_dist.astype(np.float32), np.full(target_negative, np.nan, dtype=np.float32)),
        axis=0,
    )

    perm = rng.permutation(n_corres)
    return {
        "corres1": corres1[perm],
        "corres2": corres2[perm],
        "valid_corres": valid_corres[perm],
        "distance_m": distance_m[perm],
    }


def visualize_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    corres1: np.ndarray,
    corres2: np.ndarray,
    valid_corres: np.ndarray,
    output_path: Path,
    stride: int,
    max_points: int,
) -> int:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive1 = corres1[valid_corres][::stride]
    positive2 = corres2[valid_corres][::stride]
    color_values = pixel_to_linear(positive1, image1.shape[1])
    if len(positive1) > max_points:
        pick = np.linspace(0, len(positive1) - 1, max_points).astype(np.int64)
        positive1 = positive1[pick]
        positive2 = positive2[pick]
        color_values = color_values[pick]

    plt.figure("mast3r_correspondences", figsize=[5, 6])
    plt.subplot(2, 1, 1)
    plt.imshow(image1)
    if len(positive1):
        plt.scatter(positive1[:, 0], positive1[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)

    plt.subplot(2, 1, 2)
    plt.imshow(image2)
    if len(positive2):
        plt.scatter(positive2[:, 0], positive2[:, 1], s=0.7, c=color_values, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close("all")
    return int(len(positive1))


def iter_sequence_pairs(frames: list[dict[str, Any]], max_gap: int):
    for source_idx in range(len(frames)):
        for gap in range(1, max_gap + 1):
            target_idx = source_idx + gap
            if target_idx >= len(frames):
                break
            yield source_idx, target_idx


def relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def write_jsonl_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def count_sequence_pairs(num_frames: int, max_gap: int) -> int:
    total = 0
    for source_idx in range(num_frames):
        total += max(0, min(max_gap, num_frames - source_idx - 1))
    return total


def iter_selected_pairs(records: list[dict[str, Any]], max_gap: int):
    for record in records:
        sequence_id = str(record.get("sequence_id", "sequence"))
        frames = list(record.get("frames", []))
        for source_idx, target_idx in iter_sequence_pairs(frames, max_gap):
            yield sequence_id, frames, source_idx, target_idx


def process_pair(
    sequence_id: str,
    frames: list[dict[str, Any]],
    source_idx: int,
    target_idx: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    frame1 = frames[source_idx]
    frame2 = frames[target_idx]
    ok1, reason1 = is_frame_usable(frame1)
    ok2, reason2 = is_frame_usable(frame2)
    if not ok1:
        return None, f"source_{reason1}", None
    if not ok2:
        return None, f"target_{reason2}", None

    source_id = frame_id(frame1, source_idx)
    target_id = frame_id(frame2, target_idx)
    pair_name = f"{source_id}__{target_id}"
    sequence_part = sanitize_path_part(sequence_id)
    pair_path = args.output_dir / "pairs" / sequence_part / f"{pair_name}.npz"
    viz_path = args.output_dir / "visualizations" / sequence_part / f"{pair_name}.jpg"

    image1_path = Path(frame1["image"])
    image2_path = Path(frame2["image"])
    depth1 = read_depth_meters(Path(frame1["depth"]))
    depth2 = read_depth_meters(Path(frame2["depth"]))
    intrinsics1 = np.asarray(frame1["camera_intrinsics"], dtype=np.float32)
    intrinsics2 = np.asarray(frame2["camera_intrinsics"], dtype=np.float32)
    pose1 = np.asarray(frame1["camera_pose"], dtype=np.float32)
    pose2 = np.asarray(frame2["camera_pose"], dtype=np.float32)

    positives = find_positive_correspondences(
        depth1,
        depth2,
        intrinsics1,
        intrinsics2,
        pose1,
        pose2,
        args.min_depth,
        args.dist_thresh,
    )
    arrays = build_mast3r_arrays(
        positives,
        depth1,
        depth2,
        args.n_corres,
        args.nneg,
        args.min_positive,
        args.min_depth,
        rng,
    )

    image1 = read_rgb(image1_path)
    image2 = read_rgb(image2_path)
    visualized = visualize_matches(
        image1,
        image2,
        arrays["corres1"],
        arrays["corres2"],
        arrays["valid_corres"],
        viz_path,
        args.viz_stride,
        args.max_viz_points,
    )

    pair_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pair_path,
        **arrays,
        sequence_id=np.asarray(sequence_id),
        source_frame_id=np.asarray(source_id),
        target_frame_id=np.asarray(target_id),
        source_image=np.asarray(str(image1_path)),
        target_image=np.asarray(str(image2_path)),
        image_shape1=np.asarray(depth1.shape, dtype=np.int32),
        image_shape2=np.asarray(depth2.shape, dtype=np.int32),
        n_corres=np.asarray(args.n_corres, dtype=np.int32),
        nneg=np.asarray(args.nneg, dtype=np.float32),
        dist_thresh=np.asarray(args.dist_thresh, dtype=np.float32),
        min_depth=np.asarray(args.min_depth, dtype=np.float32),
    )

    num_positive = int(arrays["valid_corres"].sum())
    num_negative = int(len(arrays["valid_corres"]) - num_positive)
    manifest = {
        "pair_path": relative_to_output(pair_path, args.output_dir),
        "viz_path": relative_to_output(viz_path, args.output_dir),
        "num_corres": int(args.n_corres),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "source_image": str(image1_path),
        "target_image": str(image2_path),
        "sequence_id": sequence_id,
        "source_frame_id": source_id,
        "target_frame_id": target_id,
        "positive_stats": positives["stats"],
        "visualized": visualized,
    }
    quality = {
        "available_positive": int(len(positives["corres1"])),
        "sampled_positive": num_positive,
        "sampled_negative": num_negative,
        "visualized": visualized,
    }
    return manifest, None, quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MAST3R-style correspondence pairs from an indexed RGB-D dataset.")
    parser.add_argument("--config", type=Path, help="Dataset config JSON. The selected entry must define index_file.")
    parser.add_argument("--label", help="Dataset label in --config. Required when config contains multiple datasets.")
    parser.add_argument("--index-file", type=Path, help="Direct .npy index path. Overrides --config/--label.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to outputs/mast3r_correspondences/<label-or-index-stem>.")
    parser.add_argument("--sequence", default=None, help="Optional sequence_id to process. Defaults to every sequence.")
    parser.add_argument("--n-corres", type=int, default=8192)
    parser.add_argument("--nneg", type=float, default=0.5)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--dist-thresh", type=float, default=0.25)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-positive", type=int, default=1)
    parser.add_argument("--viz-stride", type=int, default=50)
    parser.add_argument("--max-viz-points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_corres <= 0:
        raise ValueError("--n-corres must be positive")
    if not 0 <= args.nneg < 1:
        raise ValueError("--nneg must be in [0, 1)")
    if args.max_gap <= 0:
        raise ValueError("--max-gap must be positive")
    if args.viz_stride <= 0:
        raise ValueError("--viz-stride must be positive")
    if args.max_viz_points <= 0:
        raise ValueError("--max-viz-points must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    index_file = resolve_index_file(args)
    if args.output_dir is None:
        output_name = args.resolved_label or index_file.stem
        args.output_dir = Path("outputs") / "mast3r_correspondences" / sanitize_path_part(output_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(index_file)
    records = list(index.get("sequences", []))
    if args.sequence is not None:
        records = [record for record in records if str(record.get("sequence_id")) == args.sequence]
    if not records:
        raise RuntimeError("no sequences selected")

    planned_pairs = sum(count_sequence_pairs(len(list(record.get("frames", []))), args.max_gap) for record in records)

    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "summary.json"
    skip_reasons: Counter[str] = Counter()
    positive_counts: list[int] = []
    total_pairs = 0
    success_pairs = 0
    visualizations_saved = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        pair_iter = iter_selected_pairs(records, args.max_gap)
        pair_iter = tqdm(pair_iter, total=planned_pairs, unit="pair", desc="building correspondences")
        for sequence_id, frames, source_idx, target_idx in pair_iter:
            total_pairs += 1
            try:
                manifest, skip_reason, quality = process_pair(sequence_id, frames, source_idx, target_idx, args, rng)
            except Exception as exc:
                manifest = None
                quality = None
                skip_reason = type(exc).__name__ + ":" + str(exc)
            if manifest is None:
                skip_reasons[str(skip_reason)] += 1
                pair_iter.set_postfix(success=success_pairs, skipped=total_pairs - success_pairs)
                continue
            write_jsonl_record(manifest_file, manifest)
            success_pairs += 1
            visualizations_saved += 1
            if quality is not None:
                positive_counts.append(int(quality["available_positive"]))
            pair_iter.set_postfix(success=success_pairs, skipped=total_pairs - success_pairs)

    summary = {
        "config": str(args.config) if args.config is not None else None,
        "label": args.resolved_label,
        "dataset": args.resolved_dataset,
        "index_file": str(index_file),
        "output_dir": str(args.output_dir),
        "total_pairs": total_pairs,
        "success_pairs": success_pairs,
        "skipped_pairs": total_pairs - success_pairs,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "positive_count_min": int(min(positive_counts)) if positive_counts else 0,
        "positive_count_max": int(max(positive_counts)) if positive_counts else 0,
        "positive_count_mean": float(np.mean(positive_counts)) if positive_counts else 0.0,
        "visualizations_saved": visualizations_saved,
        "parameters": {
            "n_corres": args.n_corres,
            "nneg": args.nneg,
            "max_gap": args.max_gap,
            "dist_thresh": args.dist_thresh,
            "min_depth": args.min_depth,
            "min_positive": args.min_positive,
            "viz_stride": args.viz_stride,
            "max_viz_points": args.max_viz_points,
            "seed": args.seed,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"total pairs: {total_pairs}")
    print(f"success pairs: {success_pairs}")
    print(f"visualizations saved: {visualizations_saved}")
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")
    return 0 if success_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
