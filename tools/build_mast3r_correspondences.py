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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unidata_skill.cli import _coerce_dataset_kwargs, _loader_spec
from unidata_skill.config import DatasetConfig, load_dataset_configs


SOURCE_CODE = {"negative": 0, "geometry": 1, "feature": 2, "both": 3}
SOURCE_NAMES = np.asarray(["negative", "geometry", "feature", "both"])


class PairSkip(RuntimeError):
    pass


def sanitize(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text).strip("_") or "unknown"


def construct_dataset(config: DatasetConfig, args: argparse.Namespace):
    spec = _loader_spec(config.dataset)
    kwargs = _coerce_dataset_kwargs(spec, config)
    kwargs["frame_num"] = args.views_per_sample
    kwargs["resolution"] = [[args.width, args.height]]
    return spec["class"](**kwargs)


def get_views(dataset: Any, sample_idx: int, args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    if hasattr(dataset, "_get_views"):
        return dataset._get_views(sample_idx, [args.width, args.height], rng)  # noqa: SLF001
    return dataset[sample_idx]


def as_image_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1) * 255 if np.issubdtype(array.dtype, np.floating) else np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    return array[..., :3]


def view_id(view: dict[str, Any], fallback: int) -> str:
    for key in ("prefix", "instance", "image_path", "label"):
        value = view.get(key)
        if value:
            return sanitize(Path(value).stem if key == "image_path" else value)
    return f"{fallback:04d}"


def pixel_to_linear(xy: np.ndarray, width: int) -> np.ndarray:
    return xy[:, 0].astype(np.int64) + width * xy[:, 1].astype(np.int64)


def pair_key(corres1: np.ndarray, corres2: np.ndarray, width1: int, width2: int, height2: int) -> np.ndarray:
    return pixel_to_linear(corres1, width1) * np.int64(width2 * height2) + pixel_to_linear(corres2, width2)


def world_points(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float64)
    z = depth.astype(np.float64)
    xyz = np.stack(
        (
            (x - intrinsics[0, 2]) / intrinsics[0, 0] * z,
            (y - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
            np.ones_like(z),
        ),
        axis=-1,
    )
    points = (pose.astype(np.float64) @ xyz.reshape(-1, 4).T).T[:, :3]
    return points.reshape(height, width, 3).astype(np.float32)


def project_world(points_world: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = points_world.reshape(-1, 3).astype(np.float64)
    homog = np.concatenate((flat, np.ones((len(flat), 1), dtype=np.float64)), axis=1)
    cam = (np.linalg.inv(pose.astype(np.float64)) @ homog.T).T[:, :3]
    z = cam[:, 2]
    xy = np.empty((len(cam), 2), dtype=np.float64)
    xy[:, 0] = intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2]
    xy[:, 1] = intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]
    return xy.reshape(*points_world.shape[:2], 2), z.reshape(points_world.shape[:2])


def make_positive(
    corres1: np.ndarray,
    corres2: np.ndarray,
    distance: np.ndarray,
    source: str,
    feature_score: np.ndarray | None = None,
    depth_error: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    count = len(corres1)
    return {
        "corres1": corres1.astype(np.int32),
        "corres2": corres2.astype(np.int32),
        "distance_m": distance.astype(np.float32),
        "source_code": np.full(count, SOURCE_CODE[source], dtype=np.int8),
        "feature_score": np.full(count, np.nan, dtype=np.float32) if feature_score is None else feature_score.astype(np.float32),
        "target_depth_error_m": np.full(count, np.nan, dtype=np.float32) if depth_error is None else depth_error.astype(np.float32),
    }


def empty_positive() -> dict[str, np.ndarray]:
    return make_positive(np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), "geometry")


def geometry_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth1 = np.asarray(view1["depthmap"], dtype=np.float32)
    depth2 = np.asarray(view2["depthmap"], dtype=np.float32)
    k1 = np.asarray(view1["camera_intrinsics"], dtype=np.float32)
    k2 = np.asarray(view2["camera_intrinsics"], dtype=np.float32)
    pose1 = np.asarray(view1["camera_pose"], dtype=np.float32)
    pose2 = np.asarray(view2["camera_pose"], dtype=np.float32)
    h1, w1 = depth1.shape
    h2, w2 = depth2.shape

    pts1 = world_points(depth1, k1, pose1)
    xy2_float, z2_projected = project_world(pts1, k2, pose2)
    y1, x1 = np.indices((h1, w1), dtype=np.int64)
    src = np.stack((x1.reshape(-1), y1.reshape(-1)), axis=1)
    dst = np.rint(xy2_float.reshape(-1, 2)).astype(np.int64)
    z2_projected = z2_projected.reshape(-1)

    inside = (dst[:, 0] >= 0) & (dst[:, 0] < w2) & (dst[:, 1] >= 0) & (dst[:, 1] < h2)
    src_depth = depth1.reshape(-1)
    safe_x2 = np.clip(dst[:, 0], 0, w2 - 1)
    safe_y2 = np.clip(dst[:, 1], 0, h2 - 1)
    target_depth = depth2[safe_y2, safe_x2]
    depth_error = np.abs(z2_projected - target_depth)
    keep = (
        inside
        & np.isfinite(src_depth)
        & np.isfinite(target_depth)
        & np.isfinite(z2_projected)
        & (src_depth > args.min_depth)
        & (target_depth > args.min_depth)
        & (z2_projected > args.min_depth)
        & (src_depth <= args.max_depth)
        & (target_depth <= args.max_depth)
        & (z2_projected <= args.max_depth)
        & (depth_error <= args.depth_consistency_thresh)
    )
    positives = make_positive(src[keep], dst[keep], depth_error[keep], "geometry", depth_error=depth_error[keep])
    return positives, {"raw": int(len(src)), "after_filter": int(keep.sum())}


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
    scores = feats.get("keypoint_scores") or feats.get("scores")
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

    src = np.rint(xy1).astype(np.int64)
    inside1 = (src[:, 0] >= 0) & (src[:, 0] < w1) & (src[:, 1] >= 0) & (src[:, 1] < h1)
    safe_x1 = np.clip(src[:, 0], 0, w1 - 1)
    safe_y1 = np.clip(src[:, 1], 0, h1 - 1)
    z1 = depth1[safe_y1, safe_x1].astype(np.float64)
    cam1 = np.stack(((xy1[:, 0] - k1[0, 2]) / k1[0, 0] * z1, (xy1[:, 1] - k1[1, 2]) / k1[1, 1] * z1, z1, np.ones_like(z1)), axis=1)
    world = (pose1 @ cam1.T).T
    cam2 = (np.linalg.inv(pose2) @ world.T).T[:, :3]
    z2 = cam2[:, 2]
    xy2 = np.empty((len(xy1), 2), dtype=np.float64)
    xy2[:, 0] = k2[0, 0] * cam2[:, 0] / z2 + k2[0, 2]
    xy2[:, 1] = k2[1, 1] * cam2[:, 1] / z2 + k2[1, 2]
    dst = np.rint(xy2).astype(np.int64)

    inside2 = (dst[:, 0] >= 0) & (dst[:, 0] < w2) & (dst[:, 1] >= 0) & (dst[:, 1] < h2)
    safe_x2 = np.clip(dst[:, 0], 0, w2 - 1)
    safe_y2 = np.clip(dst[:, 1], 0, h2 - 1)
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
    positives = make_positive(src[keep], dst[keep], depth_error[keep], "feature", feature_score=score[keep], depth_error=depth_error[keep])
    return positives, {"method": method, "raw": int(len(xy1)), "after_filter": int(keep.sum())}


def union_positives(
    geometry: dict[str, np.ndarray],
    feature: dict[str, np.ndarray],
    depth1_shape: tuple[int, int],
    depth2_shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if len(geometry["corres1"]) == 0:
        return feature, {"geometry": 0, "feature": int(len(feature["corres1"])), "both": 0}
    if len(feature["corres1"]) == 0:
        return geometry, {"geometry": int(len(geometry["corres1"])), "feature": 0, "both": 0}

    h1, w1 = depth1_shape
    h2, w2 = depth2_shape
    all_pos = {key: np.concatenate((geometry[key], feature[key]), axis=0) for key in geometry}
    keys = pair_key(all_pos["corres1"], all_pos["corres2"], w1, w2, h2)
    groups: dict[int, list[int]] = {}
    for index, key in enumerate(keys.tolist()):
        groups.setdefault(int(key), []).append(index)

    keep = []
    source_codes = []
    for indices in groups.values():
        codes = all_pos["source_code"][indices]
        has_geo = np.any(codes == SOURCE_CODE["geometry"])
        has_feat = np.any(codes == SOURCE_CODE["feature"])
        chosen = indices[-1] if has_feat else indices[0]
        keep.append(chosen)
        source_codes.append(SOURCE_CODE["both"] if has_geo and has_feat else int(all_pos["source_code"][chosen]))

    keep = np.asarray(keep, dtype=np.int64)
    merged = {key: value[keep] for key, value in all_pos.items()}
    merged["source_code"] = np.asarray(source_codes, dtype=np.int8)
    counts = {
        "geometry": int((merged["source_code"] == SOURCE_CODE["geometry"]).sum()),
        "feature": int((merged["source_code"] == SOURCE_CODE["feature"]).sum()),
        "both": int((merged["source_code"] == SOURCE_CODE["both"]).sum()),
    }
    return merged, counts


def build_positives(view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    geom = empty_positive()
    feat = empty_positive()
    stats: dict[str, Any] = {}
    if args.positive_source in {"geometry", "mixed"}:
        geom, stats["geometry"] = geometry_positives(view1, view2, args)
    if args.positive_source in {"features", "mixed"}:
        try:
            feat, stats["feature"] = feature_positives(view1, view2, args)
        except PairSkip as exc:
            if args.positive_source == "features":
                raise
            stats["feature"] = {"error": str(exc), "after_filter": 0}
    if args.positive_source == "geometry":
        return geom, stats
    if args.positive_source == "features":
        return feat, stats
    merged, counts = union_positives(geom, feat, np.asarray(view1["depthmap"]).shape, np.asarray(view2["depthmap"]).shape)
    stats["union"] = counts
    return merged, stats


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def sample_positive_indices(pos: dict[str, np.ndarray], target: int, rng: np.random.Generator) -> np.ndarray:
    count = len(pos["corres1"])
    if count < target:
        raise PairSkip(f"positive_matches_below_threshold:{count}<{target}")
    codes = pos["source_code"]
    feature_related = np.flatnonzero((codes == SOURCE_CODE["feature"]) | (codes == SOURCE_CODE["both"]))
    geometry_only = np.flatnonzero(codes == SOURCE_CODE["geometry"])
    if len(feature_related) and len(geometry_only):
        n_feat = min(len(feature_related), max(1, target // 2))
        n_geo = target - n_feat
        if len(geometry_only) < n_geo:
            n_feat += n_geo - len(geometry_only)
            n_geo = len(geometry_only)
        picks = np.concatenate((rng.choice(feature_related, n_feat, replace=False), rng.choice(geometry_only, n_geo, replace=False)))
        return rng.permutation(picks)
    return rng.choice(count, target, replace=False)


def sample_negatives(depth1: np.ndarray, depth2: np.ndarray, pos1: np.ndarray, pos2: np.ndarray, count: int, args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if count == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    h1, w1 = depth1.shape
    h2, w2 = depth2.shape
    valid1 = np.argwhere((depth1 > args.min_depth) & (depth1 <= args.max_depth))
    valid2 = np.argwhere((depth2 > args.min_depth) & (depth2 <= args.max_depth))
    if len(valid1) == 0 or len(valid2) == 0:
        raise PairSkip("no_valid_pixels_for_negatives")
    positive_keys = set(pair_key(pos1, pos2, w1, w2, h2).tolist())
    out1, out2 = [], []
    while sum(len(chunk) for chunk in out1) < count:
        draw = max(count * 4, 1024)
        cand1 = valid1[rng.choice(len(valid1), draw, replace=True)][:, ::-1].astype(np.int32)
        cand2 = valid2[rng.choice(len(valid2), draw, replace=True)][:, ::-1].astype(np.int32)
        keys = pair_key(cand1, cand2, w1, w2, h2)
        keep = np.asarray([int(key) not in positive_keys for key in keys], dtype=bool)
        if keep.any():
            need = count - sum(len(chunk) for chunk in out1)
            out1.append(cand1[keep][:need])
            out2.append(cand2[keep][:need])
    return np.concatenate(out1, axis=0)[:count], np.concatenate(out2, axis=0)[:count]


def make_arrays(pos: dict[str, np.ndarray], view1: dict[str, Any], view2: dict[str, Any], args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_pos = int(args.n_corres * (1.0 - args.nneg))
    n_neg = args.n_corres - n_pos
    if len(pos["corres1"]) < max(args.min_positive, n_pos):
        raise PairSkip(f"positive_matches_below_threshold:{len(pos['corres1'])}<{max(args.min_positive, n_pos)}")
    pick = sample_positive_indices(pos, n_pos, rng)
    pos1 = pos["corres1"][pick]
    pos2 = pos["corres2"][pick]
    neg1, neg2 = sample_negatives(np.asarray(view1["depthmap"]), np.asarray(view2["depthmap"]), pos1, pos2, n_neg, args, rng)

    arrays = {
        "corres1": np.concatenate((pos1, neg1), axis=0).astype(np.int32),
        "corres2": np.concatenate((pos2, neg2), axis=0).astype(np.int32),
        "valid_corres": np.concatenate((np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool))),
        "distance_m": np.concatenate((pos["distance_m"][pick], np.full(n_neg, np.nan, dtype=np.float32))),
        "positive_source_code": np.concatenate((pos["source_code"][pick], np.full(n_neg, SOURCE_CODE["negative"], dtype=np.int8))),
        "feature_score": np.concatenate((pos["feature_score"][pick], np.full(n_neg, np.nan, dtype=np.float32))),
        "target_depth_error_m": np.concatenate((pos["target_depth_error_m"][pick], np.full(n_neg, np.nan, dtype=np.float32))),
    }
    perm = rng.permutation(args.n_corres)
    arrays = {key: value[perm] for key, value in arrays.items()}
    if args.save_stride > 1:
        valid = arrays["valid_corres"]
        keep = np.sort(np.concatenate((np.flatnonzero(valid)[:: args.save_stride], np.flatnonzero(~valid)[:: args.save_stride])))
        arrays = {key: value[keep] for key, value in arrays.items()}
    arrays["tracks"] = np.stack((arrays["corres1"], arrays["corres2"]), axis=0).astype(np.float32)
    arrays["track_positive_mask"] = arrays["valid_corres"].copy()
    arrays["track_vis_mask"] = np.stack((arrays["valid_corres"], arrays["valid_corres"]), axis=0)
    return arrays


def visualize(image1: np.ndarray, image2: np.ndarray, arrays: dict[str, np.ndarray], path: Path, args: argparse.Namespace) -> int:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos1 = arrays["corres1"][arrays["valid_corres"]][:: args.viz_stride]
    pos2 = arrays["corres2"][arrays["valid_corres"]][:: args.viz_stride]
    if len(pos1) > args.max_viz_points:
        pick = np.linspace(0, len(pos1) - 1, args.max_viz_points).astype(np.int64)
        pos1, pos2 = pos1[pick], pos2[pick]
    colors = np.arange(len(pos1))
    plt.figure("mast3r_correspondences", figsize=(5, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(image1)
    if len(pos1):
        plt.scatter(pos1[:, 0], pos1[:, 1], s=0.7, c=colors, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    plt.subplot(2, 1, 2)
    plt.imshow(image2)
    if len(pos2):
        plt.scatter(pos2[:, 0], pos2[:, 1], s=0.7, c=colors, cmap="jet")
    plt.gca().tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close("all")
    return int(len(pos1))


def iter_pairs(views: list[dict[str, Any]], max_gap: int):
    for i in range(len(views)):
        for gap in range(1, max_gap + 1):
            j = i + gap
            if j >= len(views):
                break
            yield i, j


def write_pair(sample_idx: int, i: int, j: int, view1: dict[str, Any], view2: dict[str, Any], output_dir: Path, args: argparse.Namespace, rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, int]]:
    positives, positive_stats = build_positives(view1, view2, args)
    arrays = make_arrays(positives, view1, view2, args, rng)
    source_id = view_id(view1, i)
    target_id = view_id(view2, j)
    sequence_id = sanitize(view1.get("label", f"sample_{sample_idx:06d}"))
    pair_name = f"{sample_idx:06d}_{source_id}__{target_id}"
    pair_path = output_dir / "pairs" / sequence_id / f"{pair_name}.npz"
    viz_path = output_dir / "visualizations" / sequence_id / f"{pair_name}.jpg"
    image1 = as_image_array(view1["img"])
    image2 = as_image_array(view2["img"])
    visualized = 0 if args.no_visualization else visualize(image1, image2, arrays, viz_path, args)

    pair_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pair_path,
        **arrays,
        sequence_id=np.asarray(sequence_id),
        source_frame_id=np.asarray(source_id),
        target_frame_id=np.asarray(target_id),
        source_image=np.asarray(str(view1.get("image_path", ""))),
        target_image=np.asarray(str(view2.get("image_path", ""))),
        image_paths=np.asarray([str(view1.get("image_path", "")), str(view2.get("image_path", ""))]),
        image_shape1=np.asarray(np.asarray(view1["depthmap"]).shape, dtype=np.int32),
        image_shape2=np.asarray(np.asarray(view2["depthmap"]).shape, dtype=np.int32),
        n_corres=np.asarray(len(arrays["valid_corres"]), dtype=np.int32),
        requested_n_corres=np.asarray(args.n_corres, dtype=np.int32),
        positive_source=np.asarray(args.positive_source),
        positive_source_code_names=SOURCE_NAMES,
        save_stride=np.asarray(args.save_stride, dtype=np.int32),
    )

    codes = arrays["positive_source_code"][arrays["valid_corres"]]
    counts = {
        "geometry": int((codes == SOURCE_CODE["geometry"]).sum()),
        "feature": int((codes == SOURCE_CODE["feature"]).sum()),
        "both": int((codes == SOURCE_CODE["both"]).sum()),
    }
    manifest = {
        "pair_path": str(pair_path.relative_to(output_dir)),
        "viz_path": None if args.no_visualization else str(viz_path.relative_to(output_dir)),
        "sequence_id": sequence_id,
        "source_frame_id": source_id,
        "target_frame_id": target_id,
        "source_image": str(view1.get("image_path", "")),
        "target_image": str(view2.get("image_path", "")),
        "num_corres": int(len(arrays["valid_corres"])),
        "requested_num_corres": int(args.n_corres),
        "num_positive": int(arrays["valid_corres"].sum()),
        "num_negative": int((~arrays["valid_corres"]).sum()),
        "num_geometry_positive": counts["geometry"],
        "num_feature_positive": counts["feature"],
        "num_both_positive": counts["both"],
        "positive_stats": positive_stats,
        "visualized": visualized,
    }
    return manifest, counts


def process_config(config: DatasetConfig, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    dataset = construct_dataset(config, args)
    output_dir = args.output_dir / sanitize(config.label)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    manifest_path = output_dir / "manifest.jsonl"
    skipped: Counter[str] = Counter()
    totals = Counter()

    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample_idx in tqdm(range(limit), desc=config.label, unit="sample"):
            try:
                views = get_views(dataset, sample_idx, args, rng)
            except Exception as exc:
                skipped[f"load_sample:{exc}"] += 1
                continue
            if len(views) < 2:
                skipped["fewer_than_two_views"] += 1
                continue
            for i, j in iter_pairs(views, args.max_gap):
                totals["total_pairs"] += 1
                try:
                    manifest, counts = write_pair(sample_idx, i, j, views[i], views[j], output_dir, args, rng)
                except PairSkip as exc:
                    skipped[str(exc)] += 1
                    continue
                handle.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
                totals["success_pairs"] += 1
                totals["sampled_geometry_positive"] += counts["geometry"]
                totals["sampled_feature_positive"] += counts["feature"]
                totals["sampled_both_positive"] += counts["both"]

    summary = {
        "label": config.label,
        "dataset": config.dataset,
        "output_dir": str(output_dir),
        "samples": int(limit),
        "total_pairs": int(totals["total_pairs"]),
        "success_pairs": int(totals["success_pairs"]),
        "skipped_pairs": int(totals["total_pairs"] - totals["success_pairs"]),
        "skip_reasons": dict(sorted(skipped.items())),
        "sampled_geometry_positive": int(totals["sampled_geometry_positive"]),
        "sampled_feature_positive": int(totals["sampled_feature_positive"]),
        "sampled_both_positive": int(totals["sampled_both_positive"]),
        "parameters": jsonable_args(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"[{config.label}] success pairs: {summary['success_pairs']}")
    print(f"[{config.label}] manifest: {manifest_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MAST3R-style pairs from UniData Pi3X dataloaders.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mast3r_correspondences"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--views-per-sample", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--positive-source", choices=["geometry", "features", "mixed"], default="mixed")
    parser.add_argument("--n-corres", type=int, default=8192)
    parser.add_argument("--nneg", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--depth-consistency-thresh", type=float, default=0.25)
    parser.add_argument("--feature-method", choices=["sift", "aliked", "superpoint", "sp", "lightglue_sift"], default="sift")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--detection-threshold", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-positive", type=int, default=1)
    parser.add_argument("--save-stride", type=int, default=1)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--viz-stride", type=int, default=50)
    parser.add_argument("--max-viz-points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.views_per_sample < 2:
        raise ValueError("--views-per-sample must be at least 2")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.n_corres <= 0:
        raise ValueError("--n-corres must be positive")
    if not 0 <= args.nneg < 1:
        raise ValueError("--nneg must be in [0, 1)")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    for key in ("max_gap", "max_keypoints", "save_stride", "viz_stride", "max_viz_points"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    configs = load_dataset_configs(args.config)
    if args.label is not None:
        configs = [config for config in configs if config.label == args.label]
    if not configs:
        raise RuntimeError("no dataset configs selected")
    summaries = [process_config(config, args, rng) for config in configs]
    top = {
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "datasets": summaries,
        "success_pairs": int(sum(item["success_pairs"] for item in summaries)),
        "total_pairs": int(sum(item["total_pairs"] for item in summaries)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(top, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0 if top["success_pairs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
