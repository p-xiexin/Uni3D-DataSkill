from __future__ import annotations

import argparse

import numpy as np


SOURCE_CODE = {"negative": 0, "geometry": 1, "feature": 2, "both": 3}
SOURCE_NAMES = np.asarray(["negative", "geometry", "feature", "both"])


class PairSkip(RuntimeError):
    pass


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
        "corres1": np.asarray(corres1, dtype=np.float32),
        "corres2": np.asarray(corres2, dtype=np.float32),
        "distance_m": distance.astype(np.float32),
        "source_code": np.full(count, SOURCE_CODE[source], dtype=np.int8),
        "feature_score": np.full(count, np.nan, dtype=np.float32) if feature_score is None else feature_score.astype(np.float32),
        "target_depth_error_m": np.full(count, np.nan, dtype=np.float32) if depth_error is None else depth_error.astype(np.float32),
    }


def empty_positive() -> dict[str, np.ndarray]:
    return make_positive(np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), "geometry")


def pixel_to_linear(xy: np.ndarray, width: int) -> np.ndarray:
    xy_round = np.rint(xy).astype(np.int64)
    return xy_round[:, 0] + width * xy_round[:, 1]


def pair_key(corres1: np.ndarray, corres2: np.ndarray, width1: int, width2: int, height2: int) -> np.ndarray:
    return pixel_to_linear(corres1, width1) * np.int64(width2 * height2) + pixel_to_linear(corres2, width2)


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

    _, width1 = depth1_shape
    height2, width2 = depth2_shape
    all_pos = {key: np.concatenate((geometry[key], feature[key]), axis=0) for key in geometry}
    keys = pair_key(all_pos["corres1"], all_pos["corres2"], width1, width2, height2)
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

    keep_array = np.asarray(keep, dtype=np.int64)
    merged = {key: value[keep_array] for key, value in all_pos.items()}
    merged["source_code"] = np.asarray(source_codes, dtype=np.int8)
    counts = {
        "geometry": int((merged["source_code"] == SOURCE_CODE["geometry"]).sum()),
        "feature": int((merged["source_code"] == SOURCE_CODE["feature"]).sum()),
        "both": int((merged["source_code"] == SOURCE_CODE["both"]).sum()),
    }
    return merged, counts


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


def sample_negatives(
    depth1: np.ndarray,
    depth2: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if count == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    height1, width1 = depth1.shape
    height2, width2 = depth2.shape
    valid1 = np.argwhere((depth1 > args.min_depth) & (depth1 <= args.max_depth))
    valid2 = np.argwhere((depth2 > args.min_depth) & (depth2 <= args.max_depth))
    if len(valid1) == 0 or len(valid2) == 0:
        raise PairSkip("no_valid_pixels_for_negatives")
    positive_keys = set(pair_key(pos1, pos2, width1, width2, height2).tolist())
    out1, out2 = [], []
    attempts = 0
    while sum(len(chunk) for chunk in out1) < count and attempts < 50:
        draw = max(count * 4, 1024)
        cand1 = valid1[rng.choice(len(valid1), draw, replace=True)][:, ::-1].astype(np.int32)
        cand2 = valid2[rng.choice(len(valid2), draw, replace=True)][:, ::-1].astype(np.int32)
        keys = pair_key(cand1, cand2, width1, width2, height2)
        keep = np.asarray([int(key) not in positive_keys for key in keys], dtype=bool)
        if keep.any():
            need = count - sum(len(chunk) for chunk in out1)
            out1.append(cand1[keep][:need])
            out2.append(cand2[keep][:need])
        attempts += 1
    if not out1 or sum(len(chunk) for chunk in out1) < count:
        sampled = sum(len(chunk) for chunk in out1)
        raise PairSkip(f"not_enough_negative_matches:{sampled}<{count}")
    return np.concatenate(out1, axis=0)[:count], np.concatenate(out2, axis=0)[:count]


def make_arrays(pos: dict[str, np.ndarray], view1: dict, view2: dict, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_pos = int(args.n_corres * (1.0 - args.nneg))
    n_neg = args.n_corres - n_pos
    if len(pos["corres1"]) < max(args.min_positive, n_pos):
        raise PairSkip(f"positive_matches_below_threshold:{len(pos['corres1'])}<{max(args.min_positive, n_pos)}")
    pick = sample_positive_indices(pos, n_pos, rng)
    pos1 = pos["corres1"][pick]
    pos2 = pos["corres2"][pick]
    neg1, neg2 = sample_negatives(np.asarray(view1["depthmap"]), np.asarray(view2["depthmap"]), pos1, pos2, n_neg, args, rng)

    arrays = {
        "corres1": np.concatenate((pos1, neg1), axis=0).astype(np.float32),
        "corres2": np.concatenate((pos2, neg2), axis=0).astype(np.float32),
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
        if not arrays["valid_corres"][keep].any():
            raise PairSkip(f"save_stride_removed_all_positives:{args.save_stride}")
        arrays = {key: value[keep] for key, value in arrays.items()}
    arrays["tracks"] = np.stack((arrays["corres1"], arrays["corres2"]), axis=0).astype(np.float32)
    arrays["track_positive_mask"] = arrays["valid_corres"].copy()
    arrays["track_vis_mask"] = np.stack((arrays["valid_corres"], arrays["valid_corres"]), axis=0)
    return arrays
