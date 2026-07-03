from __future__ import annotations

import numpy as np


SOURCE_CODE = {"geom": 1, "feat": 2, "both": 3}
SOURCE_NAMES = np.asarray(["unused", "geom", "feat", "both"])


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
    return make_positive(np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), "geom")


def stride_positive(pos: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    if stride <= 1 or len(pos["corres1"]) == 0:
        return pos
    return {key: value[::stride] for key, value in pos.items()}


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
        return feature, {"geom": 0, "feat": int(len(feature["corres1"])), "both": 0}
    if len(feature["corres1"]) == 0:
        return geometry, {"geom": int(len(geometry["corres1"])), "feat": 0, "both": 0}

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
        has_geo = np.any(codes == SOURCE_CODE["geom"])
        has_feat = np.any(codes == SOURCE_CODE["feat"])
        chosen = indices[-1] if has_feat else indices[0]
        keep.append(chosen)
        source_codes.append(SOURCE_CODE["both"] if has_geo and has_feat else int(all_pos["source_code"][chosen]))

    keep_array = np.asarray(keep, dtype=np.int64)
    merged = {key: value[keep_array] for key, value in all_pos.items()}
    merged["source_code"] = np.asarray(source_codes, dtype=np.int8)
    counts = {
        "geom": int((merged["source_code"] == SOURCE_CODE["geom"]).sum()),
        "feat": int((merged["source_code"] == SOURCE_CODE["feat"]).sum()),
        "both": int((merged["source_code"] == SOURCE_CODE["both"]).sum()),
    }
    return merged, counts


def make_arrays(pos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(pos["corres1"])
    if count == 0:
        raise PairSkip("no_correspondences")
    arrays = {
        "corres1": pos["corres1"].astype(np.float32),
        "corres2": pos["corres2"].astype(np.float32),
        "distance_m": pos["distance_m"].astype(np.float32),
        "source_code": pos["source_code"].astype(np.int8),
        "feat_score": pos["feature_score"].astype(np.float32),
        "depth_err": pos["target_depth_error_m"].astype(np.float32),
    }
    arrays["tracks"] = np.stack((arrays["corres1"], arrays["corres2"]), axis=0).astype(np.float32)
    return arrays
