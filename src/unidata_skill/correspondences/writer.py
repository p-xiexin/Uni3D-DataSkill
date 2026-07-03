from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_views import as_image_array, sanitize
from .sampling import SOURCE_CODE, SOURCE_NAMES


def select_viz_points(pos1: np.ndarray, pos2: np.ndarray, codes: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos1 = pos1[:: args.viz_stride]
    pos2 = pos2[:: args.viz_stride]
    codes = codes[:: args.viz_stride]
    if len(pos1) > args.max_viz_points:
        pick = np.linspace(0, len(pos1) - 1, args.max_viz_points).astype(np.int64)
        pos1, pos2, codes = pos1[pick], pos2[pick], codes[pick]
    return pos1, pos2, codes


def draw_crosses(axis, xy: np.ndarray, color_values: np.ndarray, size: float = 6.0, cmap: str = "hsv") -> None:  # noqa: ANN001
    if len(xy) == 0:
        return
    x = xy[:, 0]
    y = xy[:, 1]
    span = size / 2.0
    norm = color_values / max(float(color_values.max()), 1.0)
    import matplotlib.pyplot as plt

    colors = plt.get_cmap(cmap)(norm)
    axis.hlines(y, x - span, x + span, colors=colors, linewidth=1.2)
    axis.vlines(x, y - span, y + span, colors=colors, linewidth=1.2)


def draw_source_aware_points(axis, xy: np.ndarray, codes: np.ndarray, color_values: np.ndarray) -> None:  # noqa: ANN001
    geometry = codes == SOURCE_CODE["geometry"]
    feature = codes == SOURCE_CODE["feature"]
    both = codes == SOURCE_CODE["both"]
    if geometry.any():
        axis.scatter(xy[geometry, 0], xy[geometry, 1], s=0.7, c=color_values[geometry], cmap="jet")
    if feature.any():
        draw_crosses(axis, xy[feature], color_values[feature], size=6.0, cmap="hsv")
    if both.any():
        axis.scatter(xy[both, 0], xy[both, 1], s=2.0, c=color_values[both], cmap="jet")
        draw_crosses(axis, xy[both], color_values[both], size=7.0, cmap="hsv")


def positives_to_viz_arrays(viz_positives: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = viz_positives.get("merged", {})
    if len(source.get("corres1", [])) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int8)
    return select_viz_points(source["corres1"], source["corres2"], source["source_code"], args)


def visualize(image1: np.ndarray, image2: np.ndarray, viz_positives: dict[str, dict[str, np.ndarray]], path: Path, args: argparse.Namespace) -> int:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos1, pos2, codes = positives_to_viz_arrays(viz_positives, args)
    colors = pos1[:, 0].astype(np.float32) + image1.shape[1] * pos1[:, 1].astype(np.float32) if len(pos1) else np.empty((0,), dtype=np.float32)
    plt.figure("correspondence_dataset", figsize=(5, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(image1)
    if len(pos1):
        draw_source_aware_points(ax1, pos1, codes, colors)
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(image2)
    if len(pos2):
        draw_source_aware_points(ax2, pos2, codes, colors)
    ax2.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close("all")
    return int(len(pos1))


def write_pair(
    sequence_index: int,
    sequence_id: str,
    source_id: str,
    target_id: str,
    view1: dict[str, Any],
    view2: dict[str, Any],
    arrays: dict[str, np.ndarray],
    viz_positives: dict[str, dict[str, np.ndarray]],
    positive_stats: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    sequence_part = sanitize(sequence_id)
    source_part = sanitize(source_id)
    target_part = sanitize(target_id)
    pair_name = f"{sequence_index:06d}_{sequence_part}__{source_part}__{target_part}"
    pair_path = output_dir / "pairs" / sequence_part / f"{pair_name}.npz"
    viz_path = output_dir / "visualizations" / sequence_part / f"{pair_name}.jpg"
    image1 = as_image_array(view1["img"])
    image2 = as_image_array(view2["img"])
    visualized = 0 if args.no_visualization else visualize(image1, image2, viz_positives, viz_path, args)

    pair_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pair_path,
        **arrays,
        sequence_id=np.asarray(sequence_part),
        source_frame_id=np.asarray(source_part),
        target_frame_id=np.asarray(target_part),
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
        "sequence_id": sequence_part,
        "source_frame_id": source_part,
        "target_frame_id": target_part,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
