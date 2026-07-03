from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_views import as_image_array, sanitize
from .sampling import SOURCE_CODE, SOURCE_NAMES


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
    plt.figure("correspondence_dataset", figsize=(5, 6))
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


def write_pair(
    sequence_index: int,
    sequence_id: str,
    source_id: str,
    target_id: str,
    view1: dict[str, Any],
    view2: dict[str, Any],
    arrays: dict[str, np.ndarray],
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
    visualized = 0 if args.no_visualization else visualize(image1, image2, arrays, viz_path, args)

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
