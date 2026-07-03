from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from unidata_skill.config import DatasetConfig, load_dataset_configs

from .dataset_views import construct_dataset, frame_label, iter_frame_pairs, iter_sequences, jsonable_args, load_pair_views, sanitize
from .features import feature_positives
from .geometry import geometry_positives
from .sampling import PairSkip, empty_positive, make_arrays, union_positives
from .writer import write_json, write_pair


def build_positives(view1: dict, view2: dict, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict, dict[str, dict[str, np.ndarray]]]:
    geometry = empty_positive()
    feature = empty_positive()
    stats = {}
    if args.positive_source in {"geometry", "mixed"}:
        geometry, stats["geometry"] = geometry_positives(view1, view2, args)
    if args.positive_source in {"features", "mixed"}:
        try:
            feature, stats["feature"] = feature_positives(view1, view2, args)
        except PairSkip as exc:
            if args.positive_source == "features":
                raise
            stats["feature"] = {"error": str(exc), "after_filter": 0}
    if args.positive_source == "geometry":
        return geometry, stats, {"geometry": geometry, "feature": feature, "merged": geometry}
    if args.positive_source == "features":
        return feature, stats, {"geometry": geometry, "feature": feature, "merged": feature}
    merged, counts = union_positives(geometry, feature, np.asarray(view1["depthmap"]).shape, np.asarray(view2["depthmap"]).shape)
    stats["union"] = counts
    return merged, stats, {"geometry": geometry, "feature": feature, "merged": merged}


def process_config(config: DatasetConfig, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    dataset = construct_dataset(config, args)
    output_dir = args.output_dir / sanitize(config.label)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = iter_sequences(dataset)
    if not sequences:
        raise RuntimeError(f"dataset '{config.label}' does not expose ordered sequence frames")
    manifest_path = output_dir / "manifest.jsonl"
    skipped: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    planned_pairs = sum(max(0, len(sequence.frames) - args.frame_gap) for sequence in sequences)

    with manifest_path.open("w", encoding="utf-8") as handle:
        pair_bar = tqdm(total=planned_pairs, desc=config.label, unit="pair")
        try:
            for sequence in sequences:
                if len(sequence.frames) < 2:
                    skipped[f"{sequence.sequence_id}:fewer_than_two_frames"] += 1
                    continue
                for source_idx, target_idx in iter_frame_pairs(len(sequence.frames), args.frame_gap):
                    totals["total_pairs"] += 1
                    source_id = frame_label(sequence.frames[source_idx], source_idx)
                    target_id = frame_label(sequence.frames[target_idx], target_idx)
                    try:
                        views = load_pair_views(dataset, sequence, source_idx, target_idx, args)
                        if len(views) != 2:
                            raise PairSkip(f"loaded_pair_view_count:{len(views)}")
                        positives, positive_stats, viz_positives = build_positives(views[0], views[1], args)
                        arrays = make_arrays(positives, views[0], views[1], args, rng)
                        manifest, counts = write_pair(
                            sequence.index,
                            sequence.sequence_id,
                            source_id,
                            target_id,
                            views[0],
                            views[1],
                            arrays,
                            viz_positives,
                            positive_stats,
                            output_dir,
                            args,
                        )
                    except PairSkip as exc:
                        skipped[str(exc)] += 1
                        pair_bar.update(1)
                        pair_bar.set_postfix(success=totals["success_pairs"], skipped=sum(skipped.values()))
                        continue
                    except Exception as exc:
                        skipped[f"load_or_process_pair:{exc}"] += 1
                        pair_bar.update(1)
                        pair_bar.set_postfix(success=totals["success_pairs"], skipped=sum(skipped.values()))
                        continue
                    handle.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
                    totals["success_pairs"] += 1
                    totals["sampled_geometry_positive"] += counts["geometry"]
                    totals["sampled_feature_positive"] += counts["feature"]
                    totals["sampled_both_positive"] += counts["both"]
                    pair_bar.update(1)
                    pair_bar.set_postfix(success=totals["success_pairs"], skipped=sum(skipped.values()))
        finally:
            pair_bar.close()

    summary = {
        "label": config.label,
        "dataset": config.dataset,
        "output_dir": str(output_dir),
        "sequences": int(len(sequences)),
        "planned_pairs": int(planned_pairs),
        "total_pairs": int(totals["total_pairs"]),
        "success_pairs": int(totals["success_pairs"]),
        "skipped_pairs": int(totals["total_pairs"] - totals["success_pairs"]),
        "skip_reasons": dict(sorted(skipped.items())),
        "sampled_geometry_positive": int(totals["sampled_geometry_positive"]),
        "sampled_feature_positive": int(totals["sampled_feature_positive"]),
        "sampled_both_positive": int(totals["sampled_both_positive"]),
        "parameters": jsonable_args(args),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[{config.label}] success pairs: {summary['success_pairs']}")
    print(f"[{config.label}] manifest: {manifest_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build correspondence pairs from UniData Pi3X dataloaders.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/correspondence_dataset"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--resize-views", action="store_true", help="Use Pi3 crop/resize to --width/--height before correspondence extraction.")
    parser.add_argument("--frame-gap", type=int, default=1, help="Fixed frame gap for ordered sequence pairs.")
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
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.n_corres <= 0:
        raise ValueError("--n-corres must be positive")
    if not 0 <= args.nneg < 1:
        raise ValueError("--nneg must be in [0, 1)")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.depth_consistency_thresh <= 0:
        raise ValueError("--depth-consistency-thresh must be positive")
    for key in ("frame_gap", "max_keypoints", "save_stride", "viz_stride", "max_viz_points"):
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
    write_json(args.output_dir / "summary.json", top)
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0 if top["success_pairs"] else 2
