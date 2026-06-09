from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .config import DatasetConfig, load_dataset_configs
from .datasets.pi3x_validator import validate_pi3x_dataset


DATASET_LOADERS = {
    "kitti360": {
        "aliases": {"kitti360", "kitti-360"},
        "module": "unidata_skill.datasets.kitti360_dataset",
        "class": "Kitti360Pi3XDataset",
        "root_arg": "kitti360_root",
        "default_resolution": "512x384",
        "defaults": {
            "cameras": ["image_00"],
            "frame_num": 8,
            "stride": 5,
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "dense depth is not available in the first KITTI-360 workflow; depthmap is a placeholder",
    },
    "blendedmvs": {
        "aliases": {"blendedmvs", "blendedmvg"},
        "module": "unidata_skill.datasets.blendedmvg_dataset",
        "class": "BlendedMVGDataset",
        "root_arg": "data_root",
        "default_resolution": "768x576",
        "defaults": {
            "mode": "train",
            "frame_num": 8,
            "max_samples": 4,
            "batch_size": 1,
            "verbose": False,
        },
    },
}


def _parse_resolution(value: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    width, height = (int(item) for item in value.lower().split("x", 1))
    return width, height


def _loader_spec(dataset: str) -> dict[str, Any]:
    for spec in DATASET_LOADERS.values():
        if dataset in spec["aliases"]:
            return spec
    raise ValueError(f"unsupported dataset '{dataset}'")


def _add_pi3_root(pi3_root: str | None) -> None:
    if not pi3_root:
        return
    root = str(Path(pi3_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _coerce_dataset_kwargs(spec: dict[str, Any], config: DatasetConfig, pi3_root: str | None = None) -> tuple[dict[str, Any], int, int, list[str]]:
    options = {**spec.get("defaults", {}), **config.options}
    effective_pi3_root = pi3_root or options.get("pi3_root")
    _add_pi3_root(effective_pi3_root)

    kwargs: dict[str, Any] = {spec["root_arg"]: config.root}
    if effective_pi3_root and spec["root_arg"] == "kitti360_root":
        kwargs["pi3_root"] = effective_pi3_root
    if "sequences" in options:
        kwargs["sequences"] = options["sequences"]
    if "cameras" in options:
        kwargs["cameras"] = tuple(options["cameras"])
    if "mode" in options:
        kwargs["mode"] = options["mode"]
    if "stride" in options:
        kwargs["stride"] = int(options["stride"])
    if "verbose" in options:
        kwargs["verbose"] = bool(options["verbose"])

    frame_num = int(options.get("frame_num", 8))
    max_samples = int(options.get("max_samples", 4))
    batch_size = int(options.get("batch_size", 1))
    resolution = _parse_resolution(options.get("resolution", spec["default_resolution"]))
    kwargs["frame_num"] = frame_num
    kwargs["resolution"] = resolution

    warnings = []
    if spec.get("warning"):
        warnings.append(spec["warning"])
    return kwargs, frame_num, max_samples, batch_size, warnings


def _build_dataset_from_config(config: DatasetConfig, pi3_root: str | None = None):
    spec = _loader_spec(config.dataset)
    kwargs, frame_num, max_samples, batch_size, warnings = _coerce_dataset_kwargs(spec, config, pi3_root=pi3_root)
    module = importlib.import_module(spec["module"])
    dataset_class = getattr(module, spec["class"])
    dataset = dataset_class(**kwargs)
    return dataset, frame_num, max_samples, batch_size, warnings


def _validate_config_entry(config: DatasetConfig, pi3_root: str | None = None):
    dataset, frame_num, max_samples, batch_size, warnings = _build_dataset_from_config(config, pi3_root=pi3_root)
    return validate_pi3x_dataset(
        dataset,
        expected_frame_num=frame_num,
        max_samples=max_samples,
        batch_size=batch_size,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unidata-skill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-kitti360-pi3x",
        help="Validate direct KITTI-360 loading through a Pi3X-compatible dataset.",
    )
    validate_parser.add_argument("--kitti360-root", required=True, help="KITTI-360 dataset root.")
    validate_parser.add_argument("--pi3-root", help="Local yyfz/Pi3 training-branch checkout.")
    validate_parser.add_argument("--sequence", action="append", dest="sequences", help="Sequence to include. Can be repeated.")
    validate_parser.add_argument("--camera", action="append", dest="cameras", choices=["image_00", "image_01"], help="Camera to include. Can be repeated.")
    validate_parser.add_argument("--frame-num", type=int, default=8)
    validate_parser.add_argument("--stride", type=int, default=5)
    validate_parser.add_argument("--resolution", default="512x384", help="Width x height, for example 512x384.")
    validate_parser.add_argument("--max-samples", type=int, default=4)
    validate_parser.add_argument("--batch-size", type=int, default=1)

    config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a Pi3X dataset from a label/root mapping config.",
    )
    config_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    config_parser.add_argument("--label", help="Dataset label to validate. Defaults to the first config entry.")
    config_parser.add_argument("--pi3-root", help="Override local yyfz/Pi3 training-branch checkout.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-kitti360-pi3x":
        result = _validate_config_entry(
            DatasetConfig(
                label="kitti360_cli",
                dataset="kitti360",
                root=args.kitti360_root,
                options={
                    "pi3_root": args.pi3_root,
                    "sequences": args.sequences,
                    "cameras": args.cameras or ["image_00"],
                    "frame_num": args.frame_num,
                    "stride": args.stride,
                    "resolution": args.resolution,
                    "max_samples": args.max_samples,
                    "batch_size": args.batch_size,
                },
            ),
            pi3_root=args.pi3_root,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.status == "ok" else 2

    if args.command == "validate-config":
        configs = load_dataset_configs(args.config)
        selected = configs[0] if args.label is None else next((item for item in configs if item.label == args.label), None)
        if selected is None:
            parser.error(f"label not found in config: {args.label}")
        result = _validate_config_entry(selected, pi3_root=args.pi3_root)
        report = result.to_dict()
        report["label"] = selected.label
        report["dataset"] = selected.dataset
        report["root"] = selected.root
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if result.status == "ok" else 2

    parser.error(f"unknown command: {args.command}")
    return 2
