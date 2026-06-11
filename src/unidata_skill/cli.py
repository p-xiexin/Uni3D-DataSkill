from __future__ import annotations

import argparse
from typing import Any

from .config import DatasetConfig, load_dataset_configs
from .datasets.arkit_scenes_dataset import ARKitScenesPi3XDataset
from .datasets.blendedmvg_dataset import BlendedMVGDataset
from .datasets.hypersim_dataset import HypersimPi3XDataset
from .datasets.kitti360_dataset import Kitti360Pi3XDataset
from .datasets.kitti_odometry_dataset import KittiOdometryPi3XDataset
from .datasets.nuscenes_dataset import NuScenesPi3XDataset
from .datasets.sage_dataset import SagePi3XDataset
from .datasets.uco3d_dataset import UCO3DPi3XDataset
from .datasets.waymo_kitti_dataset import WaymoKittiPi3XDataset
from .datasets.wayve_dataset import WayveScenesPi3XDataset


DATASET_LOADERS = {
    "arkitscenes": {
        "aliases": {"arkitscenes", "arkit-scenes", "arkit"},
        "class": ARKitScenesPi3XDataset,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "kitti360": {
        "aliases": {"kitti360", "kitti-360"},
        "class": Kitti360Pi3XDataset,
        "defaults": {"cameras": ["image_00"]},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "blendedmvs": {
        "aliases": {"blendedmvs", "blendedmvg"},
        "class": BlendedMVGDataset,
        "defaults": {"mode": "train"},
        "frame_num": 8,
        "resolution": [768, 576],
    },
    "kitti": {
        "aliases": {"kitti", "kitti-odometry"},
        "class": KittiOdometryPi3XDataset,
        "defaults": {"cameras": ["image_2"]},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "hypersim": {
        "aliases": {"hypersim", "hyper-sim"},
        "class": HypersimPi3XDataset,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "nuscenes": {
        "aliases": {"nuscenes", "nuScenes"},
        "class": NuScenesPi3XDataset,
        "defaults": {"version": "v1.0-mini"},
        "frame_num": 6,
        "resolution": [512, 288],
    },
    "sage": {
        "aliases": {"sage", "sage-10k", "sage10k"},
        "class": SagePi3XDataset,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "wayve": {
        "aliases": {"wayve", "wayvescenes", "wayvescenes101"},
        "class": WayveScenesPi3XDataset,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 288],
    },
    "uco3d": {
        "aliases": {"uco3d", "uco3d-depth"},
        "class": UCO3DPi3XDataset,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 512],
    },
    "waymo-kitti": {
        "aliases": {"waymo-kitti", "waymo_kitti", "waymo-converted-kitti"},
        "class": WaymoKittiPi3XDataset,
        "defaults": {"cameras": ["image_2"]},
        "frame_num": 8,
        "resolution": [512, 384],
    },
}


DATASET_OPTION_KEYS = (
    "roots",
    "optional_roots",
    "list_name",
    "sequences",
    "cameras",
    "mode",
    "version",
    "scene_dirs",
    "transforms_name",
    "scan_ids",
    "splits",
    "camera_ids",
    "fov_x_degrees",
    "subsets",
    "subset_lists_name",
    "set_lists_file",
    "pick_sequences",
    "limit_sequences_to",
    "domains",
    "layouts",
    "settings",
    "route_ids",
)


def _loader_spec(dataset: str) -> dict[str, Any]:
    for spec in DATASET_LOADERS.values():
        if dataset in spec["aliases"]:
            return spec
    raise ValueError(f"unsupported dataset '{dataset}'")


def _coerce_dataset_kwargs(spec: dict[str, Any], config: DatasetConfig) -> dict[str, Any]:
    options = {**spec.get("defaults", {}), **config.options}
    resolution = list(spec["resolution"])
    kwargs: dict[str, Any] = {
        "data_root": config.root,
        "verbose": bool(options.get("verbose", True)),
        "frame_num": int(spec["frame_num"]),
        "resolution": [resolution],
    }
    for key in DATASET_OPTION_KEYS:
        if key not in options:
            continue
        value = options[key]
        if key in {"cameras", "splits"}:
            value = tuple(value)
        elif key == "limit_sequences_to":
            value = int(value)
        elif key == "fov_x_degrees":
            value = float(value)
        kwargs[key] = value
    return kwargs


def _build_dataset_from_config(config: DatasetConfig):
    spec = _loader_spec(config.dataset)
    kwargs = _coerce_dataset_kwargs(spec, config)
    return spec["class"](**kwargs)


def _print_sequence_summary(dataset: Any, max_items: int = 20) -> None:
    sequences = list(getattr(dataset, "sequences", []))
    print("num sequences:", len(dataset))
    print("first sequences:", sequences[:max_items])

    num_imgs = getattr(dataset, "num_imgs", None)
    if isinstance(num_imgs, dict):
        print("num_imgs:")
        for sequence in sequences[:max_items]:
            print(f"  {sequence}: {num_imgs.get(sequence, 0)}")


def _probe_dataset(config: DatasetConfig) -> int:
    dataset = _build_dataset_from_config(config)
    print("label:", config.label)
    print("dataset:", config.dataset)
    print("root:", config.root)
    _print_sequence_summary(dataset)

    if len(dataset) == 0:
        raise RuntimeError("No valid sequences found.")
    return 0


def _select_configs(configs: list[DatasetConfig], label: str | None) -> list[DatasetConfig]:
    if label is None:
        return configs
    selected = next((item for item in configs if item.label == label), None)
    if selected is None:
        raise ValueError(f"label not found in config: {label}")
    return [selected]


def _probe_configs(configs: list[DatasetConfig]) -> int:
    status = 0
    for index, config in enumerate(configs):
        if index:
            print("=" * 80)
        try:
            _probe_dataset(config)
        except Exception as exc:
            status = 2
            print("label:", config.label)
            print("dataset:", config.dataset)
            print("root:", config.root)
            print("error:", exc)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unidata-skill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Load dataset entries and print sequence-level information.",
    )
    dataset_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    dataset_parser.add_argument("--label", help="Dataset label to validate. Defaults to every config entry.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-dataset":
        configs = load_dataset_configs(args.config)
        try:
            selected_configs = _select_configs(configs, args.label)
        except ValueError as exc:
            parser.error(str(exc))
        return _probe_configs(selected_configs)

    parser.error(f"unknown command: {args.command}")
    return 2
