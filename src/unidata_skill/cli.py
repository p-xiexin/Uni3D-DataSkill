from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import DatasetConfig, load_dataset_configs
from .datasets.aria_synthetic_environments_dataset import AriaSyntheticEnvironmentsPi3XDataset, generate_ase_index
from .datasets.arkit_scenes_dataset import ARKitScenesPi3XDataset, generate_arkit_scenes_index
from .datasets.blendedmvg_dataset import BlendedMVGDataset
from .datasets.hypersim_dataset import HypersimPi3XDataset, generate_hypersim_index
from .datasets.kitti360_dataset import Kitti360Pi3XDataset, generate_kitti360_index
from .datasets.kitti_odometry_dataset import KittiOdometryPi3XDataset, generate_kitti_odometry_index
from .datasets.nuscenes_dataset import NuScenesPi3XDataset, generate_nuscenes_index
from .datasets.sage_dataset import SagePi3XDataset, generate_sage_index
from .datasets.uco3d_dataset import UCO3DPi3XDataset
from .datasets.waymo_kitti_dataset import WaymoKittiPi3XDataset, generate_waymo_kitti_index
from .datasets.wayve_dataset import WayveScenesPi3XDataset, generate_wayve_index


DATASET_LOADERS = {
    "ase": {
        "aliases": {"ase", "aria-synthetic-environments", "aria_synthetic_environments"},
        "class": AriaSyntheticEnvironmentsPi3XDataset,
        "index_builder": generate_ase_index,
        "defaults": {"fov_x_degrees": 90.0},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "arkitscenes": {
        "aliases": {"arkitscenes", "arkit-scenes", "arkit"},
        "class": ARKitScenesPi3XDataset,
        "index_builder": generate_arkit_scenes_index,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "kitti360": {
        "aliases": {"kitti360", "kitti-360"},
        "class": Kitti360Pi3XDataset,
        "index_builder": generate_kitti360_index,
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
        "index_builder": generate_kitti_odometry_index,
        "defaults": {"cameras": ["image_2"]},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "hypersim": {
        "aliases": {"hypersim", "hyper-sim"},
        "class": HypersimPi3XDataset,
        "index_builder": generate_hypersim_index,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "nuscenes": {
        "aliases": {"nuscenes", "nuScenes"},
        "class": NuScenesPi3XDataset,
        "index_builder": generate_nuscenes_index,
        "defaults": {"version": "v1.0-mini"},
        "frame_num": 6,
        "resolution": [512, 288],
    },
    "sage": {
        "aliases": {"sage", "sage-10k", "sage10k"},
        "class": SagePi3XDataset,
        "index_builder": generate_sage_index,
        "defaults": {},
        "frame_num": 8,
        "resolution": [512, 384],
    },
    "wayve": {
        "aliases": {"wayve", "wayvescenes", "wayvescenes101"},
        "class": WayveScenesPi3XDataset,
        "index_builder": generate_wayve_index,
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
        "index_builder": generate_waymo_kitti_index,
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
    "index_file",
)


INDEX_OPTION_KEYS = (
    "roots",
    "optional_roots",
    "sequences",
    "cameras",
    "version",
    "scene_dirs",
    "transforms_name",
    "scan_ids",
    "splits",
    "camera_ids",
    "fov_x_degrees",
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
        "verbose": bool(options.get("verbose", False)),
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
        elif key == "index_file" and Path(value).suffix != ".npy":
            raise ValueError(f"index_file must end with .npy: {value}")
        kwargs[key] = value
    return kwargs


def _build_dataset_from_config(config: DatasetConfig):
    spec = _loader_spec(config.dataset)
    kwargs = _coerce_dataset_kwargs(spec, config)
    return spec["class"](**kwargs)


def _coerce_index_kwargs(config: DatasetConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"data_root": config.root}
    for key in INDEX_OPTION_KEYS:
        if key in config.options:
            kwargs[key] = config.options[key]
    return kwargs


def _print_sequence_summary(dataset: Any) -> None:
    sequences = list(getattr(dataset, "sequences", []))
    print("sequences:", len(dataset))
    if sequences:
        print("first sequence:", sequences[0])

    num_imgs = getattr(dataset, "num_imgs", None)
    if isinstance(num_imgs, dict):
        counts = [int(num_imgs.get(sequence, 0)) for sequence in sequences]
        if counts:
            print("frames:", sum(counts), f"(min={min(counts)}, max={max(counts)})")


def _print_dataset_header(config: DatasetConfig) -> None:
    print("label:", config.label)
    print("dataset:", config.dataset)
    print("root:", config.root)


def _sample_dataset(config: DatasetConfig) -> int:
    dataset = _build_dataset_from_config(config)
    _print_dataset_header(config)
    _print_sequence_summary(dataset)

    if len(dataset) == 0:
        raise RuntimeError("No valid sequences found.")

    views = dataset[0]
    print("sample:", f"index=0 views={len(views)}")
    if views:
        first_view = views[0]
        print("sample label:", first_view.get("label"))
    return 0


def _reindex_dataset(config: DatasetConfig) -> int:
    spec = _loader_spec(config.dataset)
    index_file = config.options.get("index_file")
    if not index_file:
        _print_dataset_header(config)
        print("skip: no index_file")
        return 0
    if Path(index_file).suffix != ".npy":
        raise ValueError(f"index_file must end with .npy: {index_file}")
    index_builder = spec.get("index_builder")
    if index_builder is None:
        raise RuntimeError(f"dataset '{config.dataset}' does not define an index builder")

    index = index_builder(output_path=index_file, **_coerce_index_kwargs(config))
    _print_dataset_header(config)
    print("index_file:", index_file)
    print("num indexed sequences:", len(index.get("sequences", [])))
    return 0


def _select_configs(configs: list[DatasetConfig], label: str | None) -> list[DatasetConfig]:
    if label is None:
        return configs
    selected = next((item for item in configs if item.label == label), None)
    if selected is None:
        raise ValueError(f"label not found in config: {label}")
    return [selected]


def _run_configs(configs: list[DatasetConfig], action) -> int:
    status = 0
    for index, config in enumerate(configs):
        if index:
            print("=" * 80)
        try:
            action(config)
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

    reindex_parser = subparsers.add_parser(
        "reindex-dataset",
        help="Rebuild dataset index files from configured raw dataset roots.",
    )
    reindex_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    reindex_parser.add_argument("--label", help="Dataset label to reindex. Defaults to every config entry.")

    sample_parser = subparsers.add_parser(
        "sample-dataset",
        help="Load dataset entries and run one sampling probe.",
    )
    sample_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    sample_parser.add_argument("--label", help="Dataset label to sample. Defaults to every config entry.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"reindex-dataset", "sample-dataset"}:
        configs = load_dataset_configs(args.config)
        try:
            selected_configs = _select_configs(configs, args.label)
        except ValueError as exc:
            parser.error(str(exc))
        if args.command == "reindex-dataset":
            return _run_configs(selected_configs, _reindex_dataset)
        return _run_configs(selected_configs, _sample_dataset)

    parser.error(f"unknown command: {args.command}")
    return 2
