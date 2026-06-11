from __future__ import annotations

import argparse
from typing import Any

import numpy as np

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


def _coerce_dataset_kwargs(spec: dict[str, Any], config: DatasetConfig) -> tuple[dict[str, Any], list[int]]:
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
    return kwargs, resolution


def _build_dataset_from_config(config: DatasetConfig):
    spec = _loader_spec(config.dataset)
    kwargs, resolution = _coerce_dataset_kwargs(spec, config)
    return spec["class"](**kwargs), resolution


def _print_array_summary(name: str, value: Any) -> None:
    if not hasattr(value, "shape"):
        print(f"{name}: {type(value)}")
        return
    print(f"{name}: {type(value)} {value.shape} {value.dtype}")


def _print_depth_summary(depthmap: Any) -> None:
    _print_array_summary("depthmap", depthmap)
    if hasattr(depthmap, "min") and hasattr(depthmap, "max"):
        print("depth range:", float(depthmap.min()), float(depthmap.max()))


def _print_view(index: int, view: dict[str, Any]) -> None:
    print("=" * 80)
    print("view:", index)
    for key in ("label", "instance", "image_path", "depth_path", "route_dir"):
        if key in view:
            print(f"{key}:", view[key])
    _print_array_summary("img", view.get("img"))
    _print_depth_summary(view.get("depthmap"))
    _print_array_summary("camera_pose", view.get("camera_pose"))
    _print_array_summary("camera_intrinsics", view.get("camera_intrinsics"))


def _probe_dataset(config: DatasetConfig, index: int, seed: int) -> int:
    dataset, resolution = _build_dataset_from_config(config)
    print("label:", config.label)
    print("dataset:", config.dataset)
    print("root:", config.root)
    print("num sequences:", len(dataset))
    print("first sequences:", getattr(dataset, "sequences", [])[:5])

    if len(dataset) == 0:
        raise RuntimeError("No valid sequences found.")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index out of range: {index}, dataset length: {len(dataset)}")

    rng = np.random.default_rng(seed)
    views = dataset._get_views(index=index, resolution=resolution, rng=rng, is_test=True)
    print("num views:", len(views))
    for view_index, view in enumerate(views):
        _print_view(view_index, view)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unidata-skill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Load one dataset entry and print a verbose sample probe.",
    )
    dataset_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    dataset_parser.add_argument("--label", help="Dataset label to validate. Defaults to the first config entry.")
    dataset_parser.add_argument("--index", type=int, default=0, help="Dataset item index to probe.")
    dataset_parser.add_argument("--seed", type=int, default=0, help="Random seed for view selection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-dataset":
        configs = load_dataset_configs(args.config)
        selected = configs[0] if args.label is None else next((item for item in configs if item.label == args.label), None)
        if selected is None:
            parser.error(f"label not found in config: {args.label}")
        return _probe_dataset(selected, index=args.index, seed=args.seed)

    parser.error(f"unknown command: {args.command}")
    return 2
