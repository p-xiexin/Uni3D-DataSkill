from __future__ import annotations

import argparse
import json
from typing import Any

from .config import DatasetConfig, load_dataset_configs
from .datasets.arkit_scenes_dataset import ARKitScenesPi3XDataset
from .datasets.blendedmvg_dataset import BlendedMVGDataset
from .datasets.hypersim_dataset import HypersimPi3XDataset
from .datasets.kitti360_dataset import Kitti360Pi3XDataset
from .datasets.kitti_odometry_dataset import KittiOdometryPi3XDataset
from .datasets.nuscenes_dataset import NuScenesPi3XDataset
from .datasets.pi3x_validator import validate_pi3x_dataset
from .datasets.sage_dataset import SagePi3XDataset
from .datasets.uco3d_dataset import UCO3DPi3XDataset
from .datasets.waymo_kitti_dataset import WaymoKittiPi3XDataset
from .datasets.wayve_dataset import WayveScenesPi3XDataset


DATASET_LOADERS = {
    "arkitscenes": {
        "aliases": {"arkitscenes", "arkit-scenes", "arkit"},
        "class": ARKitScenesPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {},
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
    },
    "kitti360": {
        "aliases": {"kitti360", "kitti-360"},
        "class": Kitti360Pi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {
            "cameras": ["image_00"],
        },
        "validation": {
            "frame_num": 8,
            "stride": 5,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "KITTI-360 depthmap is projected sparse depth from Velodyne point clouds; dense depth is not produced online",
    },
    "blendedmvs": {
        "aliases": {"blendedmvs", "blendedmvg"},
        "class": BlendedMVGDataset,
        "root_arg": "data_root",
        "constructor_defaults": {
            "mode": "train",
            "verbose": False,
        },
        "validation": {
            "frame_num": 8,
            "resolution": "768x576",
            "max_samples": 4,
            "batch_size": 1,
        },
    },
    "kitti": {
        "aliases": {"kitti", "kitti-odometry"},
        "class": KittiOdometryPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {
            "cameras": ["image_2"],
        },
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "KITTI dense depth is not read in this direct loader; depthmap is a placeholder unless a derived depth path is attached later",
    },
    "hypersim": {
        "aliases": {"hypersim", "hyper-sim"},
        "class": HypersimPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {},
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "Hypersim depth_meters is ray distance; this loader converts it to planar z-depth using configured intrinsics assumptions",
    },
    "nuscenes": {
        "aliases": {"nuscenes", "nuScenes"},
        "class": NuScenesPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {
            "version": "v1.0-mini",
        },
        "validation": {
            "frame_num": 6,
            "stride": 1,
            "resolution": "512x288",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "nuScenes dense depth is not read in this direct loader; depthmap is a placeholder",
    },
    "sage": {
        "aliases": {"sage", "sage-10k", "sage10k"},
        "class": SagePi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {},
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
    },
    "wayve": {
        "aliases": {"wayve", "wayvescenes", "wayvescenes101"},
        "class": WayveScenesPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {},
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x288",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "WayveScenes101 does not provide GT dense depth; depthmap is a placeholder",
    },
    "uco3d": {
        "aliases": {"uco3d", "uco3d-depth"},
        "class": UCO3DPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {},
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x512",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "uCO3D depth maps are aligned monocular pseudo depth, not GT dense depth",
    },
    "waymo-kitti": {
        "aliases": {"waymo-kitti", "waymo_kitti", "waymo-converted-kitti"},
        "class": WaymoKittiPi3XDataset,
        "root_arg": "data_root",
        "constructor_defaults": {
            "cameras": ["image_2"],
        },
        "validation": {
            "frame_num": 8,
            "stride": 1,
            "resolution": "512x384",
            "max_samples": 4,
            "batch_size": 1,
        },
        "warning": "Waymo native TFRecord/protobuf is not parsed here; this loader targets KITTI-style converted Waymo geometry",
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


def _coerce_dataset_kwargs(spec: dict[str, Any], config: DatasetConfig) -> tuple[dict[str, Any], int, int, int, list[str]]:
    options = {**spec.get("constructor_defaults", {}), **config.options}
    validation = spec.get("validation", {})

    kwargs: dict[str, Any] = {spec["root_arg"]: config.root}
    for key in ("roots", "optional_roots", "list_name"):
        if key in options:
            kwargs[key] = options[key]
    if "sequences" in options:
        kwargs["sequences"] = options["sequences"]
    if "cameras" in options:
        kwargs["cameras"] = tuple(options["cameras"])
    if "mode" in options:
        kwargs["mode"] = options["mode"]
    if "verbose" in options:
        kwargs["verbose"] = bool(options["verbose"])
    if "version" in options:
        kwargs["version"] = options["version"]
    if "scene_dirs" in options:
        kwargs["scene_dirs"] = options["scene_dirs"]
    if "transforms_name" in options:
        kwargs["transforms_name"] = options["transforms_name"]
    if "scan_ids" in options:
        kwargs["scan_ids"] = options["scan_ids"]
    if "splits" in options:
        kwargs["splits"] = tuple(options["splits"])
    if "camera_ids" in options:
        kwargs["camera_ids"] = options["camera_ids"]
    if "fov_x_degrees" in options:
        kwargs["fov_x_degrees"] = float(options["fov_x_degrees"])
    if "subsets" in options:
        kwargs["subsets"] = options["subsets"]
    if "subset_lists_name" in options:
        kwargs["subset_lists_name"] = options["subset_lists_name"]
    if "set_lists_file" in options:
        kwargs["set_lists_file"] = options["set_lists_file"]
    if "pick_sequences" in options:
        kwargs["pick_sequences"] = options["pick_sequences"]
    if "limit_sequences_to" in options:
        kwargs["limit_sequences_to"] = int(options["limit_sequences_to"])
    if "domains" in options:
        kwargs["domains"] = options["domains"]
    if "settings" in options:
        kwargs["settings"] = options["settings"]
    if "route_ids" in options:
        kwargs["route_ids"] = options["route_ids"]

    frame_num = int(validation.get("frame_num", 8))
    max_samples = int(validation.get("max_samples", 4))
    batch_size = int(validation.get("batch_size", 1))
    resolution = _parse_resolution(validation.get("resolution", "512x384"))
    kwargs["frame_num"] = frame_num
    kwargs["resolution"] = resolution
    if "stride" in validation:
        kwargs["stride"] = int(validation["stride"])

    warnings = []
    if spec.get("warning"):
        warnings.append(spec["warning"])
    return kwargs, frame_num, max_samples, batch_size, warnings


def _build_dataset_from_config(config: DatasetConfig):
    spec = _loader_spec(config.dataset)
    kwargs, frame_num, max_samples, batch_size, warnings = _coerce_dataset_kwargs(spec, config)
    dataset_class = spec["class"]
    dataset = dataset_class(**kwargs)
    return dataset, frame_num, max_samples, batch_size, warnings


def _validate_config_entry(config: DatasetConfig):
    dataset, frame_num, max_samples, batch_size, warnings = _build_dataset_from_config(config)
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

    config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a Pi3X dataset from a label/root mapping config.",
    )
    config_parser.add_argument("--config", required=True, help="Dataset mapping config JSON.")
    config_parser.add_argument("--label", help="Dataset label to validate. Defaults to the first config entry.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        configs = load_dataset_configs(args.config)
        selected = configs[0] if args.label is None else next((item for item in configs if item.label == args.label), None)
        if selected is None:
            parser.error(f"label not found in config: {args.label}")
        result = _validate_config_entry(selected)
        report = result.to_dict()
        report["label"] = selected.label
        report["dataset"] = selected.dataset
        report["root"] = selected.root
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if result.status == "ok" else 2

    parser.error(f"unknown command: {args.command}")
    return 2
