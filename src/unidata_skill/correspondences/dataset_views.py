from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from unidata_skill.cli import _coerce_dataset_kwargs, _loader_spec
from unidata_skill.config import DatasetConfig


def sanitize(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text).strip("_") or "unknown"


def construct_dataset(config: DatasetConfig, args: argparse.Namespace):
    spec = _loader_spec(config.dataset)
    kwargs = _coerce_dataset_kwargs(spec, config)
    kwargs["frame_num"] = args.views_per_sample
    kwargs["resolution"] = [[args.width, args.height]]
    return spec["class"](**kwargs)


def get_views(dataset: Any, sample_idx: int, args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
    if hasattr(dataset, "_get_views"):
        return dataset._get_views(sample_idx, [args.width, args.height], rng)  # noqa: SLF001
    return dataset[sample_idx]


def iter_view_pairs(views: list[dict[str, Any]], max_gap: int):
    for source_idx in range(len(views)):
        for gap in range(1, max_gap + 1):
            target_idx = source_idx + gap
            if target_idx >= len(views):
                break
            yield source_idx, target_idx


def as_image_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0, 1) * 255
        else:
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    return array[..., :3]


def view_id(view: dict[str, Any], fallback: int) -> str:
    for key in ("prefix", "instance", "image_path", "label"):
        value = view.get(key)
        if value:
            return sanitize(Path(value).stem if key == "image_path" else value)
    return f"{fallback:04d}"


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out

