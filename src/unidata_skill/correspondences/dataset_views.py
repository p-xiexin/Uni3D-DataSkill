from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    kwargs["frame_num"] = 2
    kwargs["resolution"] = [[args.width, args.height]]
    return spec["class"](**kwargs)


@dataclass(frozen=True)
class SequenceFrames:
    index: int
    sequence_id: str
    frames: list[dict[str, Any]]
    source: str


class OrderedPairRng:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):  # noqa: ANN001
        if isinstance(a, int) and a == 2 and size == 2 and not replace:
            return np.asarray([0, 1], dtype=np.int64)
        return self._rng.choice(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)

    def integers(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self._rng.integers(*args, **kwargs)


def iter_sequences(dataset: Any) -> list[SequenceFrames]:
    if hasattr(dataset, "records") and hasattr(dataset, "frames"):
        out = []
        for index, record in enumerate(dataset.records):
            sequence_id = str(record.get("sequence_id", index))
            out.append(SequenceFrames(index, sequence_id, list(dataset.frames.get(sequence_id, [])), "frames"))
        return out
    if hasattr(dataset, "sequences") and hasattr(dataset, "frames"):
        out = []
        for index, sequence_id in enumerate(dataset.sequences):
            sequence_text = str(sequence_id)
            out.append(SequenceFrames(index, sequence_text, list(dataset.frames.get(sequence_text, [])), "frames"))
        return out
    if hasattr(dataset, "routes"):
        out = []
        for index, route in enumerate(dataset.routes):
            sequence_id = str(route.get("sequence_id", index))
            out.append(SequenceFrames(index, sequence_id, list(route.get("frames", [])), "routes"))
        return out
    return []


def load_pair_views(
    dataset: Any,
    sequence: SequenceFrames,
    source_frame_idx: int,
    target_frame_idx: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not hasattr(dataset, "_get_views"):
        raise TypeError("strict sequence pair loading requires a dataset with _get_views")
    old_frame_num = getattr(dataset, "frame_num", None)
    if sequence.source == "routes":
        old_frames = dataset.routes[sequence.index]["frames"]
        dataset.routes[sequence.index]["frames"] = [sequence.frames[source_frame_idx], sequence.frames[target_frame_idx]]
    else:
        old_frames = dataset.frames[sequence.sequence_id]
        dataset.frames[sequence.sequence_id] = [sequence.frames[source_frame_idx], sequence.frames[target_frame_idx]]
    dataset.frame_num = 2
    try:
        return dataset._get_views(  # noqa: SLF001
            sequence.index,
            [args.width, args.height],
            OrderedPairRng(args.seed + source_frame_idx * 1000003 + target_frame_idx),
        )
    finally:
        if sequence.source == "routes":
            dataset.routes[sequence.index]["frames"] = old_frames
        else:
            dataset.frames[sequence.sequence_id] = old_frames
        if old_frame_num is not None:
            dataset.frame_num = old_frame_num


def iter_frame_pairs(frame_count: int, frame_gap: int):
    for source_idx in range(frame_count):
        target_idx = source_idx + frame_gap
        if target_idx >= frame_count:
            break
        yield source_idx, target_idx


def frame_label(frame: dict[str, Any], fallback: int) -> str:
    parts = []
    for key in ("camera_id", "channel", "camera", "sensor"):
        if frame.get(key) is not None:
            parts.append(str(frame[key]))
            break
    for key in ("frame_id", "timestamp", "image_id", "token"):
        if frame.get(key) is not None:
            parts.append(str(frame[key]))
            break
    if not parts:
        for key in ("image", "color", "preview", "depth"):
            if frame.get(key):
                parts.append(Path(str(frame[key])).stem)
                break
    return sanitize("_".join(parts) if parts else f"{fallback:06d}")


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
