from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REMOVED_DATASET_CONFIG_KEYS = {"layout", "frame_num", "stride", "resolution", "max_samples", "batch_size"}


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    dataset: str
    root: str
    options: dict[str, Any]


def _validate_path_mapping(entry: dict[str, Any], key: str, entry_name: str, allow_null: bool) -> None:
    if key not in entry:
        return
    mapping = entry[key]
    if not isinstance(mapping, dict):
        raise ValueError(f"{entry_name}.{key} must be an object")
    for root_name, value in mapping.items():
        if not isinstance(root_name, str) or not root_name:
            raise ValueError(f"{entry_name}.{key} contains an invalid root name")
        if value is None:
            if allow_null:
                continue
            raise ValueError(f"{entry_name}.{key}.{root_name} must be a path, got null")
        if not isinstance(value, str):
            raise ValueError(f"{entry_name}.{key}.{root_name} must be a path string")
        if not Path(value).exists():
            raise FileNotFoundError(f"{entry_name}.{key}.{root_name} does not exist: {value}")


def load_dataset_configs(path: str | Path) -> list[DatasetConfig]:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    entries = data.get("datasets", data if isinstance(data, list) else None)
    if not isinstance(entries, list):
        raise ValueError("config must be a list or an object with a 'datasets' list")

    configs: list[DatasetConfig] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        label = entry.get("label")
        dataset = entry.get("dataset") or entry.get("type")
        root = entry.get("root") or entry.get("path")
        if not label or not dataset or not root:
            raise ValueError(f"datasets[{index}] must include label, dataset, and root")
        entry_name = f"datasets[{index}]"
        removed_keys = sorted(REMOVED_DATASET_CONFIG_KEYS.intersection(entry))
        if removed_keys:
            keys = ", ".join(removed_keys)
            raise ValueError(f"{entry_name} contains runtime-only or removed dataset config field(s): {keys}")
        _validate_path_mapping(entry, "roots", entry_name, allow_null=False)
        _validate_path_mapping(entry, "optional_roots", entry_name, allow_null=True)
        options = {key: value for key, value in entry.items() if key not in {"label", "dataset", "type", "root", "path"}}
        configs.append(DatasetConfig(label=str(label), dataset=str(dataset).lower(), root=str(root), options=options))
    return configs
