from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    dataset: str
    root: str
    options: dict[str, Any]


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
        options = {key: value for key, value in entry.items() if key not in {"label", "dataset", "type", "root", "path"}}
        configs.append(DatasetConfig(label=str(label), dataset=str(dataset).lower(), root=str(root), options=options))
    return configs
