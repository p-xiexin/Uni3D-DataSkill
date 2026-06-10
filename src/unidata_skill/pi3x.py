from __future__ import annotations

import importlib
import sys
from pathlib import Path


def default_pi3_root() -> Path:
    return Path(__file__).resolve().parents[2] / "thirdparty" / "Pi3"


def resolve_pi3_root() -> Path:
    path = default_pi3_root().resolve()
    if (path / "datasets").is_dir():
        return path
    raise FileNotFoundError(f"Pi3 training checkout not found. Install yyfz/Pi3 training under {default_pi3_root()}")


def add_pi3_root() -> Path:
    root = resolve_pi3_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_pi3_base_dataset() -> type:
    root = add_pi3_root()
    try:
        module = importlib.import_module("datasets.base.base_dataset")
        return module.BaseDataset
    except Exception as exc:
        raise ImportError(f"failed to import Pi3 BaseDataset from {root}") from exc
