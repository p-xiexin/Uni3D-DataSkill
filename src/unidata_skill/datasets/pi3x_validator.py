from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    status: str
    dataset_len: int
    checked_samples: int
    checked_batches: int
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dataset_len": self.dataset_len,
            "checked_samples": self.checked_samples,
            "checked_batches": self.checked_batches,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_pi3x_view(view: dict[str, Any], sample_idx: int, view_idx: int) -> list[str]:
    prefix = f"sample={sample_idx} view={view_idx}"
    errors: list[str] = []

    for key in ("img", "depthmap", "camera_intrinsics", "camera_pose", "dataset", "label", "instance"):
        if key not in view:
            errors.append(f"{prefix}: missing {key}")

    image_path = view.get("image_path")
    if image_path is not None and not Path(image_path).is_file():
        errors.append(f"{prefix}: missing image_path {image_path}")

    img = np.asarray(view.get("img"))
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        errors.append(f"{prefix}: invalid img")

    depth = np.asarray(view.get("depthmap"))
    if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
        errors.append(f"{prefix}: invalid depthmap")

    k = np.asarray(view.get("camera_intrinsics"))
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        errors.append(f"{prefix}: invalid camera_intrinsics")

    pose = np.asarray(view.get("camera_pose"))
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        errors.append(f"{prefix}: invalid camera_pose")

    return errors


def validate_pi3x_dataset(
    dataset: Any,
    expected_frame_num: int | None = None,
    max_samples: int = 4,
    batch_size: int = 1,
    warnings: list[str] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    report_warnings = list(warnings or [])
    dataset_len = len(dataset)

    if dataset_len == 0:
        errors.append("dataset contains no samples")
        return ValidationResult("error", dataset_len, 0, 0, errors, report_warnings)

    checked_samples = min(max_samples, dataset_len)
    for sample_idx in range(checked_samples):
        views = dataset[sample_idx]
        if not isinstance(views, list):
            errors.append(f"sample={sample_idx}: expected list of views")
            continue
        if len(views) == 0:
            errors.append(f"sample={sample_idx}: no views returned")
            continue
        if expected_frame_num is not None and len(views) > expected_frame_num:
            errors.append(f"sample={sample_idx}: expected at most {expected_frame_num} views, got {len(views)}")
        for view_idx, view in enumerate(views):
            if not isinstance(view, dict):
                errors.append(f"sample={sample_idx} view={view_idx}: expected dict")
                continue
            errors.extend(validate_pi3x_view(view, sample_idx, view_idx))

    checked_batches = 0
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)
        for checked_batches, _batch in enumerate(loader, start=1):
            if checked_batches >= 1:
                break
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            report_warnings.append("torch is not installed; skipped PyTorch DataLoader batch validation")
        else:
            errors.append(f"dataloader batch validation failed: {exc}")
    except Exception as exc:
        errors.append(f"dataloader batch validation failed: {exc}")

    return ValidationResult("ok" if not errors else "error", dataset_len, checked_samples, checked_batches, errors, report_warnings)
