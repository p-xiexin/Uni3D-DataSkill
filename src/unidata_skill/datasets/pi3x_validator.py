from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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
    return _validate_view_payload(view, prefix, batched=False)


def validate_pi3x_batch_view(view: dict[str, Any], batch_idx: int, view_idx: int) -> list[str]:
    prefix = f"batch={batch_idx} view={view_idx}"
    return _validate_view_payload(view, prefix, batched=True)


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _validate_image(value: Any, prefix: str, batched: bool) -> list[str]:
    if not isinstance(value, torch.Tensor):
        return [f"{prefix}: img must be torch.Tensor"]
    if value.numel() == 0:
        return [f"{prefix}: invalid img"]
    if batched:
        valid_shape = value.ndim == 4 and value.shape[0] > 0 and value.shape[1] == 3
    else:
        valid_shape = value.ndim == 3 and value.shape[0] == 3
    errors = []
    if not valid_shape:
        errors.append(f"{prefix}: invalid img shape {tuple(value.shape)}")
    if not _is_finite_tensor(value):
        errors.append(f"{prefix}: img contains non-finite values")
    return errors


def _validate_depth(value: Any, prefix: str, batched: bool) -> list[str]:
    if not isinstance(value, torch.Tensor):
        return [f"{prefix}: depthmap must be torch.Tensor"]
    valid_shape = value.ndim == 2 or (batched and value.ndim in (3, 4))
    if not valid_shape or value.numel() == 0 or not _is_finite_tensor(value):
        return [f"{prefix}: invalid depthmap"]
    return []


def _validate_intrinsics(value: Any, prefix: str, batched: bool) -> list[str]:
    if not isinstance(value, torch.Tensor):
        return [f"{prefix}: camera_intrinsics must be torch.Tensor"]
    if tuple(value.shape) == (3, 3):
        valid = _is_finite_tensor(value) and value[0, 0].item() > 0 and value[1, 1].item() > 0
    elif batched and value.ndim == 3 and tuple(value.shape[1:]) == (3, 3):
        valid = _is_finite_tensor(value) and bool((value[:, 0, 0] > 0).all().item()) and bool((value[:, 1, 1] > 0).all().item())
    else:
        valid = False
    if not valid:
        return [f"{prefix}: invalid camera_intrinsics"]
    return []


def _validate_pose(value: Any, prefix: str, batched: bool) -> list[str]:
    if not isinstance(value, torch.Tensor):
        return [f"{prefix}: camera_pose must be torch.Tensor"]
    last_row = torch.tensor([0, 0, 0, 1], dtype=value.dtype, device=value.device)
    if tuple(value.shape) == (4, 4):
        valid = _is_finite_tensor(value) and bool(torch.allclose(value[3], last_row))
    elif batched and value.ndim == 3 and tuple(value.shape[1:]) == (4, 4):
        valid = _is_finite_tensor(value) and bool(torch.allclose(value[:, 3, :], last_row.expand(value.shape[0], -1)))
    else:
        valid = False
    if not valid:
        return [f"{prefix}: invalid camera_pose"]
    return []


def _validate_image_path(value: Any, prefix: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        paths = value
    else:
        paths = [value]
    errors = []
    for path in paths:
        if isinstance(path, str) and not Path(path).is_file():
            errors.append(f"{prefix}: missing image_path {path}")
    return errors


def _validate_view_payload(view: dict[str, Any], prefix: str, batched: bool) -> list[str]:
    errors: list[str] = []

    for key in ("img", "depthmap", "camera_intrinsics", "camera_pose", "dataset", "label", "instance"):
        if key not in view:
            errors.append(f"{prefix}: missing {key}")

    errors.extend(_validate_image_path(view.get("image_path"), prefix))
    errors.extend(_validate_image(view.get("img"), prefix, batched))
    errors.extend(_validate_depth(view.get("depthmap"), prefix, batched))
    errors.extend(_validate_intrinsics(view.get("camera_intrinsics"), prefix, batched))
    errors.extend(_validate_pose(view.get("camera_pose"), prefix, batched))

    return errors


def validate_pi3x_batch(batch: Any, batch_idx: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(batch, list):
        return [f"batch={batch_idx}: expected list of batched views"]
    if len(batch) == 0:
        return [f"batch={batch_idx}: no views returned"]
    for view_idx, view in enumerate(batch):
        if not isinstance(view, dict):
            errors.append(f"batch={batch_idx} view={view_idx}: expected dict")
            continue
        errors.extend(validate_pi3x_batch_view(view, batch_idx, view_idx))
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
            for key in ("img", "depthmap", "camera_intrinsics", "camera_pose", "dataset", "label", "instance"):
                if key not in view:
                    errors.append(f"sample={sample_idx} view={view_idx}: missing {key}")

    checked_batches = 0
    try:
        from torch.utils.data import DataLoader
        from datasets.base.base_dataset import unified_collate_fn

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)
        for checked_batches, batch in enumerate(loader, start=1):
            errors.extend(validate_pi3x_batch(batch, checked_batches - 1))
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
