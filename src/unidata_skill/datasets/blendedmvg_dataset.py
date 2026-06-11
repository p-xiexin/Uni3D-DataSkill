from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datasets.base.base_dataset import BaseDataset


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def _strip_inline_comment(line: str) -> str:
    for marker in ("#", "//"):
        line = line.split(marker, 1)[0]
    return line.strip()


def _parse_float_row(line: str, path: str, line_no: int, expected_cols: int) -> list[float]:
    tokens = _strip_inline_comment(line).split()
    if len(tokens) != expected_cols:
        raise ValueError(f"{path}:{line_no}: expected {expected_cols} numeric values, got {len(tokens)}")
    try:
        return [float(item) for item in tokens]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: expected numeric values") from exc


def _read_matrix_after_section(lines: list[str], section: str, rows: int, cols: int, path: str) -> np.ndarray:
    section_start = None
    for idx, line in enumerate(lines):
        if section in _strip_inline_comment(line).lower():
            section_start = idx + 1
            break
    if section_start is None:
        raise ValueError(f"{path}: section not found: {section}")

    matrix_rows = []
    for idx in range(section_start, len(lines)):
        clean_line = _strip_inline_comment(lines[idx])
        if not clean_line:
            continue
        lowered = clean_line.lower()
        if lowered in {"extrinsic", "intrinsic"} and matrix_rows:
            break
        if lowered in {"extrinsic", "intrinsic"}:
            continue
        matrix_rows.append(_parse_float_row(clean_line, path, idx + 1, cols))
        if len(matrix_rows) == rows:
            matrix = np.array(matrix_rows, dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError(f"{path}: section {section} contains non-finite values")
            return matrix

    raise ValueError(f"{path}: section {section} expected {rows} rows, got {len(matrix_rows)}")


def _read_camera_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    camera_pose = _read_matrix_after_section(lines, "extrinsic", 4, 4, path)
    intrinsics = _read_matrix_after_section(lines, "intrinsic", 3, 3, path)
    if not np.allclose(camera_pose[3], [0, 0, 0, 1]):
        raise ValueError(f"{path}: extrinsic last row must be [0, 0, 0, 1]")
    return camera_pose, intrinsics


def read_pfm(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError(f"Not a PFM file: {filename}")

        dim_line = f.readline().decode("ascii").strip()
        try:
            width, height = (int(item) for item in dim_line.split())
        except ValueError as exc:
            raise ValueError(f"{filename}: invalid PFM dimensions") from exc

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=endian + "f4")
        if channels == 1:
            data = data.reshape(height, width)
        else:
            data = data.reshape(height, width, channels)
        return np.flipud(data) * abs(scale)


class BlendedMVGDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | None = None,
        verbose: bool = False,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        list_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if data_root is None:
            raise ValueError("data_root is required")

        self.verbose = verbose
        self.dataset_label = "BlendedMVG"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scenes_root = _require_dir(component_roots.get("scenes", self.data_root), "roots.scenes")
        self.images_root = component_roots.get("images")
        self.depth_root = component_roots.get("depth")
        self.cameras_root = component_roots.get("cameras")
        for name, path in (("roots.images", self.images_root), ("roots.depth", self.depth_root), ("roots.cameras", self.cameras_root)):
            if path is not None:
                _require_dir(path, name)

        default_list_name = list_name or ("BlendedMVG_list.txt" if self.mode == "train" else "BlendedMVS_list.txt")
        list_file = component_roots.get("list", self.data_root / default_list_name)
        if not list_file.is_file():
            raise FileNotFoundError(f"List file not found: {list_file}")

        with open(list_file, "r", encoding="utf-8") as f:
            self.sequences = [_strip_inline_comment(line) for line in f.readlines()]
        self.sequences = [line for line in self.sequences if line]

        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scenes_root}", flush=True)

        self.num_imgs = {}
        for seq in self.sequences:
            img_path = self._scene_component_dir(seq, "blended_images", self.images_root)
            if img_path.is_dir():
                img_files = [name for name in os.listdir(img_path) if name.endswith(".jpg") and not name.endswith("_masked.jpg")]
                self.num_imgs[seq] = len(img_files)
            else:
                self.num_imgs[seq] = 0

    def __len__(self) -> int:
        return len(self.sequences)

    def _scene_component_dir(self, scene: str, component: str, override_root: Path | None) -> Path:
        if override_root is not None:
            return override_root / scene / component
        return self.scenes_root / scene / component

    def _get_views(self, index: int, resolution: tuple[int, int], rng: np.random.Generator, is_test: bool = False):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]

        if num_imgs <= self.frame_num:
            idxs = range(num_imgs)
        else:
            img_idx = int(rng.integers(0, num_imgs))
            front_num = (self.frame_num - 1) // 2
            back_num = self.frame_num - 1 - front_num
            if img_idx - front_num < 0:
                begin = 0
                end = self.frame_num
            elif img_idx + back_num >= num_imgs:
                begin = num_imgs - self.frame_num
                end = num_imgs
            else:
                begin = img_idx - front_num
                end = img_idx + back_num + 1
            idxs = range(begin, end)

        self.this_views_info = dict(scene=scene, idxs=list(idxs))

        views = []
        image_dir = self._scene_component_dir(scene, "blended_images", self.images_root)
        depth_dir = self._scene_component_dir(scene, "rendered_depth_maps", self.depth_root)
        camera_dir = self._scene_component_dir(scene, "cams", self.cameras_root)
        for idx in idxs:
            img_name = f"{idx:08d}.jpg"
            img_path = image_dir / img_name
            depth_path = depth_dir / f"{idx:08d}.pfm"
            cam_path = camera_dir / f"{idx:08d}_cam.txt"

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}", flush=True)
                continue
            if not depth_path.exists():
                print(f"Warning: Depth not found: {depth_path}", flush=True)
                continue
            if not cam_path.exists():
                print(f"Warning: Camera not found: {cam_path}", flush=True)
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to load image: {img_path}", flush=True)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                depthmap = read_pfm(str(depth_path))
            except Exception as exc:
                print(f"Warning: Failed to load depth: {depth_path}, error: {exc}", flush=True)
                continue

            try:
                camera_pose, intrinsics = _read_camera_file(str(cam_path))
            except Exception as exc:
                print(f"Warning: Failed to load camera: {cam_path}, error: {exc}", flush=True)
                intrinsics = np.array(
                    [[500.0, 0, img.shape[1] / 2], [0, 500.0, img.shape[0] / 2], [0, 0, 1]],
                    dtype=np.float32,
                )
                camera_pose = np.eye(4, dtype=np.float32)

            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(img_path),
            )

            views.append(
                dict(
                    img=img,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset=self.dataset_label,
                    label=scene,
                    instance=img_name,
                    prefix=f"{scene}_{img_name}",
                    depth_source="gt_dense",
                    pose_source="gt",
                    intrinsics_source="gt",
                    pseudo_label=False,
                    valid_mask_required=True,
                )
            )

        return views
