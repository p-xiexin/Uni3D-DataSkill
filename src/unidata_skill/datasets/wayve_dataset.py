from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]


def _read_rgb_image(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def _as_resolution(resolution: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(resolution) == 1 and isinstance(resolution[0], (list, tuple)):  # type: ignore[index]
        resolution = resolution[0]  # type: ignore[assignment]
    return int(resolution[0]), int(resolution[1])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WayveFrame:
    scene: str
    camera_id: str
    frame_id: str
    image_path: Path
    camera_intrinsics: np.ndarray
    camera_pose: np.ndarray
    split: str


class WayveScenesPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        scene_dirs: list[str] | None = None,
        transforms_name: str = "transforms.json",
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "WayveScenesPi3X"
        self.data_root = Path(data_root)
        self.transforms_name = transforms_name
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scenes_root = _require_dir(component_roots.get("scenes", self.data_root), "roots.scenes")
        self.images_root = component_roots.get("images")
        if self.images_root is not None:
            _require_dir(self.images_root, "roots.images")

        self.sequences = scene_dirs or sorted(path.name for path in self.scenes_root.iterdir() if (path / transforms_name).is_file())
        self.frames = {scene: self._build_scene_frames(scene) for scene in self.sequences}
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _build_scene_frames(self, scene: str) -> list[WayveFrame]:
        scene_dir = self.scenes_root / scene
        transforms = _read_json(scene_dir / self.transforms_name)
        frames: list[WayveFrame] = []
        for idx, item in enumerate(transforms.get("frames", [])):
            image_path = scene_dir / item["file_path"]
            if self.images_root is not None:
                image_path = self.images_root / scene / item["file_path"]
            width = float(item.get("w", transforms.get("w", 1.0)))
            height = float(item.get("h", transforms.get("h", 1.0)))
            fx = float(item.get("fl_x", transforms.get("fl_x", 1.0)))
            fy = float(item.get("fl_y", transforms.get("fl_y", fx)))
            cx = float(item.get("cx", transforms.get("cx", width / 2.0)))
            cy = float(item.get("cy", transforms.get("cy", height / 2.0)))
            frames.append(
                WayveFrame(
                    scene=scene,
                    camera_id=str(item.get("camera", item.get("camera_id", "camera"))),
                    frame_id=str(item.get("frame_id", idx)),
                    image_path=image_path,
                    camera_intrinsics=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
                    camera_pose=np.array(item["transform_matrix"], dtype=np.float32).reshape(4, 4),
                    split=item.get("split", transforms.get("split", "unknown")),
                )
            )
        return sorted(frames, key=lambda frame: (frame.frame_id, frame.camera_id))

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        scene = self.sequences[index]
        frames = self.frames.get(scene, [])
        if not frames:
            self.this_views_info = dict(scene=scene, idxs=[])
            return []

        should_replace = len(frames) < self.frame_num
        idxs = list(rng.choice(len(frames), self.frame_num, replace=should_replace))
        self.this_views_info = dict(scene=scene, idxs=idxs)

        views = []
        target_width, target_height = _as_resolution(resolution)
        for idx in idxs:
            frame = frames[idx]
            img = _read_rgb_image(frame.image_path)
            if img is None:
                print(f"Warning: Failed to load image: {frame.image_path}", flush=True)
                continue

            height, width = img.shape[:2]
            depthmap = np.ones((height, width), dtype=np.float32)
            intrinsics = frame.camera_intrinsics.copy()
            view = {
                "img": img,
                "depthmap": depthmap,
                "camera_pose": frame.camera_pose.astype(np.float32),
                "camera_intrinsics": intrinsics.astype(np.float32),
                "dataset": self.dataset_label,
                "label": scene,
                "instance": f"{frame.camera_id}_{frame.frame_id}{frame.image_path.suffix}",
                "prefix": f"{scene}_{frame.camera_id}_{frame.frame_id}",
                "image_path": str(frame.image_path),
                "depth_source": "placeholder_missing_dense_depth",
                "split": frame.split,
            }
            img2, depth2, intrinsics2 = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                (target_width, target_height),
                rng=rng,
                info=str(frame.image_path),
            )[:3]
            view["img"] = img2
            view["depthmap"] = depth2.astype(np.float32)
            view["camera_intrinsics"] = intrinsics2.astype(np.float32)
            views.append(view)
        return views
