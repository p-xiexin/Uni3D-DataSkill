from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pi3x_validator import ValidationResult, validate_pi3x_dataset

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]


KITTI360_CAMERAS = ("image_00", "image_01")


class _FallbackBaseDataset:
    """Small local stand-in used when the Pi3X training branch is not importable."""

    def __init__(
        self,
        resolution: list[int] | tuple[int, int] | None = None,
        frame_num: int = 2,
        shuffle: bool = False,
        **_: Any,
    ) -> None:
        self.frame_num = frame_num
        self.shuffle = shuffle
        self._resolutions = [list(resolution or [512, 384])]
        self._rng = np.random.default_rng(2024)

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        return self._get_views(idx, self._resolutions[0], self._rng)


def _load_pi3_base_dataset(pi3_root: str | Path | None) -> type:
    if pi3_root:
        root = str(Path(pi3_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        module = importlib.import_module("datasets.base.base_dataset")
        return module.BaseDataset
    except Exception:
        return _FallbackBaseDataset


def _parse_matrix_line(line: str) -> tuple[str, np.ndarray] | None:
    if not line.strip() or ":" not in line:
        return None
    name, values = line.split(":", 1)
    floats = [float(item) for item in values.split()]
    if len(floats) == 9:
        return name.strip(), np.array(floats, dtype=np.float32).reshape(3, 3)
    if len(floats) == 12:
        return name.strip(), np.array(floats, dtype=np.float32).reshape(3, 4)
    return None


def load_perspective_intrinsics(kitti360_root: Path) -> dict[str, np.ndarray]:
    path = kitti360_root / "calibration" / "perspective.txt"
    records: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_matrix_line(line)
        if parsed is not None:
            records[parsed[0]] = parsed[1]

    intrinsics: dict[str, np.ndarray] = {}
    for camera_id, key in (("image_00", "P_rect_00"), ("image_01", "P_rect_01")):
        if key not in records:
            continue
        intrinsics[camera_id] = records[key][:3, :3].astype(np.float32)
    return intrinsics


def load_cam0_to_world(kitti360_root: Path, sequence: str) -> dict[int, np.ndarray]:
    path = kitti360_root / "data_poses" / sequence / "cam0_to_world.txt"
    poses: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) not in (13, 17):
            continue
        frame_id = int(parts[0])
        values = [float(item) for item in parts[1:]]
        pose = np.eye(4, dtype=np.float32)
        if len(values) == 12:
            pose[:3, :] = np.array(values, dtype=np.float32).reshape(3, 4)
        else:
            pose[:, :] = np.array(values, dtype=np.float32).reshape(4, 4)
        poses[frame_id] = pose
    return poses


def discover_sequences(kitti360_root: Path) -> list[str]:
    image_root = kitti360_root / "data_2d_raw"
    if not image_root.is_dir():
        return []
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


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


def _resize_rgb_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BILINEAR))


def _resize_depth(depthmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(depthmap, size, interpolation=cv2.INTER_NEAREST)
    return np.asarray(Image.fromarray(depthmap).resize(size, Image.Resampling.NEAREST))


@dataclass(frozen=True)
class Kitti360Frame:
    sequence: str
    camera_id: str
    frame_id: int
    image_path: Path


def make_kitti360_pi3x_dataset_class(pi3_root: str | Path | None = None) -> type:
    base_dataset = _load_pi3_base_dataset(pi3_root)

    class Kitti360Pi3XDataset(base_dataset):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            kitti360_root: str | Path,
            sequences: list[str] | None = None,
            cameras: tuple[str, ...] = ("image_00",),
            frame_num: int = 8,
            stride: int = 5,
            resolution: list[int] | tuple[int, int] = (512, 384),
            pi3_root: str | Path | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
            self.dataset_label = "KITTI360Pi3X"
            self.kitti360_root = Path(kitti360_root)
            self.sequences = sequences or discover_sequences(self.kitti360_root)
            self.cameras = cameras
            self.stride = stride
            self.intrinsics = load_perspective_intrinsics(self.kitti360_root)
            self.poses = {sequence: load_cam0_to_world(self.kitti360_root, sequence) for sequence in self.sequences}
            self.frames = self._build_frames()
            self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
            print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.kitti360_root}", file=sys.stderr, flush=True)

        def __len__(self) -> int:
            return len(self.sequences)

        def _build_frames(self) -> dict[str, list[Kitti360Frame]]:
            sequence_frames: dict[str, list[Kitti360Frame]] = {}
            for sequence in self.sequences:
                frames: list[Kitti360Frame] = []
                sequence_poses = self.poses.get(sequence, {})
                for camera_id in self.cameras:
                    if camera_id not in self.intrinsics:
                        continue
                    image_dir = self.kitti360_root / "data_2d_raw" / sequence / camera_id / "data_rect"
                    if not image_dir.is_dir():
                        continue
                    for image_path in sorted(image_dir.glob("*.png")):
                        frame_id = int(image_path.stem)
                        if frame_id in sequence_poses:
                            frames.append(Kitti360Frame(sequence, camera_id, frame_id, image_path))
                sequence_frames[sequence] = sorted(frames, key=lambda item: (item.frame_id, item.camera_id))
            return sequence_frames

        def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
            if len(resolution) == 1 and isinstance(resolution[0], (list, tuple)):
                resolution = list(resolution[0])
            scene = self.sequences[index]
            frames = self.frames.get(scene, [])
            num_imgs = len(frames)

            if num_imgs == 0:
                self.this_views_info = dict(scene=scene, idxs=[])
                return []

            required_span = (self.frame_num - 1) * self.stride + 1
            if num_imgs <= required_span:
                idxs = range(0, num_imgs, self.stride)
            else:
                begin = int(rng.integers(0, num_imgs - required_span + 1))
                idxs = range(begin, begin + required_span, self.stride)

            self.this_views_info = dict(
                scene=scene,
                idxs=list(idxs),
            )

            views = []
            for idx in idxs:
                frame = frames[idx]
                img = _read_rgb_image(frame.image_path)
                if img is None:
                    print(f"Warning: Failed to load image: {frame.image_path}", flush=True)
                    continue
                height, width = img.shape[:2]
                depthmap = np.ones((height, width), dtype=np.float32)
                view = {
                    "img": img,
                    "depthmap": depthmap,
                    "camera_intrinsics": self.intrinsics[frame.camera_id].copy(),
                    "camera_pose": self.poses[frame.sequence][frame.frame_id].copy(),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": f"{frame.camera_id}_{frame.frame_id:010d}.png",
                    "prefix": f"{scene}_{frame.camera_id}_{frame.frame_id:010d}",
                    "image_path": str(frame.image_path),
                    "depth_source": "placeholder_missing_dense_depth",
                }
                if hasattr(self, "_crop_resize_if_necessary"):
                    img2, depth2, intrinsics2 = self._crop_resize_if_necessary(
                        img,
                        depthmap,
                        view["camera_intrinsics"],
                        resolution,
                        rng=rng,
                        info=view["label"],
                    )[:3]
                    view["img"] = img2
                    view["depthmap"] = depth2.astype(np.float32)
                    view["camera_intrinsics"] = intrinsics2.astype(np.float32)
                else:
                    target_width, target_height = resolution
                    if (width, height) != (target_width, target_height):
                        factor_w = target_width / width
                        factor_h = target_height / height
                        view["camera_intrinsics"][0, 0] *= factor_w
                        view["camera_intrinsics"][1, 1] *= factor_h
                        view["camera_intrinsics"][0, 2] *= factor_w
                        view["camera_intrinsics"][1, 2] *= factor_h
                        view["img"] = _resize_rgb_image(img, (target_width, target_height))
                        view["depthmap"] = _resize_depth(depthmap, (target_width, target_height))
                    view["depthmap"] = view["depthmap"].astype(np.float32)
                    view["camera_intrinsics"] = view["camera_intrinsics"].astype(np.float32)
                view["camera_pose"] = view["camera_pose"].astype(np.float32)
                views.append(view)
            return views

    return Kitti360Pi3XDataset


class Kitti360Pi3XDataset(make_kitti360_pi3x_dataset_class(None)):  # type: ignore[misc, valid-type]
    pass


def validate_kitti360_pi3x_dataloader(
    kitti360_root: str | Path,
    pi3_root: str | Path | None = None,
    sequences: list[str] | None = None,
    cameras: tuple[str, ...] = ("image_00",),
    frame_num: int = 8,
    stride: int = 5,
    resolution: tuple[int, int] = (512, 384),
    max_samples: int = 4,
    batch_size: int = 1,
) -> ValidationResult:
    dataset_class = make_kitti360_pi3x_dataset_class(pi3_root)
    dataset = dataset_class(
        kitti360_root=kitti360_root,
        sequences=sequences,
        cameras=cameras,
        frame_num=frame_num,
        stride=stride,
        resolution=resolution,
        pi3_root=pi3_root,
    )

    warnings = ["dense depth is not available in the first KITTI-360 workflow; depthmap is a placeholder"]
    return validate_pi3x_dataset(
        dataset,
        expected_frame_num=frame_num,
        max_samples=max_samples,
        batch_size=batch_size,
        warnings=warnings,
    )
