from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..pi3x import load_pi3_base_dataset

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


def _resize_rgb_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BILINEAR))


def _resize_depth(depthmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(depthmap, size, interpolation=cv2.INTER_NEAREST)
    return np.asarray(Image.fromarray(depthmap).resize(size, Image.Resampling.NEAREST))


def _as_resolution(resolution: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(resolution) == 1 and isinstance(resolution[0], (list, tuple)):  # type: ignore[index]
        resolution = resolution[0]  # type: ignore[assignment]
    return int(resolution[0]), int(resolution[1])


def _pose_from_3x4(values: Iterable[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :] = np.array(list(values), dtype=np.float32).reshape(3, 4)
    return pose


def _parse_calib(path: Path) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = [float(item) for item in value.split()]
        if len(values) == 12:
            records[key.strip()] = np.array(values, dtype=np.float32).reshape(3, 4)
    return records


def _camera_key(camera: str) -> str:
    suffix = camera.split("_")[-1]
    return f"P{int(suffix)}"


@dataclass(frozen=True)
class WaymoKittiFrame:
    sequence: str
    camera_id: str
    frame_id: str
    image_path: Path
    camera_intrinsics: np.ndarray
    camera_pose: np.ndarray


class WaymoKittiPi3XDataset(load_pi3_base_dataset()):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        data_root: str | Path,
        sequences: list[str] | None = None,
        cameras: tuple[str, ...] = ("image_2",),
        frame_num: int = 8,
        stride: int = 1,
        resolution: list[int] | tuple[int, int] = (512, 384),
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
        self.dataset_label = "WaymoKittiPi3X"
        self.data_root = Path(data_root)
        self.cameras = cameras
        self.stride = stride
        self.verbose = verbose

        sequence_root = self.data_root / "sequences"
        self.sequences = sequences or sorted(path.name for path in sequence_root.iterdir() if path.is_dir())
        self.frames = {sequence: self._build_sequence_frames(sequence) for sequence in self.sequences}
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _build_sequence_frames(self, sequence: str) -> list[WaymoKittiFrame]:
        sequence_dir = self.data_root / "sequences" / sequence
        calib = _parse_calib(sequence_dir / "calib.txt")
        pose_path = self.data_root / "poses" / f"{sequence}.txt"
        poses = [_pose_from_3x4(float(item) for item in line.split()) for line in pose_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        frames: list[WaymoKittiFrame] = []
        for camera in self.cameras:
            camera_key = _camera_key(camera)
            if camera_key not in calib:
                continue
            intrinsics = calib[camera_key][:3, :3].astype(np.float32)
            image_dir = sequence_dir / camera
            if not image_dir.is_dir() and camera.startswith("image_"):
                image_dir = sequence_dir / f"image_{int(camera.split('_')[-1]):02d}"
            if not image_dir.is_dir():
                continue
            image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
            for image_path in image_paths:
                idx = int(image_path.stem)
                if idx >= len(poses):
                    continue
                frames.append(WaymoKittiFrame(sequence, camera, image_path.stem, image_path, intrinsics, poses[idx]))
        return sorted(frames, key=lambda item: (item.frame_id, item.camera_id))

    def _window_indices(self, num_imgs: int, rng: np.random.Generator) -> range:
        required_span = (self.frame_num - 1) * self.stride + 1
        if num_imgs <= required_span:
            return range(0, num_imgs, self.stride)
        begin = int(rng.integers(0, num_imgs - required_span + 1))
        return range(begin, begin + required_span, self.stride)

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        scene = self.sequences[index]
        frames = self.frames.get(scene, [])
        if not frames:
            self.this_views_info = dict(scene=scene, idxs=[])
            return []

        idxs = self._window_indices(len(frames), rng)
        self.this_views_info = dict(scene=scene, idxs=list(idxs))

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
            }
            if hasattr(self, "_crop_resize_if_necessary"):
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
            elif (width, height) != (target_width, target_height):
                factor_w = target_width / width
                factor_h = target_height / height
                intrinsics[0, 0] *= factor_w
                intrinsics[1, 1] *= factor_h
                intrinsics[0, 2] *= factor_w
                intrinsics[1, 2] *= factor_h
                view["img"] = _resize_rgb_image(img, (target_width, target_height))
                view["depthmap"] = _resize_depth(depthmap, (target_width, target_height)).astype(np.float32)
                view["camera_intrinsics"] = intrinsics.astype(np.float32)
            views.append(view)
        return views
