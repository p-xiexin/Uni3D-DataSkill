from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def _strip_inline_comment(line: str) -> str:
    for marker in ("#", "//"):
        line = line.split(marker, 1)[0]
    return line.strip()


def _parse_float_tokens(tokens: list[str], path: Path, line_no: int) -> list[float]:
    try:
        return [float(item) for item in tokens]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: expected numeric values") from exc


def _pose_from_3x4(values: Iterable[float], path: Path, line_no: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    values = list(values)
    if len(values) != 12:
        raise ValueError(f"{path}:{line_no}: expected 12 pose values, got {len(values)}")
    pose[:3, :] = np.array(values, dtype=np.float32).reshape(3, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"{path}:{line_no}: pose contains non-finite values")
    return pose


def _parse_kitti_calib(path: Path) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_inline_comment(line)
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = _parse_float_tokens(value.split(), path, line_no)
        if len(values) not in (9, 12):
            raise ValueError(f"{path}:{line_no}: expected 9 or 12 calibration values for {key.strip()!r}, got {len(values)}")
        if len(values) == 12:
            records[key.strip()] = np.array(values, dtype=np.float32).reshape(3, 4)
        else:
            matrix = np.eye(3, 4, dtype=np.float32)
            matrix[:3, :3] = np.array(values, dtype=np.float32).reshape(3, 3)
            records[key.strip()] = matrix
    return records


def _parse_kitti_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_inline_comment(line)
        if not line:
            continue
        values = _parse_float_tokens(line.split(), path, line_no)
        poses.append(_pose_from_3x4(values, path, line_no))
    return poses


def _kitti_camera_key(camera: str) -> str:
    suffix = camera.split("_")[-1]
    return f"P{int(suffix)}"


@dataclass(frozen=True)
class KittiOdometryFrame:
    sequence: str
    camera_id: str
    frame_id: str
    image_path: Path
    camera_intrinsics: np.ndarray
    camera_pose: np.ndarray


class KittiOdometryPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        sequences: list[str] | None = None,
        cameras: tuple[str, ...] = ("image_2",),
        frame_num: int = 8,
        stride: int = 1,
        resolution: list[int] | tuple[int, int] = (512, 384),
        layout: str = "official",
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
        self.dataset_label = "KITTIOdometryPi3X"
        self.data_root = Path(data_root)
        self.layout = layout
        self.cameras = cameras
        self.stride = stride
        self.verbose = verbose
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.sequences_root = _require_dir(component_roots.get("sequences", self.data_root / "sequences"), "roots.sequences")
        self.poses_root = _require_dir(component_roots.get("poses", self.data_root / "poses"), "roots.poses")

        self.sequences = sequences or sorted(path.name for path in self.sequences_root.iterdir() if path.is_dir())
        self.frames = {sequence: self._build_sequence_frames(sequence) for sequence in self.sequences}
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _build_sequence_frames(self, sequence: str) -> list[KittiOdometryFrame]:
        sequence_dir = self.sequences_root / sequence
        calib = _parse_kitti_calib(sequence_dir / "calib.txt")
        pose_path = self.poses_root / f"{sequence}.txt"
        poses = _parse_kitti_poses(pose_path)

        frames: list[KittiOdometryFrame] = []
        for camera in self.cameras:
            camera_key = _kitti_camera_key(camera)
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
                frames.append(KittiOdometryFrame(sequence, camera, image_path.stem, image_path, intrinsics, poses[idx]))
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
