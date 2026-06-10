from __future__ import annotations

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


KITTI360_CAMERAS = ("image_00", "image_01")


def _strip_inline_comment(line: str) -> str:
    for marker in ("#", "//"):
        line = line.split(marker, 1)[0]
    return line.strip()


def _parse_float_tokens(tokens: list[str], path: Path, line_no: int) -> list[float]:
    try:
        return [float(item) for item in tokens]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: expected numeric values") from exc


def _parse_matrix_line(line: str, path: Path, line_no: int) -> tuple[str, np.ndarray] | None:
    line = _strip_inline_comment(line)
    if not line.strip() or ":" not in line:
        return None
    name, values = line.split(":", 1)
    floats = _parse_float_tokens(values.split(), path, line_no)
    if len(floats) == 9:
        return name.strip(), np.array(floats, dtype=np.float32).reshape(3, 3)
    if len(floats) == 12:
        return name.strip(), np.array(floats, dtype=np.float32).reshape(3, 4)
    raise ValueError(f"{path}:{line_no}: expected 9 or 12 numeric values for {name.strip()!r}, got {len(floats)}")


def _parse_pose_line(line: str, path: Path, line_no: int) -> tuple[int, np.ndarray] | None:
    line = _strip_inline_comment(line)
    if not line:
        return None
    parts = line.split()
    if len(parts) not in (13, 17):
        raise ValueError(f"{path}:{line_no}: expected frame id plus 12 or 16 pose values, got {len(parts)} tokens")
    try:
        frame_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: expected integer frame id") from exc
    values = _parse_float_tokens(parts[1:], path, line_no)
    pose = np.eye(4, dtype=np.float32)
    if len(values) == 12:
        pose[:3, :] = np.array(values, dtype=np.float32).reshape(3, 4)
    else:
        pose[:, :] = np.array(values, dtype=np.float32).reshape(4, 4)
        if not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"{path}:{line_no}: expected homogeneous pose last row [0, 0, 0, 1]")
    if not np.isfinite(pose).all():
        raise ValueError(f"{path}:{line_no}: pose contains non-finite values")
    return frame_id, pose


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def load_perspective_intrinsics(calibration_root: Path) -> dict[str, np.ndarray]:
    path = calibration_root / "perspective.txt"
    records: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_matrix_line(line, path, line_no)
        if parsed is not None:
            records[parsed[0]] = parsed[1]

    intrinsics: dict[str, np.ndarray] = {}
    for camera_id, key in (("image_00", "P_rect_00"), ("image_01", "P_rect_01")):
        if key not in records:
            continue
        intrinsics[camera_id] = records[key][:3, :3].astype(np.float32)
    return intrinsics


def load_cam0_to_world(poses_root: Path, sequence: str) -> dict[int, np.ndarray]:
    path = poses_root / sequence / "cam0_to_world.txt"
    poses: dict[int, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_pose_line(line, path, line_no)
        if parsed is not None:
            poses[parsed[0]] = parsed[1]
    return poses


def discover_sequences(images_root: Path) -> list[str]:
    return sorted(path.name for path in images_root.iterdir() if path.is_dir())


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


@dataclass(frozen=True)
class Kitti360Frame:
    sequence: str
    camera_id: str
    frame_id: int
    image_path: Path


class Kitti360Pi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        sequences: list[str] | None = None,
        cameras: tuple[str, ...] = ("image_00",),
        frame_num: int = 8,
        stride: int = 5,
        resolution: list[int] | tuple[int, int] = (512, 384),
        layout: str = "official",
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
        self.dataset_label = "KITTI360Pi3X"
        self.data_root = Path(data_root)
        self.layout = layout
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.calibration_root = _require_dir(component_roots.get("calibration", self.data_root / "calibration"), "roots.calibration")
        self.images_root = _require_dir(component_roots.get("images", self.data_root / "data_2d_raw"), "roots.images")
        self.poses_root = _require_dir(component_roots.get("poses", self.data_root / "data_poses"), "roots.poses")
        self.sequences = sequences or discover_sequences(self.images_root)
        self.cameras = cameras
        self.stride = stride
        self.intrinsics = load_perspective_intrinsics(self.calibration_root)
        self.poses = {sequence: load_cam0_to_world(self.poses_root, sequence) for sequence in self.sequences}
        self.frames = self._build_frames()
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

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
                image_dir = self.images_root / sequence / camera_id / "data_rect"
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
            intrinsics = self.intrinsics[frame.camera_id].copy()
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(frame.image_path),
            )[:3]
            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "camera_pose": self.poses[frame.sequence][frame.frame_id].copy().astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": f"{frame.camera_id}_{frame.frame_id:010d}.png",
                    "prefix": f"{scene}_{frame.camera_id}_{frame.frame_id:010d}",
                    "image_path": str(frame.image_path),
                    "depth_source": "placeholder_missing_dense_depth",
                }
            )
        return views
