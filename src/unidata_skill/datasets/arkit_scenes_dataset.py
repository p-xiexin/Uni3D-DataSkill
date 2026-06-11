from __future__ import annotations

import math
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


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def _read_rgb_image(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        image = cv2.imread(str(path))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    if cv2 is not None:
        matrix, _ = cv2.Rodrigues(axis_angle.astype(np.float32))
        return matrix.astype(np.float32)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)).astype(np.float32)


def _timestamp_from_stem(path: Path) -> float:
    stem = path.stem
    if "_" in stem:
        stem = stem.rsplit("_", 1)[-1]
    return float(stem)


def _nearest_by_timestamp(timestamp: float, items: dict[float, Any], tolerance: float | None = None) -> Any | None:
    if not items:
        return None
    key = min(items, key=lambda item: abs(item - timestamp))
    if tolerance is not None and abs(key - timestamp) > tolerance:
        return None
    return items[key]


def _read_pincam(path: Path) -> np.ndarray:
    values = [float(item) for item in path.read_text(encoding="utf-8").split()]
    if len(values) != 6:
        raise ValueError(f"{path}: expected width height fx fy cx cy")
    _, _, fx, fy, cx, cy = values
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _read_trajectory(path: Path) -> dict[float, np.ndarray]:
    poses: dict[float, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 7:
            raise ValueError(f"{path}:{line_no}: expected timestamp axis-angle xyz translation")
        values = np.array([float(item) for item in parts], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _axis_angle_to_matrix(values[1:4])
        pose[:3, 3] = values[4:7]
        poses[float(values[0])] = pose
    return poses


def _read_depth_png_meters(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    return depth / 1000.0


@dataclass(frozen=True)
class ARKitScenesFrame:
    scan_id: str
    frame_id: str
    image_path: Path
    depth_path: Path
    intrinsics: np.ndarray
    camera_pose: np.ndarray


class ARKitScenesPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        scan_ids: list[str] | None = None,
        splits: tuple[str, ...] = ("Training", "Validation"),
        frame_num: int = 8,
        stride: int = 1,
        resolution: list[int] | tuple[int, int] = (512, 384),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
        self.dataset_label = "ARKitScenesPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scans_root = _require_dir(component_roots.get("scans", self._default_scans_root()), "roots.scans")
        self.splits = splits
        self.stride = stride
        self.verbose = verbose
        self.scan_dirs = self._discover_scan_dirs(scan_ids)
        self.sequences = sorted(self.scan_dirs)
        self.frames = {scan_id: self._build_scan_frames(scan_id, scan_dir) for scan_id, scan_dir in self.scan_dirs.items()}
        self.num_imgs = {scan_id: len(frames) for scan_id, frames in self.frames.items()}
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scans_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _default_scans_root(self) -> Path:
        if (self.data_root / "3dod").is_dir():
            return self.data_root / "3dod"
        return self.data_root

    def _discover_scan_dirs(self, scan_ids: list[str] | None) -> dict[str, Path]:
        if scan_ids:
            return {scan_id: self._find_scan_dir(scan_id) for scan_id in scan_ids}
        scan_dirs: dict[str, Path] = {}
        roots = [self.scans_root / split for split in self.splits if (self.scans_root / split).is_dir()]
        if (self.scans_root / "sample_data").is_dir():
            roots.append(self.scans_root / "sample_data")
        if not roots:
            roots = [self.scans_root]
        for root in roots:
            for path in sorted(root.iterdir()):
                if path.is_dir() and self._frames_dir(path) is not None:
                    scan_dirs[path.name] = path
        return scan_dirs

    def _find_scan_dir(self, scan_id: str) -> Path:
        candidates = [self.scans_root / split / scan_id for split in self.splits]
        candidates += [self.scans_root / "sample_data" / scan_id, self.scans_root / scan_id]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"ARKitScenes scan not found under {self.scans_root}: {scan_id}")

    def _frames_dir(self, scan_dir: Path) -> Path | None:
        direct = scan_dir / f"{scan_dir.name}_frames"
        if direct.is_dir():
            return direct
        matches = sorted(scan_dir.glob("*_frames"))
        return matches[0] if matches else None

    def _build_scan_frames(self, scan_id: str, scan_dir: Path) -> list[ARKitScenesFrame]:
        frames_dir = self._frames_dir(scan_dir)
        if frames_dir is None:
            return []
        image_dir = frames_dir / "wide"
        depth_dir = frames_dir / "depth_densified"
        intrinsics_dir = frames_dir / "color_intrinsics"
        pose_path = frames_dir / "color.traj"
        for name, path in (("wide", image_dir), ("depth_densified", depth_dir), ("color_intrinsics", intrinsics_dir)):
            _require_dir(path, f"{scan_id}.{name}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"ARKitScenes trajectory not found: {pose_path}")

        poses = _read_trajectory(pose_path)
        intrinsics_by_time = {_timestamp_from_stem(path): _read_pincam(path) for path in intrinsics_dir.glob("*.pincam")}
        depths_by_stem = {path.stem: path for path in depth_dir.glob("*.png")}
        frames: list[ARKitScenesFrame] = []
        for image_path in sorted(image_dir.glob("*.png")):
            timestamp = _timestamp_from_stem(image_path)
            depth_path = depths_by_stem.get(image_path.stem)
            intrinsics = _nearest_by_timestamp(timestamp, intrinsics_by_time)
            pose = _nearest_by_timestamp(timestamp, poses)
            if depth_path is None or intrinsics is None or pose is None:
                continue
            frames.append(ARKitScenesFrame(scan_id, image_path.stem, image_path, depth_path, intrinsics.copy(), pose.copy()))
        return frames

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
        for idx in idxs:
            frame = frames[idx]
            img = _read_rgb_image(frame.image_path)
            if img is None:
                continue
            depthmap = _read_depth_png_meters(frame.depth_path)
            intrinsics = frame.intrinsics.copy()
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
                    "camera_pose": frame.camera_pose.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": frame.frame_id,
                    "prefix": f"{scene}_{frame.frame_id}",
                    "image_path": str(frame.image_path),
                    "depth_path": str(frame.depth_path),
                    "depth_source": "gt_dense",
                    "pose_source": "gt",
                    "intrinsics_source": "gt",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
