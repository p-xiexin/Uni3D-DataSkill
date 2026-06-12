from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

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


def _resolve_existing_path(data_root: Path, value: str | Path, name: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, data_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found: {candidates[-1]}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


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


def _default_scans_root(data_root: Path) -> Path:
    if (data_root / "3dod").is_dir():
        return data_root / "3dod"
    return data_root


def _frames_dir(scan_dir: Path) -> Path | None:
    direct = scan_dir / f"{scan_dir.name}_frames"
    if direct.is_dir():
        return direct
    matches = sorted(scan_dir.glob("*_frames"))
    return matches[0] if matches else None


def _find_scan_dir(scans_root: Path, scan_id: str, splits: tuple[str, ...]) -> Path:
    candidates = [scans_root / split / scan_id for split in splits]
    candidates += [scans_root / "sample_data" / scan_id, scans_root / scan_id]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"ARKitScenes scan not found under {scans_root}: {scan_id}")


def _discover_scan_dirs(scans_root: Path, splits: tuple[str, ...], scan_ids: list[str] | None) -> dict[str, Path]:
    if scan_ids:
        return {scan_id: _find_scan_dir(scans_root, scan_id, splits) for scan_id in scan_ids}
    scan_dirs: dict[str, Path] = {}
    roots = [scans_root / split for split in splits if (scans_root / split).is_dir()]
    if (scans_root / "sample_data").is_dir():
        roots.append(scans_root / "sample_data")
    if not roots:
        roots = [scans_root]
    for root in roots:
        for path in sorted(root.iterdir()):
            if path.is_dir() and _frames_dir(path) is not None:
                scan_dirs[path.name] = path
    return scan_dirs


def generate_arkit_scenes_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    scan_ids: list[str] | None = None,
    splits: tuple[str, ...] = ("Training", "Validation"),
    roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    scans_root = _require_dir(_path_roots(roots).get("scans", _default_scans_root(data_root)), "roots.scans")
    scan_dirs = _discover_scan_dirs(scans_root, splits, scan_ids)

    records = []
    for scan_id in tqdm(sorted(scan_dirs), desc="[ARKitScenes] building index", unit="scan"):
        scan_dir = scan_dirs[scan_id]
        frames_dir = _frames_dir(scan_dir)
        if frames_dir is None:
            continue
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
        frames = []
        for image_path in sorted(image_dir.glob("*.png")):
            timestamp = _timestamp_from_stem(image_path)
            depth_path = depths_by_stem.get(image_path.stem)
            intrinsics = _nearest_by_timestamp(timestamp, intrinsics_by_time)
            pose = _nearest_by_timestamp(timestamp, poses)
            if depth_path is None or intrinsics is None or pose is None:
                continue
            frames.append(
                {
                    "frame_id": image_path.stem,
                    "image": _relative(image_path, scans_root),
                    "depth": _relative(depth_path, scans_root),
                    "camera_intrinsics": intrinsics.astype(np.float32).tolist(),
                    "camera_pose": pose.astype(np.float32).tolist(),
                }
            )
        records.append({"sequence_id": scan_id, "frames": frames})

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class ARKitScenesPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        scan_ids: list[str] | None = None,
        splits: tuple[str, ...] = ("Training", "Validation"),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "ARKitScenesPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scans_root = _require_dir(component_roots.get("scans", _default_scans_root(self.data_root)), "roots.scans")
        self.splits = splits

        if index_file is None:
            index = generate_arkit_scenes_index(self.data_root, scan_ids=scan_ids, splits=splits, roots=roots)
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()

        selected = set(scan_ids or [])
        self.sequences = []
        self.frames = {}
        for record in index.get("sequences", []):
            scan_id = record["sequence_id"]
            if selected and scan_id not in selected:
                continue
            self.sequences.append(scan_id)
            self.frames[scan_id] = record.get("frames", [])
        self.num_imgs = {scan_id: len(frames) for scan_id, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scans_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

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
        for idx in idxs:
            frame = frames[idx]
            image_path = self.scans_root / frame["image"]
            depth_path = self.scans_root / frame["depth"]
            img = _read_rgb_image(image_path)
            if img is None:
                continue
            depthmap = _read_depth_png_meters(depth_path)
            intrinsics = np.array(frame["camera_intrinsics"], dtype=np.float32)
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]
            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": np.array(frame["camera_pose"], dtype=np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": frame["frame_id"],
                    "prefix": f"{scene}_{frame['frame_id']}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_source": "native_gt_dense",
                    "pose_source": "native_gt",
                    "intrinsics_source": "native_gt",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
