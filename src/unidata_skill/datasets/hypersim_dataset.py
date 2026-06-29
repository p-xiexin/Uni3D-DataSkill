from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.base.base_dataset import BaseDataset


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


def _absolute(path: Path) -> str:
    return str(path.resolve())


def _read_hdf5_dataset(path: Path) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("HypersimPi3XDataset requires h5py to read official HDF5 files") from exc
    with h5py.File(path, "r") as handle:
        if "dataset" in handle:
            return np.asarray(handle["dataset"])
        first_key = next(iter(handle.keys()))
        return np.asarray(handle[first_key])


def _read_preview_or_hdf5_image(preview_path: Path | None, hdf5_path: Path | None) -> np.ndarray | None:
    if preview_path is not None and preview_path.is_file():
        return np.asarray(Image.open(preview_path).convert("RGB"))
    if hdf5_path is None or not hdf5_path.is_file():
        return None
    image = _read_hdf5_dataset(hdf5_path)
    if image.ndim != 3:
        raise ValueError(f"{hdf5_path}: expected HxWx3 color image")
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if image.max() > 1.0 or image.min() < 0.0:
        lo, hi = np.percentile(image, [1, 99])
        image = (image - lo) / max(float(hi - lo), 1e-6)
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _frame_number_from_depth(path: Path) -> int:
    parts = path.name.split(".")
    if len(parts) < 3 or parts[0] != "frame":
        raise ValueError(f"unexpected Hypersim frame filename: {path.name}")
    return int(parts[1])


def _find_preview_image(scene_dir: Path, camera_id: str, frame_no: int) -> Path | None:
    preview_dir = scene_dir / "images" / f"scene_{camera_id}_final_preview"
    patterns = (
        f"frame.{frame_no:04d}.tonemap.jpg",
        f"frame.{frame_no:04d}.tonemap.png",
        f"frame.{frame_no:04d}.color.jpg",
        f"frame.{frame_no:04d}.color.png",
        f"frame.{frame_no:04d}.jpg",
        f"frame.{frame_no:04d}.png",
    )
    for pattern in patterns:
        candidate = preview_dir / pattern
        if candidate.is_file():
            return candidate
    matches = sorted(preview_dir.glob(f"frame.{frame_no:04d}.*"))
    return matches[0] if matches else None


def _hypersim_intrinsics(width: int, height: int, fov_x_degrees: float) -> np.ndarray:
    fx = 0.5 * width / np.tan(np.deg2rad(fov_x_degrees) * 0.5)
    fy = fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _ray_distance_to_planar_depth(distance: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = distance.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_norm = np.sqrt(x * x + y * y + 1.0)
    return (distance / ray_norm).astype(np.float32)


def generate_hypersim_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    scene_dirs: list[str] | None = None,
    camera_ids: list[str] | None = None,
    roots: dict[str, str | Path] | None = None,
    fov_x_degrees: float = 60.0,
) -> dict[str, Any]:
    del fov_x_degrees
    data_root = Path(data_root)
    scenes_root = _require_dir(_path_roots(roots).get("scenes", data_root), "roots.scenes")
    scene_names = scene_dirs or sorted(path.name for path in scenes_root.iterdir() if (path / "_detail").is_dir())
    allowed_cameras = set(camera_ids or [])

    records = []
    for scene in tqdm(scene_names, desc="[Hypersim] building index", unit="scene"):
        scene_dir = scenes_root / scene
        detail_dir = scene_dir / "_detail"
        camera_dirs = sorted(path for path in detail_dir.glob("cam_*") if path.is_dir())
        if allowed_cameras:
            camera_dirs = [path for path in camera_dirs if path.name in allowed_cameras]

        frames = []
        for camera_dir in camera_dirs:
            camera_id = camera_dir.name
            orientations = _read_hdf5_dataset(camera_dir / "camera_keyframe_orientations.hdf5").astype(np.float32)
            positions = _read_hdf5_dataset(camera_dir / "camera_keyframe_positions.hdf5").astype(np.float32)
            depth_dir = scene_dir / "images" / f"scene_{camera_id}_geometry_hdf5"
            color_dir = scene_dir / "images" / f"scene_{camera_id}_final_hdf5"
            if not depth_dir.is_dir():
                continue
            for depth_path in sorted(depth_dir.glob("frame.*.depth_meters.hdf5")):
                frame_no = _frame_number_from_depth(depth_path)
                if frame_no >= len(orientations) or frame_no >= len(positions):
                    continue
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = orientations[frame_no] @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
                pose[:3, 3] = positions[frame_no]
                color_hdf5 = color_dir / f"frame.{frame_no:04d}.color.hdf5"
                preview_path = _find_preview_image(scene_dir, camera_id, frame_no)
                frames.append(
                    {
                        "camera_id": camera_id,
                        "frame_no": frame_no,
                        "preview": None if preview_path is None else _absolute(preview_path),
                        "color_hdf5": _absolute(color_hdf5) if color_hdf5.is_file() else None,
                        "depth": _absolute(depth_path),
                        "camera_pose": pose.astype(np.float32).tolist(),
                    }
                )
        frames.sort(key=lambda frame: (frame["camera_id"], frame["frame_no"]))
        records.append({"sequence_id": scene, "frames": frames})

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class HypersimPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        scene_dirs: list[str] | None = None,
        camera_ids: list[str] | None = None,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        self.fov_x_degrees = float(kwargs.pop("fov_x_degrees", 60.0))
        super().__init__(**kwargs)
        self.dataset_label = "HypersimPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scenes_root = _require_dir(component_roots.get("scenes", self.data_root), "roots.scenes")

        if index_file is None:
            index = generate_hypersim_index(
                self.data_root,
                scene_dirs=scene_dirs,
                camera_ids=camera_ids,
                roots=roots,
                fov_x_degrees=self.fov_x_degrees,
            )
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()

        selected = set(scene_dirs or [])
        self.sequences = []
        self.frames = {}
        for record in index.get("sequences", []):
            scene = record["sequence_id"]
            if selected and scene not in selected:
                continue
            frames = record.get("frames", [])
            if camera_ids is not None:
                allowed = set(camera_ids)
                frames = [frame for frame in frames if frame["camera_id"] in allowed]
            self.sequences.append(scene)
            self.frames[scene] = frames
        self.num_imgs = {scene: len(frames) for scene, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scenes_root}", file=sys.stderr, flush=True)

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
            preview_path = None if frame.get("preview") is None else Path(frame["preview"])
            color_hdf5_path = None if frame.get("color_hdf5") is None else Path(frame["color_hdf5"])
            depth_path = Path(frame["depth"])
            img = _read_preview_or_hdf5_image(preview_path, color_hdf5_path)
            if img is None:
                continue
            ray_distance = _read_hdf5_dataset(depth_path).astype(np.float32)
            intrinsics = _hypersim_intrinsics(img.shape[1], img.shape[0], self.fov_x_degrees)
            depthmap = _ray_distance_to_planar_depth(ray_distance, intrinsics)
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(depth_path),
            )[:3]
            instance = f"{frame['camera_id']}_{int(frame['frame_no']):04d}"
            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": np.array(frame["camera_pose"], dtype=np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": instance,
                    "prefix": f"{scene}_{instance}",
                    "depth_path": str(depth_path),
                    "depth_source": "native_gt_dense",
                    "depth_definition": "planar_z_from_hypersim_ray_distance",
                    "pose_source": "native_gt",
                    "intrinsics_source": "metadata",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
