from __future__ import annotations

import csv
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


def _absolute(path: Path) -> str:
    return str(path.resolve())


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


def _frame_number_from_stem(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    if not digits:
        raise ValueError(f"unexpected ASE frame filename: {path.name}")
    return int(digits)


def _discover_scene_dirs(scenes_root: Path, chunks: list[str] | None) -> list[Path]:
    roots = [scenes_root / chunk for chunk in chunks] if chunks else sorted(path for path in scenes_root.iterdir() if path.is_dir())
    scene_paths = []
    for path in roots:
        if _is_ase_scene_dir(path):
            scene_paths.append(path)
            continue
        if path.is_dir():
            scene_paths.extend(sorted(child for child in path.iterdir() if _is_ase_scene_dir(child)))
            continue
        raise FileNotFoundError(f"ASE chunk or scene directory not found: {path}")
    return scene_paths


def _trajectory_path(scene_dir: Path) -> Path | None:
    for name in ("trajectory.csv", "trajectory.txt"):
        path = scene_dir / name
        if path.is_file():
            return path
    return None


def _is_ase_scene_dir(path: Path) -> bool:
    return path.is_dir() and (path / "rgb").is_dir() and (path / "depth").is_dir() and _trajectory_path(path) is not None


def _quaternion_xyzw_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _pose_from_translation_quaternion(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _quaternion_xyzw_to_rotation(qx, qy, qz, qw)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_column(row: dict[str, str], candidates: tuple[str, ...]) -> float | None:
    normalized = {key.lower().strip(): value for key, value in row.items()}
    for candidate in candidates:
        value = _float_or_none(normalized.get(candidate))
        if value is not None:
            return value
    return None


def _pose_from_row(row: dict[str, str]) -> np.ndarray | None:
    tx = _find_column(row, ("tx", "t_x", "x", "translation_x", "device_position_x", "position_x", "tx_world_device"))
    ty = _find_column(row, ("ty", "t_y", "y", "translation_y", "device_position_y", "position_y", "ty_world_device"))
    tz = _find_column(row, ("tz", "t_z", "z", "translation_z", "device_position_z", "position_z", "tz_world_device"))
    qx = _find_column(row, ("qx", "q_x", "quat_x", "quaternion_x", "device_quaternion_x", "qx_world_device"))
    qy = _find_column(row, ("qy", "q_y", "quat_y", "quaternion_y", "device_quaternion_y", "qy_world_device"))
    qz = _find_column(row, ("qz", "q_z", "quat_z", "quaternion_z", "device_quaternion_z", "qz_world_device"))
    qw = _find_column(row, ("qw", "q_w", "quat_w", "quaternion_w", "device_quaternion_w", "qw_world_device"))
    if None not in (tx, ty, tz, qx, qy, qz, qw):
        return _pose_from_translation_quaternion(tx, ty, tz, qx, qy, qz, qw)

    values = [_float_or_none(value) for value in row.values()]
    values = [value for value in values if value is not None]
    if len(values) >= 16:
        return np.asarray(values[-16:], dtype=np.float32).reshape(4, 4)
    return None


def load_ase_trajectory(trajectory_path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"{trajectory_path}: no trajectory records found")

    first = lines[0]
    if any(char.isalpha() for char in first):
        if "," in first:
            rows = csv.DictReader(lines)
        else:
            header = first.split()
            rows = (dict(zip(header, line.split())) for line in lines[1:])
        poses = []
        for row in rows:
            pose = _pose_from_row(row)
            if pose is not None:
                poses.append(pose.astype(np.float32))
        if poses:
            return poses
        raise ValueError(f"{trajectory_path}: unsupported trajectory header columns")

    poses = []
    for line_no, line in enumerate(lines, start=1):
        parts = line.replace(",", " ").split()
        values = [float(value) for value in parts]
        if len(values) == 16:
            pose = np.asarray(values, dtype=np.float32).reshape(4, 4)
        elif len(values) == 17:
            pose = np.asarray(values[1:], dtype=np.float32).reshape(4, 4)
        elif len(values) == 8:
            pose = _pose_from_translation_quaternion(*values[1:8])
        elif len(values) == 7:
            pose = _pose_from_translation_quaternion(*values)
        else:
            raise ValueError(f"{trajectory_path}:{line_no}: unsupported trajectory record with {len(values)} values")
        if not np.isfinite(pose).all():
            raise ValueError(f"{trajectory_path}:{line_no}: pose contains non-finite values")
        poses.append(pose.astype(np.float32))
    return poses


def _read_depth_png_meters(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 1000.0


def _load_ase_rgb_calibration() -> Any:
    try:
        from projectaria_tools.projects import ase
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "AriaSyntheticEnvironmentsPi3XDataset requires projectaria_tools for official ASE RGB calibration. "
            "Install projectaria_tools in the active environment."
        ) from exc
    return ase.get_ase_rgb_calibration()


def _transform_to_matrix(transform: Any) -> np.ndarray:
    if hasattr(transform, "to_matrix"):
        matrix = transform.to_matrix()
    else:
        matrix = transform
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform from ASE calibration, got {matrix.shape}")
    return matrix


def _projection_params(calibration: Any) -> np.ndarray:
    if not hasattr(calibration, "get_projection_params"):
        raise ValueError("ASE RGB calibration does not expose get_projection_params()")
    params = np.asarray(calibration.get_projection_params(), dtype=np.float32).reshape(-1)
    if params.size < 3:
        raise ValueError(f"ASE RGB calibration projection params are too short: {params.size}")
    return params


def _intrinsics_from_ase_calibration(calibration: Any) -> np.ndarray:
    params = _projection_params(calibration)
    # ASE uses the Project Aria fisheye model. The first projection parameters
    # are focal length and principal point; higher terms are fisheye distortion.
    focal, cx, cy = float(params[0]), float(params[1]), float(params[2])
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _ray_distance_to_planar_depth(distance: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = distance.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    ray_norm = np.sqrt(x * x + y * y + 1.0)
    return (distance / ray_norm).astype(np.float32)


def _pixel_rays_from_official_calibration(calibration: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
    if not hasattr(calibration, "unproject"):
        return None

    height, width = image_shape
    rays = np.zeros((height, width, 3), dtype=np.float32)
    for v in range(height):
        for u in range(width):
            ray = calibration.unproject(np.array([float(u), float(v)], dtype=np.float64))
            if ray is None:
                continue
            ray = np.asarray(ray, dtype=np.float32).reshape(-1)
            if ray.size < 3 or not np.isfinite(ray[:3]).all():
                continue
            norm = float(np.linalg.norm(ray[:3]))
            if norm > 1e-8:
                rays[v, u] = ray[:3] / norm
    return rays


def _ray_distance_to_planar_depth_with_calibration(
    distance: np.ndarray,
    calibration: Any,
    intrinsics: np.ndarray,
    pixel_ray_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    shape = tuple(distance.shape)
    if shape not in pixel_ray_cache:
        rays = _pixel_rays_from_official_calibration(calibration, shape)
        if rays is None:
            return _ray_distance_to_planar_depth(distance, intrinsics)
        pixel_ray_cache[shape] = rays
    return (distance * pixel_ray_cache[shape][..., 2]).astype(np.float32)


def generate_ase_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    chunks: list[str] | None = None,
    roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    scenes_root = _require_dir(_path_roots(roots).get("scenes", data_root), "roots.scenes")

    records = []
    for scene_dir in tqdm(_discover_scene_dirs(scenes_root, chunks), desc="[ASE] building index", unit="scene"):
        scene = scene_dir.relative_to(scenes_root).as_posix()
        rgb_dir = _require_dir(scene_dir / "rgb", f"{scene}.rgb")
        depth_dir = _require_dir(scene_dir / "depth", f"{scene}.depth")
        trajectory_path = _trajectory_path(scene_dir)
        if trajectory_path is None:
            raise FileNotFoundError(f"ASE trajectory not found: {trajectory_path}")

        instance_dir = scene_dir / "instances"
        depth_by_frame = {_frame_number_from_stem(path): path for path in sorted(depth_dir.glob("*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        instance_by_frame = (
            {_frame_number_from_stem(path): path for path in sorted(instance_dir.glob("*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            if instance_dir.is_dir()
            else {}
        )

        frames = []
        image_paths = sorted(path for path in rgb_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for image_path in image_paths:
            frame_no = _frame_number_from_stem(image_path)
            depth_path = depth_by_frame.get(frame_no)
            if depth_path is None:
                continue
            frames.append(
                {
                    "frame_no": frame_no,
                    "frame_id": f"{frame_no:07d}",
                    "image": _absolute(image_path),
                    "depth": _absolute(depth_path),
                    "instance": None if frame_no not in instance_by_frame else _absolute(instance_by_frame[frame_no]),
                }
            )
        frames.sort(key=lambda frame: frame["frame_no"])
        records.append(
            {
                "sequence_id": scene,
                "trajectory": _absolute(trajectory_path),
                "scene_language": _absolute(scene_dir / "ase_scene_language.txt")
                if (scene_dir / "ase_scene_language.txt").is_file()
                else None,
                "object_instances_to_classes": _absolute(scene_dir / "object_instances_to_classes.json")
                if (scene_dir / "object_instances_to_classes.json").is_file()
                else None,
                "semidense_points": _absolute(scene_dir / "semidense_points.csv.gz")
                if (scene_dir / "semidense_points.csv.gz").is_file()
                else None,
                "semidense_observations": _absolute(scene_dir / "semidense_observations.csv.gz")
                if (scene_dir / "semidense_observations.csv.gz").is_file()
                else None,
                "frames": frames,
            }
        )

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class AriaSyntheticEnvironmentsPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        chunks: list[str] | None = None,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "AriaSyntheticEnvironmentsPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scenes_root = _require_dir(component_roots.get("scenes", self.data_root), "roots.scenes")
        self.trajectory_cache: dict[str, list[np.ndarray]] = {}
        self.rgb_calibration = _load_ase_rgb_calibration()
        self.camera_intrinsics = _intrinsics_from_ase_calibration(self.rgb_calibration)
        self.device_from_camera = _transform_to_matrix(self.rgb_calibration.get_transform_device_camera())
        self.pixel_ray_cache: dict[tuple[int, int], np.ndarray] = {}

        if index_file is None:
            index = generate_ase_index(self.data_root, chunks=chunks, roots=roots)
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()

        selected_chunks = set(chunks or [])
        self.records = []
        self.frames = {}
        for record in index.get("sequences", []):
            scene = record["sequence_id"]
            chunk = Path(scene).parts[0]
            if selected_chunks and chunk not in selected_chunks:
                continue
            self.records.append(record)
            self.frames[scene] = record.get("frames", [])
        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {scene: len(frames) for scene, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scenes_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def _load_trajectory(self, record: dict[str, Any]) -> list[np.ndarray]:
        scene = record["sequence_id"]
        if scene not in self.trajectory_cache:
            self.trajectory_cache[scene] = load_ase_trajectory(Path(record["trajectory"]))
        return self.trajectory_cache[scene]

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        record = self.records[index]
        scene = record["sequence_id"]
        frames = self.frames.get(scene, [])
        if not frames:
            self.this_views_info = dict(scene=scene, idxs=[])
            return []

        should_replace = len(frames) < self.frame_num
        idxs = list(rng.choice(len(frames), self.frame_num, replace=should_replace))
        self.this_views_info = dict(scene=scene, idxs=idxs)
        poses = self._load_trajectory(record)

        views = []
        for idx in idxs:
            frame = frames[idx]
            frame_no = int(frame["frame_no"])
            if frame_no >= len(poses):
                raise KeyError(f"ASE pose not found for {scene} frame {frame_no}")

            image_path = Path(frame["image"])
            depth_path = Path(frame["depth"])
            img = _read_rgb_image(image_path)
            if img is None:
                continue

            ray_distance = _read_depth_png_meters(depth_path)
            intrinsics = self.camera_intrinsics.copy()
            pixel_rays = _pixel_rays_from_official_calibration(self.rgb_calibration, tuple(ray_distance.shape))
            if pixel_rays is not None:
                self.pixel_ray_cache[tuple(ray_distance.shape)] = pixel_rays
            depthmap = _ray_distance_to_planar_depth_with_calibration(
                ray_distance,
                self.rgb_calibration,
                intrinsics,
                self.pixel_ray_cache,
            )
            original_shape = tuple(depthmap.shape)
            camera_pose = poses[frame_no] @ self.device_from_camera
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]
            ray_fields = {}
            if tuple(depthmap.shape) == original_shape and pixel_rays is not None:
                ray_fields = {
                    "pixel_rays": pixel_rays.astype(np.float32),
                    "ray_distance": ray_distance.astype(np.float32),
                }

            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": camera_pose.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": frame["frame_id"],
                    "prefix": f"{scene}_{frame['frame_id']}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "instance_path": None if frame.get("instance") is None else str(Path(frame["instance"])),
                    "scene_language_path": None
                    if record.get("scene_language") is None
                    else str(Path(record["scene_language"])),
                    "depth_source": "native_gt_dense",
                    "depth_definition": "ray_distance_m_with_pixel_rays",
                    "pose_source": "native_gt",
                    "intrinsics_source": "native_gt",
                    "camera_model": "aria_rgb_fisheye_official_calibration",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                    **ray_fields,
                }
            )
        return views
