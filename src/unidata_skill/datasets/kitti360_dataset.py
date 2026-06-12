from __future__ import annotations

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


KITTI360_CAMERAS = ("image_00", "image_01")
DEFAULT_LIDAR_ROOT_NAME = "data_3d_raw"


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


def _parse_numeric_record(line: str, path: Path, line_no: int) -> tuple[str, np.ndarray] | None:
    line = _strip_inline_comment(line)
    if not line.strip() or ":" not in line:
        return None
    name, values = line.split(":", 1)
    return name.strip(), np.array(_parse_float_tokens(values.split(), path, line_no), dtype=np.float32)


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


def _resolve_existing_path(data_root: Path, value: str | Path, name: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, data_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found: {candidates[-1]}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_perspective_calibration(calibration_root: Path) -> dict[str, dict[str, np.ndarray]]:
    path = calibration_root / "perspective.txt"
    records: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_matrix_line(line, path, line_no)
        if parsed is not None:
            records[parsed[0]] = parsed[1]

    intrinsics: dict[str, np.ndarray] = {}
    projections: dict[str, np.ndarray] = {}
    rectifications: dict[str, np.ndarray] = {}
    for camera_id, key in (("image_00", "P_rect_00"), ("image_01", "P_rect_01")):
        if key not in records:
            continue
        projections[camera_id] = records[key].astype(np.float32)
        intrinsics[camera_id] = records[key][:3, :3].astype(np.float32)
        rect_key = f"R_rect_{camera_id[-2:]}"
        rectifications[camera_id] = records.get(rect_key, records.get("R_rect_00", np.eye(3, dtype=np.float32))).astype(np.float32)
    return {"intrinsics": intrinsics, "projections": projections, "rectifications": rectifications}


def load_perspective_intrinsics(calibration_root: Path) -> dict[str, np.ndarray]:
    return load_perspective_calibration(calibration_root)["intrinsics"]


def _load_numeric_records(path: Path) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_numeric_record(line, path, line_no)
        if parsed is not None:
            records[parsed[0]] = parsed[1]
    return records


def load_cam0_to_velodyne(calibration_root: Path) -> np.ndarray:
    path = calibration_root / "calib_cam_to_velo.txt"
    if not path.is_file():
        raise FileNotFoundError(f"calib_cam_to_velo.txt not found: {path}")
    records = _load_numeric_records(path)
    if "R" not in records or "T" not in records:
        raise ValueError(f"{path}: expected R and T records")
    if records["R"].size != 9 or records["T"].size != 3:
        raise ValueError(f"{path}: expected R with 9 values and T with 3 values")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = records["R"].reshape(3, 3)
    transform[:3, 3] = records["T"].reshape(3)
    if not np.isfinite(transform).all():
        raise ValueError(f"{path}: calibration contains non-finite values")
    return transform


def make_rectified_lidar_to_camera_transform(cam0_to_velodyne: np.ndarray, rectification: np.ndarray) -> np.ndarray:
    velodyne_to_cam0 = np.linalg.inv(cam0_to_velodyne).astype(np.float32)
    rectified = np.eye(4, dtype=np.float32)
    rectified[:3, :3] = rectification.astype(np.float32)
    return (rectified @ velodyne_to_cam0).astype(np.float32)


def read_velodyne_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if values.size % 4 == 0:
        return values.reshape(-1, 4)[:, :3].astype(np.float32)
    if values.size % 3 == 0:
        return values.reshape(-1, 3).astype(np.float32)
    raise ValueError(f"{path}: expected float32 xyz or xyzi point records")


def project_lidar_points_to_depth_image(
    points_lidar: np.ndarray,
    lidar_to_camera_rect: np.ndarray,
    projection: np.ndarray,
    image_shape: tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float | None = None,
) -> np.ndarray:
    height, width = image_shape
    depth_flat = np.full(height * width, np.inf, dtype=np.float32)
    if points_lidar.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    points_h = np.concatenate([points_lidar[:, :3], np.ones((points_lidar.shape[0], 1), dtype=np.float32)], axis=1)
    points_camera = points_h @ lidar_to_camera_rect.T
    depth = points_camera[:, 2]
    finite = np.isfinite(points_camera).all(axis=1) & (depth > min_depth)
    if max_depth is not None:
        finite &= depth <= max_depth
    if not finite.any():
        return np.zeros((height, width), dtype=np.float32)

    points_camera = points_camera[finite]
    depth = depth[finite]
    projected = points_camera @ projection.T
    denom = projected[:, 2]
    valid = np.isfinite(projected).all(axis=1) & (denom > min_depth)
    if not valid.any():
        return np.zeros((height, width), dtype=np.float32)

    u = np.floor(projected[valid, 0] / denom[valid] + 0.5).astype(np.int32)
    v = np.floor(projected[valid, 1] / denom[valid] + 0.5).astype(np.int32)
    z = depth[valid].astype(np.float32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return np.zeros((height, width), dtype=np.float32)

    flat = v[inside] * width + u[inside]
    np.minimum.at(depth_flat, flat, z[inside])
    depth_flat[~np.isfinite(depth_flat)] = 0.0
    return depth_flat.reshape(height, width)


def find_velodyne_path(lidar_root: Path, sequence: str, frame_id: int) -> Path:
    candidates = (
        lidar_root / sequence / "velodyne_points" / "data" / f"{frame_id:010d}.bin",
        lidar_root / sequence / "velodyne_points" / "data" / f"{frame_id}.bin",
        lidar_root / sequence / "velodyne" / f"{frame_id:010d}.bin",
        lidar_root / sequence / "velodyne" / f"{frame_id}.bin",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Velodyne point cloud not found for {sequence} frame {frame_id:010d} under {lidar_root}")


def resolve_lidar_root(data_root: Path, component_roots: dict[str, Path], optional_roots: dict[str, Path | None]) -> Path | None:
    lidar_root = component_roots.get("lidar")
    if lidar_root is None:
        optional_lidar_root = optional_roots.get("lidar")
        if optional_lidar_root is not None:
            lidar_root = optional_lidar_root
    if lidar_root is None:
        default_root = data_root / DEFAULT_LIDAR_ROOT_NAME
        lidar_root = default_root if default_root.is_dir() else None
    if lidar_root is not None:
        _require_dir(lidar_root, "roots.lidar")
    return lidar_root

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


def generate_kitti360_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    sequences: list[str] | None = None,
    cameras: tuple[str, ...] = ("image_00",),
    roots: dict[str, str | Path] | None = None,
    optional_roots: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    component_roots = _path_roots(roots)
    optional_roots = _optional_path_roots(optional_roots)
    calibration_root = _require_dir(component_roots.get("calibration", data_root / "calibration"), "roots.calibration")
    images_root = _require_dir(component_roots.get("images", data_root / "data_2d_raw"), "roots.images")
    poses_root = _require_dir(component_roots.get("poses", data_root / "data_poses"), "roots.poses")
    perspective_calibration = load_perspective_calibration(calibration_root)
    intrinsics = perspective_calibration["intrinsics"]
    projections = perspective_calibration["projections"]
    rectifications = perspective_calibration["rectifications"]
    cam0_to_velodyne = load_cam0_to_velodyne(calibration_root)
    lidar_to_camera_rect = {
        camera_id: make_rectified_lidar_to_camera_transform(cam0_to_velodyne, rectification)
        for camera_id, rectification in rectifications.items()
    }
    del optional_roots

    sequence_names = sequences or discover_sequences(images_root)
    records = []
    for sequence in tqdm(sequence_names, desc="[KITTI360] building index", unit="sequence"):
        sequence_poses = load_cam0_to_world(poses_root, sequence)
        frames = []
        for camera_id in cameras:
            if camera_id not in intrinsics or camera_id not in projections or camera_id not in lidar_to_camera_rect:
                continue
            image_dir = images_root / sequence / camera_id / "data_rect"
            if not image_dir.is_dir():
                continue
            for image_path in sorted(image_dir.glob("*.png")):
                frame_id = int(image_path.stem)
                pose = sequence_poses.get(frame_id)
                if pose is None:
                    continue
                frames.append(
                    {
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "image": _relative(image_path, images_root),
                        "camera_pose": pose.astype(np.float32).tolist(),
                    }
                )
        frames.sort(key=lambda item: (item["frame_id"], item["camera_id"]))
        records.append({"sequence_id": sequence, "frames": frames})

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class Kitti360Pi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        sequences: list[str] | None = None,
        cameras: tuple[str, ...] = ("image_00",),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "KITTI360Pi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.calibration_root = _require_dir(component_roots.get("calibration", self.data_root / "calibration"), "roots.calibration")
        self.images_root = _require_dir(component_roots.get("images", self.data_root / "data_2d_raw"), "roots.images")
        self.poses_root = _require_dir(component_roots.get("poses", self.data_root / "data_poses"), "roots.poses")
        self.lidar_root = resolve_lidar_root(self.data_root, component_roots, self.optional_roots)
        self.cameras = cameras
        self.perspective_calibration = load_perspective_calibration(self.calibration_root)
        self.intrinsics = self.perspective_calibration["intrinsics"]
        self.projections = self.perspective_calibration["projections"]
        self.cam0_to_velodyne = load_cam0_to_velodyne(self.calibration_root)
        self.lidar_to_camera_rect = {
            camera_id: make_rectified_lidar_to_camera_transform(self.cam0_to_velodyne, rectification)
            for camera_id, rectification in self.perspective_calibration["rectifications"].items()
        }

        if index_file is None:
            index = generate_kitti360_index(
                self.data_root,
                sequences=sequences,
                cameras=cameras,
                roots=roots,
                optional_roots=optional_roots,
            )
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()

        selected = set(sequences or [])
        self.sequences = []
        self.frames = {}
        for record in index.get("sequences", []):
            sequence = record["sequence_id"]
            if selected and sequence not in selected:
                continue
            frames = record.get("frames", [])
            if cameras is not None:
                allowed = set(cameras)
                frames = [frame for frame in frames if frame["camera_id"] in allowed]
            self.sequences.append(sequence)
            self.frames[sequence] = frames
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        if len(resolution) == 1 and isinstance(resolution[0], (list, tuple)):
            resolution = list(resolution[0])
        scene = self.sequences[index]
        frames = self.frames.get(scene, [])
        num_imgs = len(frames)

        if num_imgs == 0:
            self.this_views_info = dict(scene=scene, idxs=[])
            return []

        should_replace = num_imgs < self.frame_num
        idxs = list(rng.choice(num_imgs, self.frame_num, replace=should_replace))

        self.this_views_info = dict(
            scene=scene,
            idxs=idxs,
        )

        views = []
        for idx in idxs:
            frame = frames[idx]
            image_path = self.images_root / frame["image"]
            camera_id = frame["camera_id"]
            frame_id = int(frame["frame_id"])
            img = _read_rgb_image(image_path)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}", flush=True)
                continue
            height, width = img.shape[:2]
            if self.lidar_root is None:
                raise FileNotFoundError(
                    "KITTI-360 projected depth requires roots.lidar, optional_roots.lidar, "
                    f"or {self.data_root / DEFAULT_LIDAR_ROOT_NAME}"
                )
            velodyne_path = find_velodyne_path(self.lidar_root, scene, frame_id)
            points_lidar = read_velodyne_points(velodyne_path)
            depthmap = project_lidar_points_to_depth_image(
                points_lidar,
                self.lidar_to_camera_rect[camera_id],
                self.projections[camera_id],
                (height, width),
            )
            if not np.any(depthmap > 0):
                raise ValueError(f"no projected lidar depth for {image_path}")
            intrinsics = self.intrinsics[camera_id].copy()
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
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "camera_pose": np.array(frame["camera_pose"], dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": f"{camera_id}_{frame_id:010d}.png",
                    "prefix": f"{scene}_{camera_id}_{frame_id:010d}",
                    "image_path": str(image_path),
                    "depth_source": "projected_lidar_sparse",
                    "depth_path": str(velodyne_path),
                    "sparse_depth": depthmap.astype(np.float32).copy(),
                }
            )
        return views
