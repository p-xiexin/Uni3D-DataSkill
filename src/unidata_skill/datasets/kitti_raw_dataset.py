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


EARTH_RADIUS_METERS = 6378137.0


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


def _read_kitti_depth_png(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 256.0


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


def _parse_kitti_calib(path: Path) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_inline_comment(line)
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = _parse_float_tokens(value.split(), path, line_no)
        if len(values) == 12:
            records[key.strip()] = np.asarray(values, dtype=np.float32).reshape(3, 4)
        elif len(values) == 9:
            records[key.strip()] = np.asarray(values, dtype=np.float32).reshape(3, 3)
        elif len(values) == 3:
            records[key.strip()] = np.asarray(values, dtype=np.float32)
        else:
            raise ValueError(f"{path}:{line_no}: unsupported calibration value count for {key.strip()!r}: {len(values)}")
    return records


def _camera_name(camera: str) -> str:
    suffix = int(camera.split("_")[-1])
    return f"image_{suffix:02d}"


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def _oxts_packet_to_pose(packet: np.ndarray, scale: float) -> np.ndarray:
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    tx = scale * lon * math.pi * EARTH_RADIUS_METERS / 180.0
    ty = scale * EARTH_RADIUS_METERS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    tz = alt
    rotation = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    return _make_transform(rotation, np.asarray([tx, ty, tz], dtype=np.float64))


def _parse_oxts_poses(oxts_dir: Path) -> list[np.ndarray]:
    packets = []
    for path in sorted(oxts_dir.glob("*.txt")):
        line = _strip_inline_comment(path.read_text(encoding="utf-8").strip())
        if not line:
            continue
        values = np.asarray([float(item) for item in line.split()], dtype=np.float64)
        if values.size < 6:
            raise ValueError(f"{path}: expected at least 6 OXTS values, got {values.size}")
        packets.append(values)
    if not packets:
        return []

    scale = math.cos(float(packets[0][0]) * math.pi / 180.0)
    global_poses = [_oxts_packet_to_pose(packet, scale) for packet in packets]
    origin_inv = np.linalg.inv(global_poses[0])
    return [(origin_inv @ pose).astype(np.float32) for pose in global_poses]


def _camera_to_imu_transform(date_root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cam_calib = _parse_kitti_calib(date_root / "calib_cam_to_cam.txt")
    velo_calib = _parse_kitti_calib(date_root / "calib_velo_to_cam.txt")
    imu_calib = _parse_kitti_calib(date_root / "calib_imu_to_velo.txt")

    r_rect = np.eye(4, dtype=np.float64)
    r_rect[:3, :3] = np.asarray(cam_calib.get("R_rect_00", np.eye(3)), dtype=np.float64)
    cam0_from_velo = _make_transform(np.asarray(velo_calib["R"], dtype=np.float64), np.asarray(velo_calib["T"], dtype=np.float64))
    velo_from_imu = _make_transform(np.asarray(imu_calib["R"], dtype=np.float64), np.asarray(imu_calib["T"], dtype=np.float64))
    rect_cam0_from_imu = r_rect @ cam0_from_velo @ velo_from_imu

    camera_from_imu: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    for key, projection in cam_calib.items():
        if not key.startswith("P_rect_"):
            continue
        suffix = key.rsplit("_", 1)[-1]
        camera = f"image_{int(suffix):02d}"
        projection = np.asarray(projection, dtype=np.float64)
        intrinsics[camera] = projection[:3, :3].astype(np.float32)
        cam0_from_cam = np.eye(4, dtype=np.float64)
        cam0_from_cam[0, 3] = -float(projection[0, 3]) / float(projection[0, 0])
        camera_from_imu[camera] = np.linalg.inv(cam0_from_cam) @ rect_cam0_from_imu
    return camera_from_imu, intrinsics


def _discover_sequences(raw_root: Path, sequences: list[str] | None) -> list[str]:
    if sequences:
        return sequences
    names = []
    for date_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        names.extend(path.name for path in sorted(date_dir.glob("*_sync")) if path.is_dir())
    return names


def _depth_path(depth_root: Path, split: str, sequence: str, camera: str, frame_id: str) -> Path:
    return depth_root / split / sequence / "proj_depth" / "groundtruth" / camera / f"{frame_id}.png"


def generate_kitti_raw_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    sequences: list[str] | None = None,
    cameras: tuple[str, ...] = ("image_02",),
    splits: tuple[str, ...] = ("train",),
    roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    component_roots = _path_roots(roots)
    raw_root = _require_dir(component_roots.get("raw", data_root), "roots.raw")
    calibration_root = _require_dir(component_roots.get("calibration", raw_root), "roots.calibration")
    depth_root = _require_dir(component_roots.get("depth", data_root / "data_depth_annotated"), "roots.depth")
    sequence_names = _discover_sequences(raw_root, sequences)
    cameras = tuple(_camera_name(camera) for camera in cameras)

    records = []
    for sequence in tqdm(sequence_names, desc="[KITTIRaw] building index", unit="sequence"):
        date = sequence[:10]
        sequence_dir = raw_root / date / sequence
        if not sequence_dir.is_dir():
            continue
        date_root = calibration_root / date
        if not date_root.is_dir():
            continue

        camera_from_imu, intrinsics_by_camera = _camera_to_imu_transform(date_root)
        oxts_dir = _require_dir(sequence_dir / "oxts" / "data", f"{sequence}.oxts")
        poses_world_imu = _parse_oxts_poses(oxts_dir)
        frames = []
        for split in splits:
            for camera in cameras:
                if camera not in camera_from_imu or camera not in intrinsics_by_camera:
                    continue
                image_dir = sequence_dir / camera / "data"
                depth_dir = depth_root / split / sequence / "proj_depth" / "groundtruth" / camera
                if not image_dir.is_dir() or not depth_dir.is_dir():
                    continue
                imu_from_camera = np.linalg.inv(camera_from_imu[camera])
                image_paths = sorted(path for path in image_dir.glob("*.png"))
                for image_path in image_paths:
                    frame_no = int(image_path.stem)
                    if frame_no >= len(poses_world_imu):
                        continue
                    depth_path = _depth_path(depth_root, split, sequence, camera, image_path.stem)
                    if not depth_path.is_file():
                        continue
                    camera_pose = poses_world_imu[frame_no] @ imu_from_camera
                    frames.append(
                        {
                            "camera_id": camera,
                            "frame_id": image_path.stem,
                            "split": split,
                            "image": _relative(image_path, raw_root),
                            "depth": _relative(depth_path, depth_root),
                            "camera_intrinsics": intrinsics_by_camera[camera].tolist(),
                            "camera_pose": camera_pose.astype(np.float32).tolist(),
                        }
                    )
        frames.sort(key=lambda item: (item["frame_id"], item["camera_id"]))
        records.append({"sequence_id": sequence, "date": date, "frames": frames})

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class KittiRawPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        sequences: list[str] | None = None,
        cameras: tuple[str, ...] = ("image_02",),
        splits: tuple[str, ...] = ("train",),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "KITTIRawPi3X"
        self.data_root = Path(data_root)
        self.cameras = tuple(_camera_name(camera) for camera in cameras)
        self.splits = splits
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.raw_root = _require_dir(component_roots.get("raw", self.data_root), "roots.raw")
        self.depth_root = _require_dir(component_roots.get("depth", self.data_root / "data_depth_annotated"), "roots.depth")
        self.calibration_root = _require_dir(component_roots.get("calibration", self.raw_root), "roots.calibration")

        if index_file is None:
            index = generate_kitti_raw_index(self.data_root, sequences=sequences, cameras=self.cameras, splits=splits, roots=roots)
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()

        selected = set(sequences or [])
        self.records = []
        self.frames = {}
        for record in index.get("sequences", []):
            sequence = record["sequence_id"]
            if selected and sequence not in selected:
                continue
            self.records.append(record)
            self.frames[sequence] = record.get("frames", [])
        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.raw_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.records)

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

        views = []
        target_width, target_height = _as_resolution(resolution)
        for idx in idxs:
            frame = frames[idx]
            image_path = self.raw_root / frame["image"]
            depth_path = self.depth_root / frame["depth"]
            img = _read_rgb_image(image_path)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}", flush=True)
                continue

            depthmap = _read_kitti_depth_png(depth_path)
            intrinsics = np.asarray(frame["camera_intrinsics"], dtype=np.float32)
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                (target_width, target_height),
                rng=rng,
                info=str(image_path),
            )[:3]

            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": np.asarray(frame["camera_pose"], dtype=np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": f"{frame['camera_id']}_{frame['frame_id']}{image_path.suffix}",
                    "prefix": f"{scene}_{frame['camera_id']}_{frame['frame_id']}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_source": "native_gt_sparse",
                    "depth_definition": "kitti_depth_completion_groundtruth_m",
                    "pose_source": "native_gt",
                    "intrinsics_source": "native_gt",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
