from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

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


def _resolve_existing_path(data_root: Path, value: str | Path, name: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, data_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found: {candidates[-1]}")


def _as_resolution(resolution: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(resolution) == 1 and isinstance(resolution[0], (list, tuple)):  # type: ignore[index]
        resolution = resolution[0]  # type: ignore[assignment]
    return int(resolution[0]), int(resolution[1])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _quaternion_wxyz_to_rotation(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(item) for item in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _transform(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = np.array(list(translation), dtype=np.float32)
    return pose


def _resolve_index_data_path(filename: str) -> str:
    return Path(filename).as_posix()


def _resolve_data_path(data_blob_root: Path, samples_root: Path | None, filename: str) -> Path:
    rel_path = Path(filename)
    parts = rel_path.parts
    if samples_root is not None and parts and parts[0] == "samples":
        return samples_root.joinpath(*parts[1:])
    return data_blob_root / rel_path


def generate_nuscenes_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    version: str = "v1.0-mini",
    cameras: tuple[str, ...] | None = None,
    roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    component_roots = _path_roots(roots)
    table_root = _require_dir(component_roots.get("tables", data_root / version), "roots.tables")

    scenes = {item["token"]: item for item in _read_json(table_root / "scene.json")}
    samples = {item["token"]: item for item in _read_json(table_root / "sample.json")}
    sample_data = _read_json(table_root / "sample_data.json")
    calibrated = {item["token"]: item for item in _read_json(table_root / "calibrated_sensor.json")}
    ego_poses = {item["token"]: item for item in _read_json(table_root / "ego_pose.json")}
    sensors = {item["token"]: item for item in _read_json(table_root / "sensor.json")}
    sample_to_scene = {token: scenes[sample["scene_token"]]["name"] for token, sample in samples.items()}

    frames_by_scene: dict[str, list[dict[str, Any]]] = {}
    for item in tqdm(sample_data, desc="[nuScenes] building index", unit="sample_data"):
        calib = calibrated[item["calibrated_sensor_token"]]
        sensor = sensors[calib["sensor_token"]]
        channel = sensor.get("channel", "")
        if sensor.get("modality") != "camera" or not channel.startswith("CAM_"):
            continue
        if cameras is not None and channel not in cameras:
            continue
        scene = sample_to_scene.get(item["sample_token"])
        if scene is None:
            continue

        ego = ego_poses[item["ego_pose_token"]]
        camera_to_ego = _transform(_quaternion_wxyz_to_rotation(calib["rotation"]), calib["translation"])
        ego_to_global = _transform(_quaternion_wxyz_to_rotation(ego["rotation"]), ego["translation"])
        camera_to_global = ego_to_global @ camera_to_ego
        frames_by_scene.setdefault(scene, []).append(
            {
                "channel": channel,
                "frame_id": str(item.get("timestamp", item["token"])),
                "image": _resolve_index_data_path(item["filename"]),
                "camera_intrinsics": np.array(calib["camera_intrinsic"], dtype=np.float32).tolist(),
                "camera_pose": camera_to_global.astype(np.float32).tolist(),
                "sample_token": item["sample_token"],
                "sample_data_token": item["token"],
            }
        )

    records = []
    for scene in sorted(frames_by_scene):
        frames = sorted(frames_by_scene[scene], key=lambda frame: (frame["frame_id"], frame["channel"]))
        records.append({"sequence_id": scene, "frames": frames})

    index = {"version": 1, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class NuScenesPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        version: str = "v1.0-mini",
        cameras: tuple[str, ...] | None = None,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        self.dataset_label = "NuScenesPi3X"
        self.data_root = Path(data_root)
        self.version = version
        self.cameras = cameras
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.table_root = _require_dir(component_roots.get("tables", self.data_root / version), "roots.tables")
        self.data_blob_root = _require_dir(component_roots.get("data", self.data_root), "roots.data")
        self.samples_root = component_roots.get("samples")
        if self.samples_root is not None:
            _require_dir(self.samples_root, "roots.samples")

        if index_file is None:
            index = generate_nuscenes_index(self.data_root, version=version, cameras=cameras, roots=roots)
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = json.loads(index_file_path.read_text(encoding="utf-8"))

        self.sequences = []
        self.frames = {}
        for record in index.get("sequences", []):
            scene = record["sequence_id"]
            frames = record.get("frames", [])
            if cameras is not None:
                frames = [frame for frame in frames if frame["channel"] in cameras]
            self.sequences.append(scene)
            self.frames[scene] = frames
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

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
        target_width, target_height = _as_resolution(resolution)
        for idx in idxs:
            frame = frames[idx]
            image_path = _resolve_data_path(self.data_blob_root, self.samples_root, frame["image"])
            img = _read_rgb_image(image_path)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}", flush=True)
                continue

            height, width = img.shape[:2]
            depthmap = np.ones((height, width), dtype=np.float32)
            intrinsics = np.array(frame["camera_intrinsics"], dtype=np.float32)
            view = {
                "img": img,
                "depthmap": depthmap,
                "camera_pose": np.array(frame["camera_pose"], dtype=np.float32),
                "camera_intrinsics": intrinsics.astype(np.float32),
                "dataset": self.dataset_label,
                "label": scene,
                "instance": f"{frame['channel']}_{frame['frame_id']}{image_path.suffix}",
                "prefix": f"{scene}_{frame['channel']}_{frame['frame_id']}",
                "image_path": str(image_path),
                "depth_source": "placeholder_missing_dense_depth",
                "sample_token": frame["sample_token"],
                "sample_data_token": frame["sample_data_token"],
            }
            img2, depth2, intrinsics2 = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                (target_width, target_height),
                rng=rng,
                info=str(image_path),
            )[:3]
            view["img"] = img2
            view["depthmap"] = depth2.astype(np.float32)
            view["camera_intrinsics"] = intrinsics2.astype(np.float32)
            views.append(view)
        return views
