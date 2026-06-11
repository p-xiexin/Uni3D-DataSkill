from __future__ import annotations

import re
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
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _frame_id_from_color(path: Path) -> str:
    suffix = "_color_vis"
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem.split("_", 1)[0]


def _frame_id_from_depth(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _parse_sage_opencv_matrix(camera_path: Path, matrix_name: str, shape: tuple[int, int]) -> np.ndarray:
    text = camera_path.read_text(encoding="utf-8")
    section_match = re.search(rf"^{re.escape(matrix_name)}:\s*(.*?)(?=^[A-Za-z_][A-Za-z0-9_]*:|\Z)", text, re.M | re.S)
    if section_match is None:
        raise ValueError(f"{camera_path}: missing {matrix_name} matrix")

    section = section_match.group(1)
    rows_match = re.search(r"^\s*rows:\s*(\d+)\s*$", section, re.M)
    cols_match = re.search(r"^\s*cols:\s*(\d+)\s*$", section, re.M)
    data_match = re.search(r"^\s*data:\s*\[(.*?)\]\s*$", section, re.M | re.S)
    if rows_match is None or cols_match is None or data_match is None:
        raise ValueError(f"{camera_path}: malformed {matrix_name} matrix")

    rows = int(rows_match.group(1))
    cols = int(cols_match.group(1))
    if (rows, cols) != shape:
        raise ValueError(f"{camera_path}: expected {matrix_name} shape {shape}, got {(rows, cols)}")

    values = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", data_match.group(1))]
    expected = rows * cols
    if len(values) != expected:
        raise ValueError(f"{camera_path}: expected {expected} {matrix_name} values, got {len(values)}")
    matrix = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{camera_path}: {matrix_name} contains non-finite values")
    return matrix


def load_sage_camera(camera_path: Path) -> np.ndarray:
    """Parse SAGE camera.yaml and return the OpenCV 3x3 camera matrix."""
    return _parse_sage_opencv_matrix(camera_path, "K", (3, 3))


def load_sage_trajectory(traj_path: Path) -> dict[str, np.ndarray]:
    """Parse SAGE trj_0.txt and return frame_id -> OpenCV camera-to-world pose."""
    poses: dict[str, np.ndarray] = {}
    pose_index = 0
    for line_number, line in enumerate(traj_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(item) for item in stripped.split()]
        if len(values) != 16:
            raise ValueError(f"{traj_path}:{line_number}: expected 16 pose values, got {len(values)}")
        pose = np.asarray(values, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise ValueError(f"{traj_path}:{line_number}: pose contains non-finite values")
        if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
            raise ValueError(f"{traj_path}:{line_number}: invalid homogeneous pose last row")
        poses[f"{pose_index:08d}"] = pose
        pose_index += 1
    if not poses:
        raise ValueError(f"{traj_path}: no poses found")
    return poses


def read_sage_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


@dataclass(frozen=True)
class SageRoute:
    sequence_id: str
    domain: str
    layout: str
    setting: str
    route: str
    route_dir: Path
    color_dir: Path
    depth_dir: Path
    camera_path: Path
    trajectory_path: Path


@dataclass(frozen=True)
class SageFrame:
    frame_id: str
    color_path: Path
    depth_path: Path


class SagePi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        domains: list[str] | None = None,
        settings: list[str] | None = None,
        route_ids: list[str] | None = None,
        frame_num: int = 8,
        stride: int = 1,
        resolution: list[int] | tuple[int, int] = (512, 384),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(resolution=[list(resolution)], frame_num=frame_num, shuffle=False, **kwargs)
        self.dataset_label = "SagePi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scenes_root = _require_dir(component_roots.get("scenes", self.data_root), "roots.scenes")
        self.domains = set(domains or [])
        self.settings = set(settings or [])
        self.route_ids = set(route_ids or [])
        self.stride = stride
        self.verbose = verbose

        discovered_routes = self._discover_routes()
        self.frames = {route.sequence_id: self._build_route_frames(route) for route in discovered_routes}
        self.routes = [route for route in discovered_routes if self.frames[route.sequence_id]]
        self.sequences = [route.sequence_id for route in self.routes]
        self.num_imgs = {sequence: len(frames) for sequence, frames in self.frames.items()}
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scenes_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.routes)

    def _discover_routes(self) -> list[SageRoute]:
        routes: list[SageRoute] = []
        for domain_dir in sorted(path for path in self.scenes_root.iterdir() if path.is_dir()):
            domain = domain_dir.name
            if self.domains and domain not in self.domains:
                continue
            for layout_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
                layout = layout_dir.name
                for setting_dir in sorted(path for path in layout_dir.iterdir() if path.is_dir()):
                    setting = setting_dir.name
                    if self.settings and setting not in self.settings:
                        continue
                    for route_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("route_")):
                        route_id = route_dir.name
                        if self.route_ids and route_id not in self.route_ids:
                            continue
                        color_dir = route_dir / "vis" / "color"
                        depth_dir = route_dir / "vis" / "depth"
                        camera_path = route_dir / "camera.yaml"
                        trajectory_path = route_dir / "trj_0.txt"
                        if not color_dir.is_dir() or not depth_dir.is_dir():
                            continue
                        if not camera_path.is_file() or not trajectory_path.is_file():
                            continue
                        sequence_id = f"{domain}/{layout}/{setting}/{route_id}"
                        routes.append(
                            SageRoute(
                                sequence_id=sequence_id,
                                domain=domain,
                                layout=layout,
                                setting=setting,
                                route=route_id,
                                route_dir=route_dir,
                                color_dir=color_dir,
                                depth_dir=depth_dir,
                                camera_path=camera_path,
                                trajectory_path=trajectory_path,
                            )
                        )
        return routes

    def _build_route_frames(self, route: SageRoute) -> list[SageFrame]:
        depths_by_frame: dict[str, Path] = {}
        for depth_path in sorted(path for path in route.depth_dir.iterdir() if path.suffix.lower() in {".tif", ".tiff"}):
            depths_by_frame.setdefault(_frame_id_from_depth(depth_path), depth_path)
        frames: list[SageFrame] = []
        for color_path in sorted(route.color_dir.glob("*_color_vis.png")):
            frame_id = _frame_id_from_color(color_path)
            depth_path = depths_by_frame.get(frame_id)
            if depth_path is not None:
                frames.append(SageFrame(frame_id=frame_id, color_path=color_path, depth_path=depth_path))
        return frames

    def _window_indices(self, num_imgs: int, rng: np.random.Generator) -> range:
        required_span = (self.frame_num - 1) * self.stride + 1
        if num_imgs <= required_span:
            return range(0, num_imgs, self.stride)
        begin = int(rng.integers(0, num_imgs - required_span + 1))
        return range(begin, begin + required_span, self.stride)

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        route = self.routes[index]
        frames = self.frames.get(route.sequence_id, [])
        if not frames:
            self.this_views_info = dict(scene=route.sequence_id, idxs=[])
            return []

        intrinsics = load_sage_camera(route.camera_path)
        poses = load_sage_trajectory(route.trajectory_path)
        idxs = self._window_indices(len(frames), rng)
        self.this_views_info = dict(scene=route.sequence_id, idxs=list(idxs))

        views = []
        for idx in idxs:
            frame = frames[idx]
            img = _read_rgb_image(frame.color_path)
            if img is None:
                continue
            if frame.frame_id not in poses:
                raise KeyError(f"SAGE pose not found for {route.sequence_id} frame {frame.frame_id}")
            depthmap = read_sage_depth(frame.depth_path)
            camera_intrinsics = intrinsics.copy()
            img, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                camera_intrinsics,
                resolution,
                rng=rng,
                info=str(frame.color_path),
            )[:3]
            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": poses[frame.frame_id].astype(np.float32),
                    "camera_intrinsics": camera_intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": route.sequence_id,
                    "instance": frame.frame_id,
                    "prefix": f"{route.sequence_id}_{frame.frame_id}",
                    "image_path": str(frame.color_path),
                    "depth_path": str(frame.depth_path),
                    "route_dir": str(route.route_dir),
                    "domain": route.domain,
                    "layout": route.layout,
                    "setting": route.setting,
                    "route": route.route,
                    "depth_source": "rendered",
                    "pose_source": "rendered",
                    "intrinsics_source": "metadata",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
