from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
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


def _frame_id_from_color(path: Path) -> str:
    stem = path.stem
    suffix = "_color_vis"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem.split("_", 1)[0]


def _frame_id_from_depth(path: Path) -> str:
    return path.stem.split("_", 1)[0]


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


def load_sage_camera(camera_path: Path) -> np.ndarray:
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    values = camera.get("K", {}).get("data")
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError(f"{camera_path}: missing K.data with 9 values")
    intrinsics = np.asarray(values, dtype=np.float32).reshape(3, 3)
    if not np.isfinite(intrinsics).all():
        raise ValueError(f"{camera_path}: K contains non-finite values")
    return intrinsics


def load_sage_trajectory(traj_path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for index, line in enumerate(traj_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        values = [float(item) for item in line.split()]
        if len(values) != 16:
            raise ValueError(f"{traj_path}:{index + 1}: expected 16 pose values, got {len(values)}")
        pose = np.asarray(values, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise ValueError(f"{traj_path}:{index + 1}: pose contains non-finite values")
        poses[f"{len(poses):08d}"] = pose
    if not poses:
        raise ValueError(f"{traj_path}: no poses found")
    return poses


def read_sage_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def generate_sage_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    domains: list[str] | None = None,
    layouts: list[str] | None = None,
    settings: list[str] | None = None,
    route_ids: list[str] | None = None,
    roots: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    scenes_root = _require_dir(_path_roots(roots).get("scenes", data_root), "roots.scenes")
    domain_filter = set(domains or [])
    layout_filter = set(layouts or [])
    setting_filter = set(settings or [])
    route_filter = set(route_ids or [])

    route_dirs = []
    with tqdm(desc="[SAGE] discovering route dirs", unit="dir") as progress:
        progress.update(1)
        for domain_dir in sorted(path for path in scenes_root.iterdir() if path.is_dir()):
            progress.update(1)
            domain = domain_dir.name
            if domain_filter and domain not in domain_filter:
                continue
            for layout_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
                progress.update(1)
                layout = layout_dir.name
                if layout_filter and layout not in layout_filter:
                    continue
                for setting_dir in sorted(path for path in layout_dir.iterdir() if path.is_dir()):
                    progress.update(1)
                    setting = setting_dir.name
                    if setting_filter and setting not in setting_filter:
                        continue
                    for route_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir() and path.name.startswith("route_")):
                        progress.update(1)
                        route = route_dir.name
                        if route_filter and route not in route_filter:
                            continue
                        route_dirs.append((domain, layout, setting, route, route_dir))
                        progress.set_postfix(routes=len(route_dirs), refresh=False)

    sequences = []
    for domain, layout, setting, route, route_dir in tqdm(route_dirs, desc="[SAGE] building index", unit="route"):
        color_dir = route_dir / "vis" / "color"
        depth_dir = route_dir / "vis" / "depth"
        camera_path = route_dir / "camera.yaml"
        trajectory_path = route_dir / "trj_0.txt"
        if not color_dir.is_dir() or not depth_dir.is_dir() or not camera_path.is_file() or not trajectory_path.is_file():
            continue

        depths = {
            _frame_id_from_depth(path): path
            for path in sorted(depth_dir.iterdir())
            if path.suffix.lower() in {".tif", ".tiff"}
        }
        frames = []
        for color_path in sorted(color_dir.glob("*_color_vis.png")):
            frame_id = _frame_id_from_color(color_path)
            depth_path = depths.get(frame_id)
            if depth_path is None:
                continue
            frames.append(
                {
                    "frame_id": frame_id,
                    "color": _relative(color_path, scenes_root),
                    "depth": _relative(depth_path, scenes_root),
                }
            )
        if not frames:
            continue

        sequences.append(
            {
                "sequence_id": f"{domain}/{layout}/{setting}/{route}",
                "domain": domain,
                "layout": layout,
                "setting": setting,
                "route": route,
                "route_dir": _relative(route_dir, scenes_root),
                "camera": _relative(camera_path, scenes_root),
                "trajectory": _relative(trajectory_path, scenes_root),
                "frames": frames,
            }
        )

    index = {"version": 1, "sequences": sequences}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class SagePi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        domains: list[str] | None = None,
        layouts: list[str] | None = None,
        settings: list[str] | None = None,
        route_ids: list[str] | None = None,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.verbose = verbose
        self.dataset_label = "SagePi3X"
        self.data_root = Path(data_root)
        self.scenes_root = _require_dir(_path_roots(roots).get("scenes", self.data_root), "roots.scenes")
        self.optional_roots = _optional_path_roots(optional_roots)
        self.camera_cache: dict[str, np.ndarray] = {}
        self.trajectory_cache: dict[str, dict[str, np.ndarray]] = {}

        if index_file is None:
            index = generate_sage_index(
                self.data_root,
                domains=domains,
                layouts=layouts,
                settings=settings,
                route_ids=route_ids,
                roots=roots,
            )
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = json.loads(index_file_path.read_text(encoding="utf-8"))

        self.routes = []
        domain_filter = set(domains or [])
        layout_filter = set(layouts or [])
        setting_filter = set(settings or [])
        route_filter = set(route_ids or [])
        for route in index.get("sequences", []):
            if domain_filter and route["domain"] not in domain_filter:
                continue
            if layout_filter and route["layout"] not in layout_filter:
                continue
            if setting_filter and route["setting"] not in setting_filter:
                continue
            if route_filter and route["route"] not in route_filter:
                continue
            self.routes.append(route)

        self.sequences = [route["sequence_id"] for route in self.routes]
        self.num_imgs = {route["sequence_id"]: len(route["frames"]) for route in self.routes}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scenes_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.routes)

    def _load_camera(self, route: dict[str, Any]) -> np.ndarray:
        sequence_id = route["sequence_id"]
        if sequence_id not in self.camera_cache:
            self.camera_cache[sequence_id] = load_sage_camera(self.scenes_root / route["camera"])
        return self.camera_cache[sequence_id]

    def _load_trajectory(self, route: dict[str, Any]) -> dict[str, np.ndarray]:
        sequence_id = route["sequence_id"]
        if sequence_id not in self.trajectory_cache:
            self.trajectory_cache[sequence_id] = load_sage_trajectory(self.scenes_root / route["trajectory"])
        return self.trajectory_cache[sequence_id]

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        route = self.routes[index]
        frames = route["frames"]
        should_replace = len(frames) < self.frame_num
        idxs = list(rng.choice(len(frames), self.frame_num, replace=should_replace))
        self.this_views_info = dict(scene=route["sequence_id"], idxs=idxs)

        intrinsics = self._load_camera(route)
        poses = self._load_trajectory(route)

        views = []
        for idx in idxs:
            frame = frames[idx]
            frame_id = frame["frame_id"]
            image_path = self.scenes_root / frame["color"]
            depth_path = self.scenes_root / frame["depth"]
            img = _read_rgb_image(image_path)
            if img is None:
                continue
            if frame_id not in poses:
                raise KeyError(f"SAGE pose not found for {route['sequence_id']} frame {frame_id}")

            depthmap = read_sage_depth(depth_path)
            camera_intrinsics = intrinsics.copy()
            img, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                camera_intrinsics,
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]

            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": poses[frame_id].astype(np.float32),
                    "camera_intrinsics": camera_intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": route["sequence_id"],
                    "instance": frame_id,
                    "prefix": f"{route['sequence_id']}_{frame_id}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "route_dir": str(self.scenes_root / route["route_dir"]),
                    "domain": route["domain"],
                    "layout": route["layout"],
                    "setting": route["setting"],
                    "route": route["route"],
                    "depth_source": "rendered",
                    "pose_source": "rendered",
                    "intrinsics_source": "metadata",
                    "pseudo_label": False,
                    "valid_mask_required": True,
                }
            )
        return views
