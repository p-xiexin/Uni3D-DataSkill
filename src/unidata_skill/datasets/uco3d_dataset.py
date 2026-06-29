from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from datasets.base.base_dataset import BaseDataset


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return path


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _opencv_camera_to_c2w(rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = rotation.astype(np.float32)
    world_to_camera[:3, 3] = tvec.reshape(3).astype(np.float32)
    return np.linalg.inv(world_to_camera).astype(np.float32)


class UCO3DPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        subsets: list[str] | None = None,
        subset_lists_name: str = "set_lists_3categories-debug.sqlite",
        set_lists_file: str | Path | None = None,
        pick_sequences: list[str] | None = None,
        limit_sequences_to: int = 0,
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        super().__init__(**kwargs)
        try:
            from uco3d import UCO3DDataset, UCO3DFrameDataBuilder, opencv_cameras_projection_from_uco3d
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "UCO3DPi3XDataset requires the official facebookresearch/uco3d package. "
                "Install it separately and point data_root to the uCO3D dataset root."
            ) from exc

        self._opencv_cameras_projection_from_uco3d = opencv_cameras_projection_from_uco3d
        self.dataset_label = "UCO3DPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.metadata_file = _require_file(component_roots.get("metadata", self.data_root / "metadata.sqlite"), "roots.metadata")
        if set_lists_file is None:
            set_lists_file = component_roots.get("set_lists", self.data_root / "set_lists" / subset_lists_name)
        self.set_lists_file = _require_file(Path(set_lists_file), "roots.set_lists")
        self.subsets = subsets or ["train"]

        builder = UCO3DFrameDataBuilder(
            dataset_root=str(self.data_root),
            apply_alignment=True,
            load_images=True,
            load_depths=True,
            load_depth_masks=True,
            load_masks=False,
            load_frames_from_videos=True,
            undistort_loaded_blobs=True,
            box_crop=False,
        )
        self.uco3d_dataset = UCO3DDataset(
            sqlite_metadata_file=str(self.metadata_file),
            subset_lists_file=str(self.set_lists_file),
            subsets=self.subsets,
            frame_data_builder=builder,
            pick_sequences=tuple(pick_sequences or ()),
            limit_sequences_to=limit_sequences_to,
        )
        self.sequences = list(self.uco3d_dataset.sequence_names())
        self.sequence_indices = {
            sequence: list(self.uco3d_dataset.sequence_indices_in_order(sequence)) for sequence in self.sequences
        }
        self.num_imgs = {sequence: len(indices) for sequence, indices in self.sequence_indices.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] Sequences of {self.dataset_label} dataset:", self.sequences)
        print(f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root}", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_views(self, index: int, resolution: list[int], rng: np.random.Generator, is_test: bool = False) -> list[dict[str, Any]]:
        scene = self.sequences[index]
        dataset_indices = self.sequence_indices.get(scene, [])
        if not dataset_indices:
            self.this_views_info = dict(scene=scene, idxs=[])
            return []
        should_replace = len(dataset_indices) < self.frame_num
        idxs = list(rng.choice(len(dataset_indices), self.frame_num, replace=should_replace))
        self.this_views_info = dict(scene=scene, idxs=idxs)

        views = []
        for local_idx in idxs:
            frame_data = self.uco3d_dataset[dataset_indices[local_idx]]
            if frame_data.image_rgb is None or frame_data.depth_map is None or frame_data.camera is None:
                continue
            img = (_to_numpy(frame_data.image_rgb).transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            depthmap = _to_numpy(frame_data.depth_map)[0].astype(np.float32)
            if frame_data.depth_mask is not None:
                depthmap *= (_to_numpy(frame_data.depth_mask)[0] > 0).astype(np.float32)
            image_size_hw = frame_data.effective_image_size_hw
            if image_size_hw is None:
                image_size_hw = frame_data.image_size_hw
            if image_size_hw is None:
                continue
            rotation, tvec, camera_matrix = self._opencv_cameras_projection_from_uco3d(
                frame_data.camera,
                image_size=image_size_hw[None],
            )
            rotation = _to_numpy(rotation)[0]
            tvec = _to_numpy(tvec)[0]
            intrinsics = _to_numpy(camera_matrix)[0].astype(np.float32)
            camera_pose = _opencv_camera_to_c2w(rotation, tvec)
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(frame_data.image_path),
            )[:3]
            frame_number = int(_to_numpy(frame_data.frame_number).reshape(-1)[0])
            instance = f"{frame_data.sequence_name}_{frame_number}"
            views.append(
                {
                    "img": img,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": camera_pose.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": instance,
                    "prefix": instance,
                    "image_path": str(Path(frame_data.image_path).resolve()),
                    "depth_path": str(Path(frame_data.depth_path).resolve()),
                    "depth_source": "native_pseudo_dense",
                    "pose_source": "native_pseudo",
                    "intrinsics_source": "metadata",
                    "pseudo_label": True,
                    "valid_mask_required": True,
                }
            )
        return views
