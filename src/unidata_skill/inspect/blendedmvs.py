from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlendedMVSScene:
    scene_id: str
    root: Path
    image_count: int
    depth_count: int
    camera_count: int
    has_pair_file: bool


class BlendedMVSInspector:
    """Inspect a BlendedMVS/MVSNet-style dataset tree before conversion."""

    required_entries = ("pair.txt", "cams", "blended_images", "rendered_depth_maps")

    def inspect(self, dataset_root: str | Path) -> dict[str, Any]:
        root = Path(dataset_root)
        warnings: list[str] = []
        errors: list[str] = []

        if not root.exists():
            errors.append(f"dataset root does not exist: {root}")
            return self._report(root, [], warnings, errors)

        scenes = self._discover_scenes(root)
        if not scenes:
            errors.append("no BlendedMVS scenes found")

        scene_reports = []
        for scene in scenes:
            scene_warnings, scene_errors = self._validate_scene(scene)
            warnings.extend(f"{scene.scene_id}: {item}" for item in scene_warnings)
            errors.extend(f"{scene.scene_id}: {item}" for item in scene_errors)
            scene_reports.append(
                {
                    "scene_id": scene.scene_id,
                    "root": str(scene.root),
                    "image_count": scene.image_count,
                    "depth_count": scene.depth_count,
                    "camera_count": scene.camera_count,
                    "has_pair_file": scene.has_pair_file,
                }
            )

        return self._report(root, scene_reports, warnings, errors)

    def _discover_scenes(self, root: Path) -> list[BlendedMVSScene]:
        if self._looks_like_scene(root):
            return [self._scene_from_path(root)]

        scenes = [self._scene_from_path(path) for path in sorted(root.iterdir()) if path.is_dir() and self._looks_like_scene(path)]
        return scenes

    def _looks_like_scene(self, path: Path) -> bool:
        return any((path / entry).exists() for entry in self.required_entries)

    def _scene_from_path(self, path: Path) -> BlendedMVSScene:
        image_dir = path / "blended_images"
        depth_dir = path / "rendered_depth_maps"
        cams_dir = path / "cams"
        return BlendedMVSScene(
            scene_id=path.name,
            root=path,
            image_count=self._count_files(image_dir, {".jpg", ".jpeg", ".png"}),
            depth_count=self._count_files(depth_dir, {".pfm", ".npy", ".exr", ".png"}),
            camera_count=self._count_files(cams_dir, {".txt"}),
            has_pair_file=(path / "pair.txt").is_file(),
        )

    def _count_files(self, directory: Path, suffixes: set[str]) -> int:
        if not directory.is_dir():
            return 0
        return sum(1 for item in directory.iterdir() if item.is_file() and item.suffix.lower() in suffixes)

    def _validate_scene(self, scene: BlendedMVSScene) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        root = scene.root

        for entry in self.required_entries:
            if not (root / entry).exists():
                errors.append(f"missing required entry: {entry}")

        if scene.image_count == 0:
            errors.append("no RGB images found in blended_images")
        if scene.depth_count == 0:
            warnings.append("no depth maps found in rendered_depth_maps")
        if scene.camera_count == 0:
            errors.append("no camera files found in cams")
        if scene.image_count and scene.camera_count and scene.image_count != scene.camera_count:
            warnings.append(f"image/camera count mismatch: images={scene.image_count}, cameras={scene.camera_count}")
        if scene.depth_count and scene.image_count != scene.depth_count:
            warnings.append(f"image/depth count mismatch: images={scene.image_count}, depths={scene.depth_count}")

        return warnings, errors

    def _report(self, root: Path, scenes: list[dict[str, Any]], warnings: list[str], errors: list[str]) -> dict[str, Any]:
        return {
            "dataset": "blendedmvs",
            "input_root": str(root),
            "profile": "multiview_mvs",
            "status": "ok" if not errors else "error",
            "scene_count": len(scenes),
            "scenes": scenes,
            "warnings": warnings,
            "errors": errors,
            "conventions": {
                "pose": "world_to_camera",
                "camera_model": "pinhole",
                "depth_unit": "meter",
                "depth_encoding": "pfm_m",
            },
            "modalities": {
                "rgb": "native",
                "depth": "native",
                "camera": "native",
                "pose": "native",
                "pairs": "native",
            },
        }
