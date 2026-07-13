from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NuScenesTables:
    root: Path
    table_root: Path
    scenes: dict[str, dict[str, Any]]
    samples: dict[str, dict[str, Any]]
    sample_data: dict[str, dict[str, Any]]
    calibrated: dict[str, dict[str, Any]]
    ego_poses: dict[str, dict[str, Any]]
    sensors: dict[str, dict[str, Any]]
    sample_data_by_sample_channel: dict[str, dict[str, str]]


@dataclass(frozen=True)
class CameraTarget:
    sample_index: int
    sample: dict[str, Any]
    camera_sample_data: dict[str, Any]
    camera_pose: np.ndarray
    intrinsics: np.ndarray
    image_shape: tuple[int, int]
    image_path: Path


@dataclass(frozen=True)
class ProjectionSample:
    target: CameraTarget
    points_cam: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    source_sweep_offset: np.ndarray
    depth_map: np.ndarray
    count_map: np.ndarray
    source_sweeps: list[dict[str, Any]]
    raw_accumulated_points: int


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    projection_npz: Path
    depth_npy: Path
    depth_png_uint16: Path
    count_png_uint16: Path
    depth_viz_png: Path
    count_viz_png: Path
    visualization: Path
    summary: Path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tables(root: Path, version: str) -> NuScenesTables:
    table_root = root / version
    if not table_root.is_dir():
        raise FileNotFoundError(f"nuScenes table directory not found: {table_root}")
    sample_data = {item["token"]: item for item in read_json(table_root / "sample_data.json")}
    calibrated = {item["token"]: item for item in read_json(table_root / "calibrated_sensor.json")}
    sensors = {item["token"]: item for item in read_json(table_root / "sensor.json")}
    sample_data_by_sample_channel: dict[str, dict[str, str]] = {}
    for item in sample_data.values():
        calib = calibrated[item["calibrated_sensor_token"]]
        sensor = sensors[calib["sensor_token"]]
        channel = sensor["channel"]
        channel_map = sample_data_by_sample_channel.setdefault(item["sample_token"], {})
        old_token = channel_map.get(channel)
        if old_token is None or (item.get("is_key_frame", False) and not sample_data[old_token].get("is_key_frame", False)):
            channel_map[channel] = item["token"]
    return NuScenesTables(
        root=root,
        table_root=table_root,
        scenes={item["name"]: item for item in read_json(table_root / "scene.json")},
        samples={item["token"]: item for item in read_json(table_root / "sample.json")},
        sample_data=sample_data,
        calibrated=calibrated,
        ego_poses={item["token"]: item for item in read_json(table_root / "ego_pose.json")},
        sensors=sensors,
        sample_data_by_sample_channel=sample_data_by_sample_channel,
    )


def quaternion_wxyz_to_rotation(q: list[float] | tuple[float, ...]) -> np.ndarray:
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
        dtype=np.float64,
    )


def transform_from_record(record: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_rotation(record["rotation"])
    transform[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points[:, :3], np.ones((len(points), 1), dtype=np.float64)), axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def resolve_data_path(root: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return root / path


def load_lidar_bin(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {path}")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"expected nuScenes LiDAR .bin with 5 floats per point: {path}")
    return raw.reshape(-1, 5)


def scene_samples_in_order(tables: NuScenesTables, scene_name: str) -> list[dict[str, Any]]:
    if scene_name not in tables.scenes:
        choices = ", ".join(sorted(tables.scenes)[:10])
        raise KeyError(f"scene not found: {scene_name}. First scenes: {choices}")
    scene = tables.scenes[scene_name]
    token = scene["first_sample_token"]
    samples = []
    seen = set()
    while token:
        if token in seen:
            raise RuntimeError(f"sample chain loop in scene {scene_name}: {token}")
        seen.add(token)
        sample = tables.samples[token]
        samples.append(sample)
        token = sample.get("next", "")
    return samples


def select_target_sample(samples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.target_sample_token:
        for index, sample in enumerate(samples):
            if sample["token"] == args.target_sample_token:
                return index, sample
        raise KeyError(f"target sample token not found in scene: {args.target_sample_token}")
    if args.target_index < 0 or args.target_index >= len(samples):
        raise IndexError(f"--target-index {args.target_index} outside scene sample range [0, {len(samples)})")
    return args.target_index, samples[args.target_index]


def walk_sample_data_chain(tables: NuScenesTables, start_token: str, direction: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    key = "prev" if direction == "prev" else "next"
    out = []
    token = tables.sample_data[start_token].get(key, "")
    seen = {start_token}
    while token and len(out) < limit:
        if token in seen:
            raise RuntimeError(f"sample_data {direction} chain loop at token {token}")
        seen.add(token)
        record = tables.sample_data[token]
        out.append(record)
        token = record.get(key, "")
    return out


def select_source_lidar_records(
    tables: NuScenesTables,
    target_sample: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, Any]]]:
    target_lidar = sample_data_for_channel(tables, target_sample, args.lidar)
    before = list(reversed(walk_sample_data_chain(tables, target_lidar["token"], "prev", args.sweeps_before)))
    after = walk_sample_data_chain(tables, target_lidar["token"], "next", args.sweeps_after)
    records = before + ([] if args.exclude_target_sweep else [target_lidar]) + after
    if not records:
        raise RuntimeError("no source LiDAR sweeps selected")
    target_timestamp = float(target_lidar.get("timestamp", target_sample.get("timestamp", 0)))
    source = []
    for record in records:
        timestamp = float(record.get("timestamp", target_timestamp))
        relative_index = int(round((timestamp - target_timestamp) / 1e5))
        source.append((relative_index, record))
    return source


def sample_data_pose_c2w(tables: NuScenesTables, sample_data_record: dict[str, Any]) -> np.ndarray:
    calibrated = tables.calibrated[sample_data_record["calibrated_sensor_token"]]
    ego_pose = tables.ego_poses[sample_data_record["ego_pose_token"]]
    sensor_to_ego = transform_from_record(calibrated)
    ego_to_global = transform_from_record(ego_pose)
    return ego_to_global @ sensor_to_ego


def sample_data_for_channel(tables: NuScenesTables, sample: dict[str, Any], channel: str) -> dict[str, Any]:
    channel_map = tables.sample_data_by_sample_channel.get(sample["token"], {})
    sample_data_token = channel_map.get(channel)
    if sample_data_token is None:
        available = ", ".join(sorted(channel_map)) or "none"
        raise KeyError(f"sample {sample['token']} has no channel {channel}. Available channels: {available}")
    return tables.sample_data[sample_data_token]


def camera_info(tables: NuScenesTables, sample: dict[str, Any], camera_channel: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, tuple[int, int]]:
    camera_sd = sample_data_for_channel(tables, sample, camera_channel)
    camera_pose = sample_data_pose_c2w(tables, camera_sd)
    calibrated = tables.calibrated[camera_sd["calibrated_sensor_token"]]
    intrinsics = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
    image_path = resolve_data_path(tables.root, camera_sd["filename"])
    with Image.open(image_path) as image:
        width, height = image.size
    return camera_sd, camera_pose, intrinsics, (height, width)


def build_camera_target(tables: NuScenesTables, samples: list[dict[str, Any]], args: argparse.Namespace) -> CameraTarget:
    target_index, target_sample = select_target_sample(samples, args)
    camera_sd, camera_pose, intrinsics, image_shape = camera_info(tables, target_sample, args.camera)
    image_path = resolve_data_path(tables.root, camera_sd["filename"])
    return CameraTarget(
        sample_index=target_index,
        sample=target_sample,
        camera_sample_data=camera_sd,
        camera_pose=camera_pose,
        intrinsics=intrinsics,
        image_shape=image_shape,
        image_path=image_path,
    )


def accumulate_lidar_in_target_camera(
    tables: NuScenesTables,
    source_lidar_records: list[tuple[int, dict[str, Any]]],
    target_camera_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_camera_from_world = np.linalg.inv(target_camera_pose)
    all_points_cam = []
    all_source_index = []
    source_meta = []
    for source_index, lidar_sd in source_lidar_records:
        lidar_path = resolve_data_path(tables.root, lidar_sd["filename"])
        lidar_points = load_lidar_bin(lidar_path)
        lidar_pose = sample_data_pose_c2w(tables, lidar_sd)
        points_world = transform_points(lidar_pose, lidar_points)
        points_cam = transform_points(target_camera_from_world, points_world)
        all_points_cam.append(points_cam.astype(np.float32))
        all_source_index.append(np.full(len(points_cam), source_index, dtype=np.int32))
        source_meta.append(
            {
                "sweep_offset": int(source_index),
                "sample_token": lidar_sd["sample_token"],
                "lidar_sample_data_token": lidar_sd["token"],
                "lidar_path": str(lidar_path),
                "timestamp": int(lidar_sd.get("timestamp", 0)),
                "is_key_frame": bool(lidar_sd.get("is_key_frame", False)),
                "points": int(len(points_cam)),
            }
        )
    if not all_points_cam:
        raise RuntimeError("no LiDAR points loaded")
    return np.concatenate(all_points_cam, axis=0), np.concatenate(all_source_index, axis=0), source_meta


def build_projection_sample(tables: NuScenesTables, target: CameraTarget, args: argparse.Namespace) -> ProjectionSample:
    source_lidar_records = select_source_lidar_records(tables, target.sample, args)
    points_cam, source_indices, source_sweeps = accumulate_lidar_in_target_camera(
        tables,
        source_lidar_records,
        target.camera_pose,
    )
    visible_points_cam, uv, depth, visible_source_indices = visible_projected_points(
        points_cam,
        source_indices,
        target.intrinsics,
        target.image_shape,
        args.min_depth,
        args.max_depth,
    )
    depth_map, count_map = rasterize_depth(uv, depth, target.image_shape, args.depth_conflict)
    return ProjectionSample(
        target=target,
        points_cam=visible_points_cam,
        uv=uv,
        depth=depth,
        source_sweep_offset=visible_source_indices,
        depth_map=depth_map,
        count_map=count_map,
        source_sweeps=source_sweeps,
        raw_accumulated_points=int(len(points_cam)),
    )


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = points_cam[:, 2].astype(np.float64)
    uvw = (intrinsics @ points_cam[:, :3].astype(np.float64).T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-12)
    return uv.astype(np.float32), depth.astype(np.float32)


def filter_occlusion(
    points_cam: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Placeholder for future z-buffer, mesh, or ray-consistency occlusion filtering."""
    del points_cam, uv, image_shape
    return np.ones(len(depth), dtype=bool)


def visible_projected_points(
    points_cam: np.ndarray,
    source_index: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv, depth = project_points(points_cam, intrinsics)
    height, width = image_shape
    inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > min_depth)
        & (depth <= max_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    occlusion_keep = filter_occlusion(points_cam[inside], uv[inside], depth[inside], image_shape)
    keep_indices = np.flatnonzero(inside)[occlusion_keep]
    return points_cam[keep_indices], uv[keep_indices], depth[keep_indices], source_index[keep_indices]


def rasterize_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple[int, int],
    conflict: str,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    depth_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.uint16)
    xy = np.rint(uv).astype(np.int64)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height) & np.isfinite(depth)
    for x, y, z in zip(xy[valid, 0], xy[valid, 1], depth[valid], strict=False):
        count_map[y, x] = min(int(count_map[y, x]) + 1, np.iinfo(np.uint16).max)
        old = depth_map[y, x]
        if old == 0:
            depth_map[y, x] = z
        elif conflict == "nearest":
            depth_map[y, x] = min(old, z)
        elif conflict == "farthest":
            depth_map[y, x] = max(old, z)
        elif conflict == "overwrite":
            depth_map[y, x] = z
        else:
            raise ValueError(f"unsupported depth conflict policy: {conflict}")
    return depth_map, count_map


def save_depth_png(path: Path, depth_map: np.ndarray, scale: float) -> None:
    encoded = np.clip(depth_map * scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(encoded).save(path)


def normalize_nonzero_to_uint8(array: np.ndarray, use_log: bool = False) -> np.ndarray:
    values = array.astype(np.float32)
    if use_log:
        values = np.log1p(values)
    mask = np.isfinite(values) & (values > 0)
    output = np.zeros(values.shape, dtype=np.uint8)
    if not mask.any():
        return output
    valid = values[mask]
    low = float(np.percentile(valid, 1.0))
    high = float(np.percentile(valid, 99.0))
    if high <= low:
        high = float(valid.max())
        low = float(valid.min())
    if high <= low:
        output[mask] = 255
        return output
    scaled = (values[mask] - low) / (high - low)
    output[mask] = np.clip(scaled * 255.0, 1, 255).astype(np.uint8)
    return output


def save_depth_viz(path: Path, depth_map: np.ndarray) -> None:
    Image.fromarray(normalize_nonzero_to_uint8(depth_map)).save(path)


def save_count_viz(path: Path, count_map: np.ndarray) -> None:
    Image.fromarray(normalize_nonzero_to_uint8(count_map, use_log=True)).save(path)


def visualize_projection(image_path: Path, uv: np.ndarray, depth: np.ndarray, output_path: Path, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(Image.open(image_path).convert("RGB"))
    if len(uv) > max_points:
        pick = np.linspace(0, len(uv) - 1, max_points).astype(np.int64)
        uv = uv[pick]
        depth = depth[pick]
    plt.figure("nuscenes_lidar_accumulation", figsize=(10, 5))
    plt.imshow(image)
    if len(uv):
        plt.scatter(uv[:, 0], uv[:, 1], s=0.4, c=depth, cmap="turbo")
        plt.colorbar(label="depth_m", fraction=0.025, pad=0.01)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=160)
    plt.close("all")


def prepare_output_paths(base_dir: Path, scene: str, camera: str, target_index: int) -> OutputPaths:
    scene_part = scene.replace("/", "_")
    output_dir = base_dir / scene_part / camera / f"{target_index:06d}"
    return OutputPaths(
        output_dir=output_dir,
        projection_npz=output_dir / "accumulated_lidar_projection.npz",
        depth_npy=output_dir / "semi_dense_depth.npy",
        depth_png_uint16=output_dir / "semi_dense_depth_uint16.png",
        count_png_uint16=output_dir / "projection_count_uint16.png",
        depth_viz_png=output_dir / "semi_dense_depth_viz.png",
        count_viz_png=output_dir / "projection_count_viz.png",
        visualization=output_dir / "projection_viz.jpg",
        summary=output_dir / "summary.json",
    )


def save_projection_sample(sample: ProjectionSample, paths: OutputPaths, args: argparse.Namespace) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.projection_npz,
        points_cam=sample.points_cam.astype(np.float32),
        uv=sample.uv.astype(np.float32),
        depth=sample.depth.astype(np.float32),
        source_sweep_offset=sample.source_sweep_offset.astype(np.int32),
        intrinsics=sample.target.intrinsics.astype(np.float32),
        target_camera_pose=sample.target.camera_pose.astype(np.float32),
        image_shape=np.asarray(sample.target.image_shape, dtype=np.int32),
    )
    np.save(paths.depth_npy, sample.depth_map)
    save_depth_png(paths.depth_png_uint16, sample.depth_map, args.depth_png_scale)
    Image.fromarray(sample.count_map).save(paths.count_png_uint16)
    save_depth_viz(paths.depth_viz_png, sample.depth_map)
    save_count_viz(paths.count_viz_png, sample.count_map)
    if not args.no_viz:
        visualize_projection(sample.target.image_path, sample.uv, sample.depth, paths.visualization, args.max_viz_points)


def projection_summary(tables: NuScenesTables, sample: ProjectionSample, paths: OutputPaths, args: argparse.Namespace) -> dict[str, Any]:
    height, width = sample.target.image_shape
    return {
        "root": str(tables.root),
        "version": args.version,
        "scene": args.scene,
        "camera": args.camera,
        "lidar": args.lidar,
        "accumulation_mode": "sweeps",
        "sweeps_before": int(args.sweeps_before),
        "sweeps_after": int(args.sweeps_after),
        "include_target_sweep": not bool(args.exclude_target_sweep),
        "exclude_target_sweep": bool(args.exclude_target_sweep),
        "target_index": int(sample.target.sample_index),
        "target_sample_token": sample.target.sample["token"],
        "target_camera_sample_data_token": sample.target.camera_sample_data["token"],
        "target_image": str(sample.target.image_path),
        "image_shape": [int(height), int(width)],
        "source_sweeps": sample.source_sweeps,
        "raw_accumulated_points": int(sample.raw_accumulated_points),
        "visible_projected_points": int(len(sample.depth)),
        "filled_depth_pixels": int(np.count_nonzero(sample.depth_map)),
        "depth_conflict": args.depth_conflict,
        "occlusion_filter": "placeholder_keep_all",
        "outputs": {
            "projection_npz": str(paths.projection_npz),
            "depth_npy": str(paths.depth_npy),
            "depth_png_uint16": str(paths.depth_png_uint16),
            "count_png_uint16": str(paths.count_png_uint16),
            "depth_viz_png": str(paths.depth_viz_png),
            "count_viz_png": str(paths.count_viz_png),
            "visualization": None if args.no_viz else str(paths.visualization),
        },
    }


def write_summary(paths: OutputPaths, summary: dict[str, Any]) -> None:
    paths.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accumulate nuScenes LiDAR sweeps with GT poses and project to one camera frame.")
    parser.add_argument("--root", type=Path, required=True, help="nuScenes dataset root containing samples/, sweeps/, and version tables.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", required=True, help="nuScenes scene name, e.g. scene-0061.")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--lidar", default="LIDAR_TOP")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-sample-token", default=None)
    parser.add_argument("--sweeps-before", type=int, default=20)
    parser.add_argument("--sweeps-after", type=int, default=0)
    parser.add_argument("--exclude-target-sweep", action="store_true")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--depth-conflict", choices=["nearest", "farthest", "overwrite"], default="nearest")
    parser.add_argument("--depth-png-scale", type=float, default=256.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nuscenes_lidar_accumulation_demo"))
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--max-viz-points", type=int, default=200000)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sweeps_before < 0 or args.sweeps_after < 0:
        raise ValueError("--sweeps-before and --sweeps-after must be non-negative")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.depth_png_scale <= 0:
        raise ValueError("--depth-png-scale must be positive")
    if args.max_viz_points <= 0:
        raise ValueError("--max-viz-points must be positive")
    if args.exclude_target_sweep and args.sweeps_before == 0 and args.sweeps_after == 0:
        raise ValueError("excluding the target sweep requires --sweeps-before or --sweeps-after to be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    tables = load_tables(args.root, args.version)
    samples = scene_samples_in_order(tables, args.scene)
    target = build_camera_target(tables, samples, args)
    sample = build_projection_sample(tables, target, args)
    paths = prepare_output_paths(args.output_dir, args.scene, args.camera, target.sample_index)
    save_projection_sample(sample, paths, args)
    summary = projection_summary(tables, sample, paths, args)
    write_summary(paths, summary)
    print(f"target image: {target.image_path}")
    print(f"raw accumulated points: {summary['raw_accumulated_points']}")
    print(f"visible projected points: {summary['visible_projected_points']}")
    print(f"filled depth pixels: {summary['filled_depth_pixels']}")
    print(f"summary: {paths.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
