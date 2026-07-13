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
        sample_data_by_sample_channel.setdefault(item["sample_token"], {})[channel] = item["token"]
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


def select_source_samples(samples: list[dict[str, Any]], target_index: int, args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    start = max(0, target_index - args.frames_before)
    end = min(len(samples), target_index + args.frames_after + 1)
    selected = []
    for index in range(start, end):
        if index == target_index and not args.include_target_lidar:
            continue
        selected.append((index, samples[index]))
    if not selected:
        raise RuntimeError("no source LiDAR frames selected")
    return selected


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


def accumulate_lidar_in_target_camera(
    tables: NuScenesTables,
    source_samples: list[tuple[int, dict[str, Any]]],
    target_camera_pose: np.ndarray,
    lidar_channel: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_camera_from_world = np.linalg.inv(target_camera_pose)
    all_points_cam = []
    all_source_index = []
    source_meta = []
    for source_index, sample in source_samples:
        try:
            lidar_sd = sample_data_for_channel(tables, sample, lidar_channel)
        except KeyError:
            continue
        lidar_path = resolve_data_path(tables.root, lidar_sd["filename"])
        lidar_points = load_lidar_bin(lidar_path)
        lidar_pose = sample_data_pose_c2w(tables, lidar_sd)
        points_world = transform_points(lidar_pose, lidar_points)
        points_cam = transform_points(target_camera_from_world, points_world)
        all_points_cam.append(points_cam.astype(np.float32))
        all_source_index.append(np.full(len(points_cam), source_index, dtype=np.int32))
        source_meta.append(
            {
                "sample_index": int(source_index),
                "sample_token": sample["token"],
                "lidar_sample_data_token": lidar_sd["token"],
                "lidar_path": str(lidar_path),
                "points": int(len(points_cam)),
            }
        )
    if not all_points_cam:
        raise RuntimeError(f"no LiDAR points loaded for channel {lidar_channel}")
    return np.concatenate(all_points_cam, axis=0), np.concatenate(all_source_index, axis=0), source_meta


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


def compare_with_devkit(tables: NuScenesTables, target_sample: dict[str, Any], args: argparse.Namespace, target_camera_shape: tuple[int, int]) -> dict[str, Any]:
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
    except ModuleNotFoundError as exc:
        return {"available": False, "error": str(exc)}

    nusc = NuScenes(version=args.version, dataroot=str(tables.root), verbose=False)
    devkit_target_sample = nusc.get("sample", target_sample["token"])
    devkit_points, devkit_times = LidarPointCloud.from_file_multisweep(
        nusc,
        devkit_target_sample,
        chan=args.lidar,
        ref_chan=args.camera,
        nsweeps=args.devkit_nsweeps,
        min_distance=args.devkit_min_distance,
    )
    target_camera_sd, _pose, intrinsics, _shape = camera_info(tables, target_sample, args.camera)
    uv, depth = project_points(devkit_points.points[:3].T.astype(np.float32), intrinsics)
    height, width = target_camera_shape
    inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > args.min_depth)
        & (depth <= args.max_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return {
        "available": True,
        "ref_camera_sample_data_token": target_camera_sd["token"],
        "nsweeps": int(args.devkit_nsweeps),
        "raw_points": int(devkit_points.nbr_points()),
        "projected_visible_points": int(inside.sum()),
        "time_lag_min": float(np.min(devkit_times)) if devkit_times.size else None,
        "time_lag_max": float(np.max(devkit_times)) if devkit_times.size else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accumulate nuScenes LiDAR keyframes with GT poses and project to one camera frame.")
    parser.add_argument("--root", type=Path, required=True, help="nuScenes dataset root containing samples/, sweeps/, and version tables.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", required=True, help="nuScenes scene name, e.g. scene-0061.")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--lidar", default="LIDAR_TOP")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-sample-token", default=None)
    parser.add_argument("--frames-before", type=int, default=5)
    parser.add_argument("--frames-after", type=int, default=0)
    parser.add_argument("--include-target-lidar", action="store_true")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--depth-conflict", choices=["nearest", "farthest", "overwrite"], default="nearest")
    parser.add_argument("--depth-png-scale", type=float, default=256.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nuscenes_lidar_accumulation_demo"))
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--max-viz-points", type=int, default=200000)
    parser.add_argument("--compare-devkit", action="store_true")
    parser.add_argument("--devkit-nsweeps", type=int, default=6)
    parser.add_argument("--devkit-min-distance", type=float, default=1.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.frames_before < 0 or args.frames_after < 0:
        raise ValueError("--frames-before and --frames-after must be non-negative")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.depth_png_scale <= 0:
        raise ValueError("--depth-png-scale must be positive")
    if args.max_viz_points <= 0:
        raise ValueError("--max-viz-points must be positive")
    if args.devkit_nsweeps <= 0:
        raise ValueError("--devkit-nsweeps must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    tables = load_tables(args.root, args.version)
    samples = scene_samples_in_order(tables, args.scene)
    target_index, target_sample = select_target_sample(samples, args)
    source_samples = select_source_samples(samples, target_index, args)
    target_camera_sd, target_camera_pose, intrinsics, image_shape = camera_info(tables, target_sample, args.camera)
    target_image_path = resolve_data_path(tables.root, target_camera_sd["filename"])

    points_cam, source_indices, source_meta = accumulate_lidar_in_target_camera(
        tables,
        source_samples,
        target_camera_pose,
        args.lidar,
    )
    visible_points_cam, uv, depth, visible_source_indices = visible_projected_points(
        points_cam,
        source_indices,
        intrinsics,
        image_shape,
        args.min_depth,
        args.max_depth,
    )
    depth_map, count_map = rasterize_depth(uv, depth, image_shape, args.depth_conflict)

    scene_part = args.scene.replace("/", "_")
    output_dir = args.output_dir / scene_part / args.camera / f"{target_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "accumulated_lidar_projection.npz"
    depth_npy_path = output_dir / "semi_dense_depth.npy"
    depth_png_path = output_dir / "semi_dense_depth_uint16.png"
    count_png_path = output_dir / "projection_count_uint16.png"
    depth_viz_path = output_dir / "semi_dense_depth_viz.png"
    count_viz_path = output_dir / "projection_count_viz.png"
    viz_path = output_dir / "projection_viz.jpg"
    summary_path = output_dir / "summary.json"

    np.savez_compressed(
        npz_path,
        points_cam=visible_points_cam.astype(np.float32),
        uv=uv.astype(np.float32),
        depth=depth.astype(np.float32),
        source_sample_index=visible_source_indices.astype(np.int32),
        intrinsics=intrinsics.astype(np.float32),
        target_camera_pose=target_camera_pose.astype(np.float32),
        image_shape=np.asarray(image_shape, dtype=np.int32),
    )
    np.save(depth_npy_path, depth_map)
    save_depth_png(depth_png_path, depth_map, args.depth_png_scale)
    Image.fromarray(count_map).save(count_png_path)
    save_depth_viz(depth_viz_path, depth_map)
    save_count_viz(count_viz_path, count_map)
    if not args.no_viz:
        visualize_projection(target_image_path, uv, depth, viz_path, args.max_viz_points)

    devkit_summary = compare_with_devkit(tables, target_sample, args, image_shape) if args.compare_devkit else None
    summary = {
        "root": str(args.root),
        "version": args.version,
        "scene": args.scene,
        "camera": args.camera,
        "lidar": args.lidar,
        "target_index": int(target_index),
        "target_sample_token": target_sample["token"],
        "target_camera_sample_data_token": target_camera_sd["token"],
        "target_image": str(target_image_path),
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "source_frames": source_meta,
        "raw_accumulated_points": int(len(points_cam)),
        "visible_projected_points": int(len(depth)),
        "filled_depth_pixels": int(np.count_nonzero(depth_map)),
        "depth_conflict": args.depth_conflict,
        "occlusion_filter": "placeholder_keep_all",
        "outputs": {
            "projection_npz": str(npz_path),
            "depth_npy": str(depth_npy_path),
            "depth_png_uint16": str(depth_png_path),
            "count_png_uint16": str(count_png_path),
            "depth_viz_png": str(depth_viz_path),
            "count_viz_png": str(count_viz_path),
            "visualization": None if args.no_viz else str(viz_path),
        },
        "devkit_compare": devkit_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"target image: {target_image_path}")
    print(f"raw accumulated points: {summary['raw_accumulated_points']}")
    print(f"visible projected points: {summary['visible_projected_points']}")
    print(f"filled depth pixels: {summary['filled_depth_pixels']}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
