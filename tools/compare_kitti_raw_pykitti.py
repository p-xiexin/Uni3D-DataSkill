from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from unidata_skill.datasets.kitti_raw_dataset import generate_kitti_raw_index


def _load_pykitti() -> Any:
    import pykitti

    return pykitti


def _date_drive(sequence: str) -> tuple[str, str]:
    match = re.fullmatch(r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync", sequence)
    if match is None:
        raise ValueError(sequence)
    return match.group(1), match.group(2)


def _cam_idx(camera: str) -> int:
    return int(camera.split("_")[-1])


def _camera_from_imu(calib: Any, camera: str) -> np.ndarray:
    idx = _cam_idx(camera)

    direct_name = f"T_cam{idx}_imu"
    if hasattr(calib, direct_name):
        return np.asarray(getattr(calib, direct_name), dtype=np.float64)

    velo_name = f"T_cam{idx}_velo"
    if hasattr(calib, velo_name) and hasattr(calib, "T_velo_imu"):
        return np.asarray(getattr(calib, velo_name), dtype=np.float64) @ np.asarray(calib.T_velo_imu, dtype=np.float64)

    raise RuntimeError("missing calibration")


def _load_pykitti_raw(raw_root: Path, sequence: str) -> Any:
    pykitti = _load_pykitti()
    date, drive = _date_drive(sequence)
    return pykitti.raw(str(raw_root), date, drive)


def _world_from_camera_pykitti(data: Any, camera: str, frame_no: int) -> np.ndarray:
    world_from_imu = np.asarray(data.oxts[frame_no].T_w_imu, dtype=np.float64)
    imu_from_camera = np.linalg.inv(_camera_from_imu(data.calib, camera))
    return world_from_imu @ imu_from_camera


def rigid_transform(source_poses: list[np.ndarray], target_poses: list[np.ndarray]) -> np.ndarray:
    source = np.stack([pose[:3, 3] for pose in source_poses])
    target = np.stack([pose[:3, 3] for pose in target_poses])

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)

    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    translation = target_mean - rotation @ source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def translation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def rotation_error_degrees(a: np.ndarray, b: np.ndarray) -> float:
    rotation_delta = a[:3, :3] @ b[:3, :3].T
    cos_angle = (float(np.trace(rotation_delta)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def select_frames(index: dict[str, Any], sequence: str, max_frames: int) -> list[dict[str, Any]]:
    record = next(item for item in index["sequences"] if item["sequence_id"] == sequence)
    return record["frames"][:max_frames]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare aligned UniData KITTI raw camera poses against pykitti.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--camera", default="image_02")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-frames", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    index = generate_kitti_raw_index(
        args.raw_root,
        sequences=[args.sequence],
        cameras=(args.camera,),
        splits=(args.split,),
        roots={
            "raw": args.raw_root,
            "depth": args.depth_root,
            "calibration": args.raw_root,
        },
    )
    frames = select_frames(index, args.sequence, args.max_frames)
    pykitti_data = _load_pykitti_raw(args.raw_root, args.sequence)

    ours_list = []
    reference_list = []
    for frame in frames:
        frame_no = int(frame["frame_id"])
        ours_list.append(np.asarray(frame["camera_pose"], dtype=np.float64))
        reference_list.append(_world_from_camera_pykitti(pykitti_data, args.camera, frame_no))

    alignment = rigid_transform(ours_list, reference_list)

    max_translation_error = 0.0
    max_rotation_error = 0.0
    for idx, frame in enumerate(frames):
        ours = alignment @ ours_list[idx]
        reference = reference_list[idx]
        t_err = translation_error(ours, reference)
        r_err = rotation_error_degrees(ours, reference)
        max_translation_error = max(max_translation_error, t_err)
        max_rotation_error = max(max_rotation_error, r_err)
        print(f"{frame['frame_id']}: t={t_err:.6f} r={r_err:.6f}")

    print(f"\nSUMMARY: max_t={max_translation_error:.6f}, max_r={max_rotation_error:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
