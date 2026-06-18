from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from unidata_skill.datasets.kitti_raw_dataset import generate_kitti_raw_index


def _load_pykitti() -> Any:
    try:
        import pykitti
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install pykitti in the active environment to run this comparison script.") from exc
    return pykitti


def _date_drive_from_sequence(sequence: str) -> tuple[str, str]:
    match = re.fullmatch(r"(\d{4}_\d{2}_\d{2})_drive_(\d{4})_sync", sequence)
    if match is None:
        raise ValueError(f"Expected KITTI raw sequence like 2011_09_26_drive_0001_sync, got {sequence!r}")
    return match.group(1), match.group(2)


def _camera_index(camera: str) -> int:
    return int(camera.split("_")[-1])


def _pykitti_camera_from_imu(calib: Any, camera: str) -> np.ndarray:
    camera_idx = _camera_index(camera)
    direct_name = f"T_cam{camera_idx}_imu"
    if hasattr(calib, direct_name):
        return np.asarray(getattr(calib, direct_name), dtype=np.float64)

    velo_name = f"T_cam{camera_idx}_velo"
    if hasattr(calib, velo_name) and hasattr(calib, "T_velo_imu"):
        return np.asarray(getattr(calib, velo_name), dtype=np.float64) @ np.asarray(calib.T_velo_imu, dtype=np.float64)

    available = ", ".join(sorted(name for name in dir(calib) if name.startswith("T_cam") or name == "T_velo_imu"))
    raise AttributeError(f"Could not find pykitti transform for {camera}; available calibration transforms: {available}")


def _pykitti_world_camera_pose(raw_root: Path, sequence: str, camera: str, frame_no: int) -> np.ndarray:
    pykitti = _load_pykitti()
    date, drive = _date_drive_from_sequence(sequence)
    data = pykitti.raw(str(raw_root), date, drive, frames=[frame_no])
    world_from_imu = np.asarray(data.oxts[0].T_w_imu, dtype=np.float64)
    camera_from_imu = _pykitti_camera_from_imu(data.calib, camera)
    return world_from_imu @ np.linalg.inv(camera_from_imu)


def _translation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _rotation_error_degrees(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3] @ b[:3, :3].T
    cos_angle = (float(np.trace(delta)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _select_frames(index: dict[str, Any], sequence: str, frame_id: str | None, max_frames: int) -> list[dict[str, Any]]:
    record = next((item for item in index.get("sequences", []) if item.get("sequence_id") == sequence), None)
    if record is None:
        raise RuntimeError(f"Sequence was not indexed: {sequence}")
    frames = list(record.get("frames", []))
    if frame_id is not None:
        frames = [frame for frame in frames if frame.get("frame_id") == frame_id]
    if not frames:
        raise RuntimeError("No indexed frames matched the requested filters.")
    return frames[:max_frames]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare UniData KITTI raw camera poses against pykitti.")
    parser.add_argument("--raw-root", required=True, type=Path, help="KITTI raw root containing date folders.")
    parser.add_argument("--depth-root", required=True, type=Path, help="KITTI data_depth_annotated root.")
    parser.add_argument("--sequence", required=True, help="Sequence name, for example 2011_09_26_drive_0001_sync.")
    parser.add_argument("--camera", default="image_02", help="KITTI camera folder, default: image_02.")
    parser.add_argument("--split", default="train", help="Depth split under depth root, default: train.")
    parser.add_argument("--frame-id", help="Optional zero-padded frame id, for example 0000000005.")
    parser.add_argument("--max-frames", default=10, type=int, help="Maximum indexed frames to compare.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = generate_kitti_raw_index(
        args.raw_root,
        sequences=[args.sequence],
        cameras=(args.camera,),
        splits=(args.split,),
        roots={"raw": args.raw_root, "depth": args.depth_root, "calibration": args.raw_root},
    )
    frames = _select_frames(index, args.sequence, args.frame_id, args.max_frames)

    max_t = 0.0
    max_r = 0.0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        frame_no = int(frame_id)
        ours = np.asarray(frame["camera_pose"], dtype=np.float64)
        reference = _pykitti_world_camera_pose(args.raw_root, args.sequence, args.camera, frame_no)
        t_err = _translation_error(ours, reference)
        r_err = _rotation_error_degrees(ours, reference)
        max_t = max(max_t, t_err)
        max_r = max(max_r, r_err)
        print(f"{frame_id}: translation_error_m={t_err:.9f} rotation_error_deg={r_err:.9f}")

    print(f"summary: frames={len(frames)} max_translation_error_m={max_t:.9f} max_rotation_error_deg={max_r:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
