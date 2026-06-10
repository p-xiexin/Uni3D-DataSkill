import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from pi3_test_utils import install_fake_pi3

install_fake_pi3()

from unidata_skill.cli import main
from unidata_skill.datasets import KittiOdometryPi3XDataset, NuScenesPi3XDataset, WaymoKittiPi3XDataset, WayveScenesPi3XDataset


def _write_image(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def write_tiny_kitti_odometry(root: Path, sequence: str = "00", frame_count: int = 5) -> None:
    seq_dir = root / "sequences" / sequence
    seq_dir.mkdir(parents=True)
    seq_dir.joinpath("calib.txt").write_text(
        "\n".join(
            [
                "P0: 100 0 2 0 0 100 2 0 0 0 1 0",
                "P1: 100 0 2 0 0 100 2 0 0 0 1 0",
                "P2: 100 0 2 0 0 100 2 0 0 0 1 0",
                "P3: 100 0 2 0 0 100 2 0 0 0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pose_dir = root / "poses"
    pose_dir.mkdir()
    pose_lines = []
    for idx in range(frame_count):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(idx)
        pose_lines.append(" ".join(str(float(item)) for item in pose[:3, :].reshape(-1)))
        _write_image(seq_dir / "image_2" / f"{idx:06d}.png", color=(idx * 20, 40, 60))
    pose_dir.joinpath(f"{sequence}.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")


def write_tiny_nuscenes(root: Path, frame_count: int = 4) -> None:
    table_dir = root / "v1.0-mini"
    table_dir.mkdir(parents=True)
    samples = []
    sample_data = []
    for idx in range(frame_count):
        token = f"sample-{idx}"
        samples.append({"token": token, "scene_token": "scene-token", "timestamp": idx})
        sample_data.append(
            {
                "token": f"sample-data-{idx}",
                "sample_token": token,
                "ego_pose_token": f"ego-{idx}",
                "calibrated_sensor_token": "calib-cam-front",
                "filename": f"samples/CAM_FRONT/{idx:06d}.jpg",
                "timestamp": idx,
            }
        )
        _write_image(root / "samples" / "CAM_FRONT" / f"{idx:06d}.jpg", color=(idx * 20, 40, 80))

    tables = {
        "scene.json": [{"token": "scene-token", "name": "scene-0001"}],
        "sample.json": samples,
        "sample_data.json": sample_data,
        "calibrated_sensor.json": [
            {
                "token": "calib-cam-front",
                "sensor_token": "sensor-cam-front",
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
                "camera_intrinsic": [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
            }
        ],
        "ego_pose.json": [
            {"token": f"ego-{idx}", "translation": [idx, 0, 0], "rotation": [1, 0, 0, 0]}
            for idx in range(frame_count)
        ],
        "sensor.json": [{"token": "sensor-cam-front", "modality": "camera", "channel": "CAM_FRONT"}],
    }
    for name, payload in tables.items():
        table_dir.joinpath(name).write_text(json.dumps(payload), encoding="utf-8")


def write_tiny_wayve(root: Path, scene: str = "scene_000", frame_count: int = 4) -> None:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True)
    frames = []
    for idx in range(frame_count):
        image_name = f"images/{idx:06d}.jpg"
        _write_image(scene_dir / image_name, color=(idx * 20, 50, 90))
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(idx)
        frames.append({"file_path": image_name, "transform_matrix": pose.tolist(), "frame_id": idx, "camera": "front"})
    scene_dir.joinpath("transforms.json").write_text(
        json.dumps({"fl_x": 100, "fl_y": 100, "cx": 2, "cy": 2, "w": 8, "h": 6, "frames": frames}),
        encoding="utf-8",
    )


class PriorityGeometryDatasetTests(unittest.TestCase):
    def test_kitti_odometry_reads_pi3x_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti_odometry(root)

            dataset = KittiOdometryPi3XDataset(root, sequences=["00"], frame_num=3, resolution=(4, 4))
            views = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(views), 3)
            self.assertEqual(views[0]["dataset"], "KITTIOdometryPi3X")
            self.assertEqual(views[0]["camera_intrinsics"].shape, (3, 3))
            self.assertEqual(views[0]["camera_pose"].shape, (4, 4))
            self.assertEqual(views[0]["depth_source"], "placeholder_missing_dense_depth")

    def test_nuscenes_tables_read_pi3x_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_nuscenes(root)

            dataset = NuScenesPi3XDataset(root, version="v1.0-mini", frame_num=2, resolution=(4, 4))
            views = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(views), 2)
            self.assertEqual(views[0]["dataset"], "NuScenesPi3X")
            self.assertEqual(views[0]["label"], "scene-0001")
            self.assertIn("sample_data_token", views[0])

    def test_wayve_transforms_read_pi3x_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_wayve(root)

            dataset = WayveScenesPi3XDataset(root, scene_dirs=["scene_000"], frame_num=2, resolution=(4, 4))
            views = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(views), 2)
            self.assertEqual(views[0]["dataset"], "WayveScenesPi3X")
            self.assertEqual(views[0]["label"], "scene_000")

    def test_waymo_kitti_style_reads_pi3x_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti_odometry(root)

            dataset = WaymoKittiPi3XDataset(root, sequences=["00"], frame_num=2, resolution=(4, 4))
            views = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(views), 2)
            self.assertEqual(views[0]["dataset"], "WaymoKittiPi3X")
            self.assertEqual(views[0]["depth_source"], "placeholder_missing_dense_depth")

    def test_cli_validate_config_supports_kitti(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti_odometry(root)
            config_path = root / "dataset_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "label": "tiny_kitti",
                                "dataset": "kitti",
                                "root": str(root),
                                "sequences": ["00"],
                                "frame_num": 3,
                                "resolution": "4x4",
                                "max_samples": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["validate-config", "--config", str(config_path), "--label", "tiny_kitti"])

            report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["dataset"], "kitti")


if __name__ == "__main__":
    unittest.main()
