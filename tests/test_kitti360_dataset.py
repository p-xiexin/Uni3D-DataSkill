import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from unidata_skill.cli import main
from unidata_skill.datasets import Kitti360Pi3XDataset


SEQUENCE = "2013_05_28_drive_0000_sync"


def write_tiny_kitti360(root: Path, frame_count: int = 6) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True)
    calibration.joinpath("perspective.txt").write_text(
        "\n".join(
            [
                "# generated on 2013-05-28, should be ignored",
                "P_rect_00: 100.0 0.0 2.0 0.0 0.0 100.0 2.0 0.0 0.0 0.0 1.0 0.0",
                "P_rect_01: 100.0 0.0 2.0 0.0 0.0 100.0 2.0 0.0 0.0 0.0 1.0 0.0 # inline comment",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calibration.joinpath("calib_cam_to_pose.txt").write_text(
        "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    pose_dir = root / "data_poses" / SEQUENCE
    pose_dir.mkdir(parents=True)
    pose_lines = []
    for frame_id in range(frame_count):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(frame_id)
        pose_lines.append(f"{frame_id} " + " ".join(str(float(item)) for item in pose[:3, :].reshape(-1)))
    pose_lines.insert(0, "# 2013-05-28 generated poses, should be ignored")
    pose_dir.joinpath("cam0_to_world.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    for camera_id in ("image_00", "image_01"):
        image_dir = root / "data_2d_raw" / SEQUENCE / camera_id / "data_rect"
        image_dir.mkdir(parents=True)
        for frame_id in range(frame_count):
            image = Image.new("RGB", (8, 6), color=(frame_id * 10, 20, 30))
            image.save(image_dir / f"{frame_id:010d}.png")


class Kitti360Pi3XDatasetTests(unittest.TestCase):
    def test_dataset_reads_directly_from_raw_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti360(root)

            dataset = Kitti360Pi3XDataset(data_root=root, sequences=[SEQUENCE], frame_num=3, stride=1, resolution=(4, 4))

            self.assertEqual(len(dataset), 1)
            views = dataset[0]
            self.assertEqual(len(views), 3)
            self.assertEqual(views[0]["dataset"], "KITTI360Pi3X")
            self.assertEqual(views[0]["label"], SEQUENCE)
            self.assertEqual(views[0]["camera_intrinsics"].shape, (3, 3))
            self.assertEqual(views[0]["camera_pose"].shape, (4, 4))
            self.assertEqual(views[0]["depth_source"], "placeholder_missing_dense_depth")
            self.assertTrue(Path(views[0]["image_path"]).is_file())

    def test_dataset_accepts_explicit_component_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "official"
            dummy_root = Path(tmpdir) / "dummy"
            dummy_root.mkdir()
            write_tiny_kitti360(root)

            dataset = Kitti360Pi3XDataset(
                data_root=dummy_root,
                roots={
                    "calibration": str(root / "calibration"),
                    "images": str(root / "data_2d_raw"),
                    "poses": str(root / "data_poses"),
                },
                sequences=[SEQUENCE],
                frame_num=3,
                stride=1,
                resolution=(4, 4),
            )

            views = dataset[0]
            self.assertEqual(len(views), 3)
            self.assertTrue(str(root / "data_2d_raw") in views[0]["image_path"])

    def test_dataset_preserves_unavailable_optional_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti360(root)

            dataset = Kitti360Pi3XDataset(
                data_root=root,
                sequences=[SEQUENCE],
                frame_num=3,
                stride=1,
                resolution=(4, 4),
                optional_roots={"lidar": None},
            )

            self.assertIn("lidar", dataset.optional_roots)
            self.assertIsNone(dataset.optional_roots["lidar"])

    def test_cli_validate_config_selects_dataset_by_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti360(root)
            config_path = root / "dataset_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "label": "tiny_kitti360",
                                "dataset": "kitti360",
                                "root": str(root),
                                "sequences": [SEQUENCE],
                                "frame_num": 3,
                                "stride": 1,
                                "resolution": "4x4",
                                "max_samples": 1,
                                "roots": {
                                    "calibration": str(root / "calibration"),
                                    "images": str(root / "data_2d_raw"),
                                    "poses": str(root / "data_poses"),
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["validate-config", "--config", str(config_path), "--label", "tiny_kitti360"])

            report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["label"], "tiny_kitti360")
            self.assertEqual(report["dataset"], "kitti360")


if __name__ == "__main__":
    unittest.main()
