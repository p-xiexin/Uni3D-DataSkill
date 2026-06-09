import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from unidata_skill.cli import main
from unidata_skill.datasets import Kitti360Pi3XDataset, validate_kitti360_pi3x_dataloader


SEQUENCE = "2013_05_28_drive_0000_sync"


def write_tiny_kitti360(root: Path, frame_count: int = 6) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True)
    calibration.joinpath("perspective.txt").write_text(
        "\n".join(
            [
                "P_rect_00: 100.0 0.0 2.0 0.0 0.0 100.0 2.0 0.0 0.0 0.0 1.0 0.0",
                "P_rect_01: 100.0 0.0 2.0 0.0 0.0 100.0 2.0 0.0 0.0 0.0 1.0 0.0",
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

            dataset = Kitti360Pi3XDataset(root, sequences=[SEQUENCE], frame_num=3, stride=1, resolution=(4, 4))

            self.assertEqual(len(dataset), 1)
            views = dataset[0]
            self.assertEqual(len(views), 3)
            self.assertEqual(views[0]["dataset"], "KITTI360Pi3X")
            self.assertEqual(views[0]["label"], SEQUENCE)
            self.assertEqual(views[0]["camera_intrinsics"].shape, (3, 3))
            self.assertEqual(views[0]["camera_pose"].shape, (4, 4))
            self.assertEqual(views[0]["depth_source"], "placeholder_missing_dense_depth")
            self.assertTrue(Path(views[0]["image_path"]).is_file())

    def test_validator_reports_ok_for_tiny_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti360(root)

            result = validate_kitti360_pi3x_dataloader(
                root,
                sequences=[SEQUENCE],
                frame_num=3,
                stride=1,
                resolution=(4, 4),
                max_samples=2,
            )

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.dataset_len, 1)
            self.assertEqual(result.checked_samples, 1)
            self.assertEqual(result.errors, [])

    def test_cli_validate_kitti360_pi3x(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_tiny_kitti360(root)

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "validate-kitti360-pi3x",
                        "--kitti360-root",
                        str(root),
                        "--sequence",
                        SEQUENCE,
                        "--frame-num",
                        "3",
                        "--stride",
                        "1",
                        "--resolution",
                        "4x4",
                        "--max-samples",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
