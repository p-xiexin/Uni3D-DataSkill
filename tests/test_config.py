import json
import tempfile
import unittest
from pathlib import Path

from unidata_skill.config import load_dataset_configs


class DatasetConfigTests(unittest.TestCase):
    def test_load_dataset_configs_accepts_kitti360_and_blendedmvs_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dataset_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"label": "kitti360_train", "dataset": "kitti360", "root": "E:/Datasets/KITTI-360"},
                            {"label": "blendedmvs_train", "dataset": "blendedmvs", "root": "E:/Datasets/BlendedMVG"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            configs = load_dataset_configs(config_path)

        self.assertEqual([config.label for config in configs], ["kitti360_train", "blendedmvs_train"])
        self.assertEqual([config.dataset for config in configs], ["kitti360", "blendedmvs"])

    def test_load_dataset_configs_validates_roots_and_allows_optional_null(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "data_2d_raw"
            images.mkdir()
            config_path = root / "dataset_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "label": "kitti360_train",
                                "dataset": "kitti360",
                                "root": str(root),
                                "roots": {"images": str(images)},
                                "optional_roots": {"lidar": None},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            configs = load_dataset_configs(config_path)

        self.assertEqual(configs[0].options["roots"]["images"], str(images))
        self.assertIsNone(configs[0].options["optional_roots"]["lidar"])

    def test_load_dataset_configs_rejects_missing_required_root_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing"
            config_path = root / "dataset_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "label": "kitti360_train",
                                "dataset": "kitti360",
                                "root": str(root),
                                "roots": {"images": str(missing)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, r"datasets\[0\]\.roots\.images does not exist"):
                load_dataset_configs(config_path)


if __name__ == "__main__":
    unittest.main()
