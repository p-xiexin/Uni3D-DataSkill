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


if __name__ == "__main__":
    unittest.main()
