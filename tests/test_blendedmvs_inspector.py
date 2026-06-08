import json
import tempfile
import unittest
from pathlib import Path

from unidata_skill.cli import main
from unidata_skill.inspect import BlendedMVSInspector


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "blendedmvs_tiny"


class BlendedMVSInspectorTests(unittest.TestCase):
    def test_inspect_tiny_fixture(self):
        report = BlendedMVSInspector().inspect(FIXTURE_ROOT)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["dataset"], "blendedmvs")
        self.assertEqual(report["scene_count"], 1)
        self.assertEqual(report["errors"], [])

        scene = report["scenes"][0]
        self.assertEqual(scene["scene_id"], "scene_000")
        self.assertEqual(scene["image_count"], 2)
        self.assertEqual(scene["depth_count"], 2)
        self.assertEqual(scene["camera_count"], 2)
        self.assertTrue(scene["has_pair_file"])

    def test_missing_required_entries_are_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "blended_images").mkdir()
            (root / "blended_images" / "00000000.jpg").write_text("placeholder", encoding="utf-8")

            report = BlendedMVSInspector().inspect(root)

        self.assertEqual(report["status"], "error")
        self.assertTrue(any("missing required entry: pair.txt" in error for error in report["errors"]))
        self.assertTrue(any("missing required entry: cams" in error for error in report["errors"]))

    def test_cli_writes_inspection_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "inspection_report.json"
            exit_code = main(["inspect", "--input", str(FIXTURE_ROOT), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["scene_count"], 1)


if __name__ == "__main__":
    unittest.main()
