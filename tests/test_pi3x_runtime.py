import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unidata_skill.pi3x import resolve_pi3_root


class Pi3XRuntimeTests(unittest.TestCase):
    def test_resolve_pi3_root_raises_without_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_default = Path(tmpdir) / "thirdparty" / "Pi3"
            with patch("unidata_skill.pi3x.default_pi3_root", return_value=missing_default):
                with self.assertRaises(FileNotFoundError):
                    resolve_pi3_root()

    def test_resolve_pi3_root_accepts_checkout_with_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Pi3"
            root.joinpath("datasets").mkdir(parents=True)

            with patch("unidata_skill.pi3x.default_pi3_root", return_value=root):
                self.assertEqual(resolve_pi3_root(), root.resolve())


if __name__ == "__main__":
    unittest.main()
