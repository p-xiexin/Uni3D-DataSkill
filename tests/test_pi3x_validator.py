import unittest

import numpy as np

from pi3_test_utils import install_fake_pi3

install_fake_pi3()

from unidata_skill.datasets import validate_pi3x_dataset, validate_pi3x_view


def make_view():
    return {
        "img": np.zeros((4, 4, 3), dtype=np.uint8),
        "depthmap": np.ones((4, 4), dtype=np.float32),
        "camera_intrinsics": np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "camera_pose": np.eye(4, dtype=np.float32),
        "dataset": "dummy",
        "label": "scene",
        "instance": "frame",
    }


class DummyPi3Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return [make_view(), make_view()]


class Pi3XValidatorTests(unittest.TestCase):
    def test_validate_pi3x_view_accepts_required_fields(self):
        self.assertEqual(validate_pi3x_view(make_view(), 0, 0), [])

    def test_validate_pi3x_view_reports_missing_required_fields(self):
        view = make_view()
        del view["camera_pose"]

        errors = validate_pi3x_view(view, 0, 0)

        self.assertTrue(any("missing camera_pose" in error for error in errors))
        self.assertTrue(any("invalid camera_pose" in error for error in errors))

    def test_validate_pi3x_dataset_accepts_dataset_protocol(self):
        result = validate_pi3x_dataset(DummyPi3Dataset(), expected_frame_num=2, max_samples=1)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.dataset_len, 1)
        self.assertEqual(result.checked_samples, 1)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
