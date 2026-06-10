import unittest

import torch

from unidata_skill.datasets.pi3x_validator import validate_pi3x_batch, validate_pi3x_dataset, validate_pi3x_view


def make_view():
    return {
        "img": torch.zeros((3, 4, 4), dtype=torch.float32),
        "depthmap": torch.ones((4, 4), dtype=torch.float32),
        "camera_intrinsics": torch.tensor([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        "camera_pose": torch.eye(4, dtype=torch.float32),
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

    def test_validate_pi3x_view_accepts_chw_tensor(self):
        self.assertEqual(validate_pi3x_view(make_view(), 0, 0), [])

    def test_validate_pi3x_batch_accepts_batched_chw_image(self):
        batch_view = {
            "img": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
            "depthmap": torch.ones((2, 4, 4), dtype=torch.float32),
            "camera_intrinsics": torch.stack([make_view()["camera_intrinsics"], make_view()["camera_intrinsics"]]),
            "camera_pose": torch.stack([torch.eye(4, dtype=torch.float32), torch.eye(4, dtype=torch.float32)]),
            "dataset": ["dummy", "dummy"],
            "label": ["scene", "scene"],
            "instance": ["frame0", "frame1"],
        }

        self.assertEqual(validate_pi3x_batch([batch_view], 0), [])

    def test_validate_pi3x_view_rejects_hwc_numpy_image(self):
        view = make_view()
        view["img"] = view["img"].permute(1, 2, 0).numpy()

        errors = validate_pi3x_view(view, 0, 0)

        self.assertTrue(any("img must be torch.Tensor" in error for error in errors))

    def test_validate_pi3x_view_reports_missing_required_fields(self):
        view = make_view()
        del view["camera_pose"]

        errors = validate_pi3x_view(view, 0, 0)

        self.assertTrue(any("missing camera_pose" in error for error in errors))
        self.assertTrue(any("camera_pose must be torch.Tensor" in error for error in errors))

    def test_validate_pi3x_dataset_accepts_dataset_protocol(self):
        result = validate_pi3x_dataset(DummyPi3Dataset(), expected_frame_num=2, max_samples=1)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.dataset_len, 1)
        self.assertEqual(result.checked_samples, 1)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
