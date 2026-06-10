from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class BaseDataset:
    def __init__(
        self,
        resolution: list[int] | tuple[int, int] | None = None,
        frame_num: int = 2,
        shuffle: bool = False,
        **_: Any,
    ) -> None:
        self.frame_num = frame_num
        self.shuffle = shuffle
        self._resolutions = [list(resolution or [512, 384])]
        self._rng = np.random.default_rng(2024)

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        return self._get_views(idx, self._resolutions[0], self._rng)

    def _crop_resize_if_necessary(
        self,
        img: np.ndarray,
        depthmap: np.ndarray,
        intrinsics: np.ndarray,
        resolution: tuple[int, int] | list[int],
        rng: np.random.Generator | None = None,
        info: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_width, target_height = int(resolution[0]), int(resolution[1])
        height, width = img.shape[:2]
        if (width, height) == (target_width, target_height):
            return img, depthmap.astype(np.float32), intrinsics.astype(np.float32)

        scale_w = target_width / width
        scale_h = target_height / height
        intrinsics = intrinsics.copy().astype(np.float32)
        intrinsics[0, 0] *= scale_w
        intrinsics[1, 1] *= scale_h
        intrinsics[0, 2] *= scale_w
        intrinsics[1, 2] *= scale_h
        resized_img = np.asarray(Image.fromarray(img).resize((target_width, target_height), Image.Resampling.BILINEAR))
        resized_depth = np.asarray(Image.fromarray(depthmap).resize((target_width, target_height), Image.Resampling.NEAREST))
        return resized_img, resized_depth.astype(np.float32), intrinsics
