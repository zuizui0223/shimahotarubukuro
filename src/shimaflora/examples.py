"""Small deterministic examples for tutorials, tests, and demonstrations."""
from __future__ import annotations

import cv2
import numpy as np


def synthetic_flower(size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(image_rgb, roi_mask, pattern_mask)`` for a synthetic flower.

    The example is generated rather than copied from a specimen, so it is freely
    redistributable and stable across package versions. It intentionally contains a
    tapered corolla-like ROI and several intrafloral spots suitable for exercising
    morphology, pattern, and QC APIs.
    """
    if size < 96:
        raise ValueError("size must be at least 96 pixels")
    image = np.full((size, size, 3), 242, dtype=np.uint8)
    roi = np.zeros((size, size), dtype=np.uint8)
    pattern = np.zeros_like(roi)

    s = float(size)
    pts = np.array(
        [
            [0.42 * s, 0.14 * s],
            [0.58 * s, 0.14 * s],
            [0.70 * s, 0.42 * s],
            [0.79 * s, 0.70 * s],
            [0.64 * s, 0.83 * s],
            [0.50 * s, 0.75 * s],
            [0.36 * s, 0.83 * s],
            [0.21 * s, 0.70 * s],
            [0.30 * s, 0.42 * s],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, [pts], 1)
    image[roi > 0] = np.array([225, 188, 220], dtype=np.uint8)

    centres = [
        (0.43, 0.38, 0.025),
        (0.55, 0.41, 0.022),
        (0.39, 0.52, 0.020),
        (0.51, 0.55, 0.026),
        (0.61, 0.58, 0.019),
        (0.47, 0.66, 0.017),
    ]
    for x, y, r in centres:
        cv2.circle(pattern, (int(x * s), int(y * s)), max(2, int(r * s)), 1, -1)
    pattern &= roi
    image[pattern > 0] = np.array([112, 48, 145], dtype=np.uint8)
    return image, roi, pattern


__all__ = ["synthetic_flower"]
