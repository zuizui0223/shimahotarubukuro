"""Shared scan calibration and image-loading utilities.

The former automatic corolla/guide extraction implementation was removed from the
publication pipeline. Measurements now begin with the human-reviewed ``shimask``
annotations. This module intentionally contains only the fixed 300-DPI conversion,
reviewed-ROI size bounds and canonical image loading used across the retained scripts.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

# All specimen scans are 300 DPI and were cross-checked against the printed ruler.
PXCM = 300.0 / 2.54
MM_PX = 10.0 / PXCM
MM2_PX = MM_PX ** 2

# Broad reviewed-corolla bounds used only to reject annotation noise/non-corolla loops.
AREA_MM2_MIN = 80.0
AREA_MM2_MAX = 3200.0


def load_bgr(path: str) -> np.ndarray:
    """Load a scan with EXIF orientation applied, returning OpenCV BGR order."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
