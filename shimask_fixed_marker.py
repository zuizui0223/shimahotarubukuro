# -*- coding: utf-8 -*-
"""Extract the common red/green annotation-marker colours measured from shimask.

The red bounds come from raw-vs-shimask differences across the reviewed sheets.
Measured red core across all sheets: mode RGB=(227, 13, 11), median
RGB=(216, 18, 27). Natural purple nectar guides are rejected by requiring red
to exceed blue strongly; purple has similar red and blue values.
"""
from __future__ import annotations

import cv2
import numpy as np


def _clean(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= 3:
            out[labels == index] = 1
    return out


def stroke_masks(raw: np.ndarray, annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return red and green marker strokes directly from measured common RGB.

    ``raw`` is accepted for API compatibility. Marker colours were measured from
    raw-vs-shimask differences, then applied directly to every shimask image.
    """
    if annotated.ndim != 3:
        raise ValueError("annotated must be a BGR colour image")
    b, g, r = cv2.split(annotated.astype(np.int16))

    # Empirical red-marker core over all reviewed sheets:
    # mode=(227,13,11), median=(216,18,27). Keep JPEG-darkened edge pixels while
    # excluding purple guides by requiring a large R-B separation.
    red = (
        (r >= 170)
        & (g <= 80)
        & (b <= 90)
        & (r - g >= 125)
        & (r - b >= 105)
    )

    # Green annotation is similarly highly saturated and green-dominant.
    green = (
        (g >= 170)
        & (r <= 110)
        & (b <= 110)
        & (g - r >= 100)
        & (g - b >= 100)
    )
    return _clean(red), _clean(green)
