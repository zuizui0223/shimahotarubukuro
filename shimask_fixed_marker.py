# -*- coding: utf-8 -*-
"""Extract the common red/green annotation-marker colours measured from shimask.

The colour bounds come from raw-vs-shimask differences on the 19 aligned sheets:
red cores consistently had R=255 with low G/B, and green cores had G near 255.
Natural purple nectar guides are rejected because red and blue are similar rather
than strongly red-dominant.
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

    ``raw`` is accepted for API compatibility but is not needed after the marker
    colours were measured from the aligned raw-vs-shimask pairs.
    """
    if annotated.ndim != 3:
        raise ValueError("annotated must be a BGR colour image")
    b, g, r = cv2.split(annotated.astype(np.int16))

    # Measured red marker core: R mode 255 on all 19 aligned sheets; G/B modes
    # stayed low. Strong R-vs-B separation explicitly excludes purple guides.
    red = (
        (r >= 235)
        & (r - g >= 120)
        & (r - b >= 110)
        & (g <= 115)
        & (b <= 125)
    )

    # Measured green marker core: G near 255 and strongly dominant over R/B.
    green = (
        (g >= 220)
        & (g - r >= 90)
        & (g - b >= 90)
        & (r <= 145)
        & (b <= 145)
    )
    return _clean(red), _clean(green)
