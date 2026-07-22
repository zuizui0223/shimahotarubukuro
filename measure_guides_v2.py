"""Minimal reviewed-mask geometry helper.

The former automatic v2 segmentation runner was removed from the publication
repository. ``shimask_input.py`` retains only this contour summary to validate and
order the human-reviewed red corolla regions.
"""
from __future__ import annotations

import cv2
import numpy as np


def metrics(mask: np.ndarray):
    """Return basic oriented-box and contour metrics for one binary ROI."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    width, height = cv2.minAreaRect(contour)[1]
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    area = float(mask.sum())
    long_side = max(width, height)
    short_side = min(width, height)
    return {
        "contour": contour,
        "area_px": area,
        "length_px": long_side,
        "width_px": short_side,
        "solidity": area / hull_area if hull_area else 0.0,
        "aspect": long_side / max(short_side, 1e-6),
    }
