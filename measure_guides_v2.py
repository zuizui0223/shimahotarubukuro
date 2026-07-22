"""Minimal helpers retained for reviewed annotation extraction.

The automatic v2 segmentation runner was removed. ``shimask_input.py`` uses only the
ruler/header boundary detector and the contour summary below to recover and validate
human-reviewed red/green annotations.
"""
from __future__ import annotations

import cv2
import numpy as np


def specimen_top(image: np.ndarray) -> int:
    """Return the row below a ruler/header band at the top of a scan."""
    height, _width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical_edges = (
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) > 60
    ).astype(np.float32)
    profile = cv2.GaussianBlur(
        vertical_edges.mean(axis=1).reshape(-1, 1), (1, 31), 0
    ).ravel()
    margin = max(24, int(height * 0.012))
    limit = int(height * 0.35)
    upper = profile[:limit]
    if upper.size and float(upper.max()) > 0.28:
        row = int(np.argmax(upper))
        while row + 1 < limit and profile[row + 1] > 0.10:
            row += 1
        return min(height - 1, row + margin)
    return margin


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
