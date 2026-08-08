"""Generic 2-D morphology measurements for calibrated floral masks."""
from __future__ import annotations

import math
import cv2
import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    out = (np.asarray(mask) > 0).astype(np.uint8)
    if out.ndim != 2 or out.sum() == 0:
        raise ValueError("mask must be a non-empty 2-D binary array")
    return out


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def measure_shape(mask: np.ndarray, scale_mm_per_px: float = 1.0) -> dict[str, float]:
    """Measure generic 2-D shape traits from a binary floral ROI.

    The function makes no assumption about flower orientation or taxonomy. Length
    and width are the major and minor sides of the minimum-area bounding rectangle.
    """
    m = _binary(mask)
    if scale_mm_per_px <= 0:
        raise ValueError("scale_mm_per_px must be positive")
    contour = _largest_contour(m)
    (_cx, _cy), (w, h), angle = cv2.minAreaRect(contour)
    major_px, minor_px = max(w, h), min(w, h)
    area_px = float(m.sum())
    perimeter_px = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area_px = max(float(cv2.contourArea(hull)), 1e-12)
    s = float(scale_mm_per_px)
    area = area_px * s * s
    perimeter = perimeter_px * s
    circularity = 4.0 * math.pi * area_px / max(perimeter_px * perimeter_px, 1e-12)
    return {
        "area_mm2": area,
        "major_axis_mm": major_px * s,
        "minor_axis_mm": minor_px * s,
        "aspect_ratio": major_px / max(minor_px, 1e-12),
        "perimeter_mm": perimeter,
        "solidity": area_px / hull_area_px,
        "circularity": circularity,
        "orientation_deg": float(angle),
    }


def width_profile(mask: np.ndarray, axis: str = "major") -> np.ndarray:
    """Return silhouette width along the major or minor oriented-box axis.

    Widths are returned in pixels; multiply by the image scale for physical units.
    """
    m = _binary(mask)
    contour = _largest_contour(m)
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    major_is_w = w >= h
    major_angle = angle if major_is_w else angle + 90.0
    target_angle = major_angle if axis == "major" else major_angle + 90.0
    if axis not in {"major", "minor"}:
        raise ValueError("axis must be 'major' or 'minor'")
    M = cv2.getRotationMatrix2D((cx, cy), target_angle - 90.0, 1.0)
    rot = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rot > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    widths = []
    for y in range(y0, y1 + 1):
        row = xs[ys == y]
        widths.append(float(row.max() - row.min() + 1) if row.size else 0.0)
    return np.asarray(widths, dtype=float)


def width_at(mask: np.ndarray, position: float, axis: str = "major", band: float = 0.02) -> float:
    """Median silhouette width around a relative position (0=proximal, 1=distal).

    The orientation is geometric, not biological. Taxon-specific presets may assign
    biological meanings such as throat or mouth after defining the proximal end.
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError("position must be between 0 and 1")
    prof = width_profile(mask, axis=axis)
    n = len(prof)
    centre = position * max(n - 1, 0)
    half = max(1, int(round(band * n)))
    lo = max(0, int(round(centre)) - half)
    hi = min(n, int(round(centre)) + half + 1)
    return float(np.median(prof[lo:hi]))
