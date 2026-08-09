"""Visual quality-control overlays for generic floral masks and patterns.

All public helpers return RGB ``numpy.uint8`` arrays and never write files. This
keeps visualization composable in notebooks, web apps, tests, and publication
pipelines without imposing an output format.
"""
from __future__ import annotations

import cv2
import numpy as np


def _rgb(image: np.ndarray, *, input_order: str = "rgb") -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        base = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        base = np.clip(arr, 0, 255).astype(np.uint8).copy()
        if input_order.lower() == "bgr":
            base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        elif input_order.lower() != "rgb":
            raise ValueError("input_order must be 'rgb' or 'bgr'")
    else:
        raise ValueError("image must be a 2-D grayscale or 3-channel array")
    return base


def _mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    out = np.asarray(mask) > 0
    if out.ndim != 2 or out.shape != shape:
        raise ValueError(f"{name} must be a 2-D mask matching image height/width")
    return out.astype(np.uint8)


def overlay_roi(
    image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    input_order: str = "rgb",
    line_width: int = 2,
) -> np.ndarray:
    """Draw the external ROI boundary on an image and return an RGB array."""
    out = _rgb(image, input_order=input_order)
    roi = _mask(roi_mask, out.shape[:2], "roi_mask")
    if roi.sum() == 0:
        raise ValueError("roi_mask is empty")
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 0), max(int(line_width), 1))
    return out


def overlay_morphology(
    image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    input_order: str = "rgb",
    line_width: int = 2,
) -> np.ndarray:
    """Draw the minimum-area box and its major/minor axes for a floral ROI.

    This visualizes the same orientation-free geometry used by
    :func:`shimaflora.measure_shape`; it does not assign proximal/distal biology.
    """
    out = overlay_roi(image, roi_mask, input_order=input_order, line_width=line_width)
    roi = _mask(roi_mask, out.shape[:2], "roi_mask")
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)
    cv2.polylines(out, [box], True, (255, 255, 0), max(int(line_width), 1))

    theta_w = np.deg2rad(angle)
    u_w = np.array([np.cos(theta_w), np.sin(theta_w)], dtype=float)
    u_h = np.array([-np.sin(theta_w), np.cos(theta_w)], dtype=float)
    if w >= h:
        u_major, r_major = u_w, w / 2.0
        u_minor, r_minor = u_h, h / 2.0
    else:
        u_major, r_major = u_h, h / 2.0
        u_minor, r_minor = u_w, w / 2.0
    centre = np.array([cx, cy], dtype=float)

    def endpoints(u: np.ndarray, radius: float) -> tuple[tuple[int, int], tuple[int, int]]:
        a = np.rint(centre - radius * u).astype(int)
        b = np.rint(centre + radius * u).astype(int)
        return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))

    cv2.line(out, *endpoints(u_major, r_major), (0, 255, 0), max(int(line_width), 1))
    cv2.line(out, *endpoints(u_minor, r_minor), (0, 255, 255), max(int(line_width), 1))
    return out


def overlay_pattern(
    image: np.ndarray,
    roi_mask: np.ndarray,
    pattern_mask: np.ndarray,
    *,
    input_order: str = "rgb",
    alpha: float = 0.45,
    line_width: int = 2,
) -> np.ndarray:
    """Overlay any binary intrafloral pattern inside its ROI.

    Pattern pixels outside ``roi_mask`` are ignored. The returned image also shows
    the ROI boundary, making segmentation spill-over immediately visible.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    out = _rgb(image, input_order=input_order)
    roi = _mask(roi_mask, out.shape[:2], "roi_mask").astype(bool)
    pattern = _mask(pattern_mask, out.shape[:2], "pattern_mask").astype(bool) & roi
    tint = out.copy()
    tint[pattern] = np.array([255, 0, 255], dtype=np.uint8)
    out = cv2.addWeighted(tint, float(alpha), out, 1.0 - float(alpha), 0.0)
    return overlay_roi(out, roi, input_order="rgb", line_width=line_width)


__all__ = ["overlay_roi", "overlay_morphology", "overlay_pattern"]
