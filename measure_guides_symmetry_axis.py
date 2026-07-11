# -*- coding: utf-8 -*-
"""Mask-first symmetry-axis estimation for reviewed flattened corollas.

Workflow locked for manual QC:
1. Start from the accepted per-corolla polygon/mask, not the raw image.
2. Search candidate axes and choose the line that maximizes reflection overlap
   between the mask and its mirror image.
3. Constrain the axis family to the biologically valid base-up orientation
   (ruler side = corolla base for the reviewed scan layout).
4. Project the chosen axis back onto the raw scan for visual validation.
5. Measure longitudinal traits along that axis and width traits perpendicular
   to it.

For fully opened 5-lobed corollas this axis is the working bilateral symmetry
axis of the flattened display. For half-folded corollas it is only the symmetry
axis of the *observed folded polygon* and must remain flagged as folded_half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SymmetryAxis:
    score_iou: float
    angle_deg_from_x: float
    offset_px: float
    base_xy: tuple[float, float]
    tip_xy: tuple[float, float]


def _reflect_iou(mask: np.ndarray, angle_deg: float, offset_px: float) -> float:
    """IoU between a binary mask and its reflection across a candidate axis."""
    binary = mask.astype(bool)
    ys, xs = np.where(binary)
    if xs.size == 0:
        return 0.0

    points = np.column_stack([xs, ys]).astype(np.float32)
    centroid = points.mean(axis=0)
    theta = math.radians(angle_deg)
    axis = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    origin = centroid + float(offset_px) * normal

    rel = points - origin
    reflected = (
        origin
        + (rel @ axis)[:, None] * axis
        - (rel @ normal)[:, None] * normal
    )
    xi = np.rint(reflected[:, 0]).astype(int)
    yi = np.rint(reflected[:, 1]).astype(int)
    valid = (
        (xi >= 0)
        & (xi < binary.shape[1])
        & (yi >= 0)
        & (yi < binary.shape[0])
    )
    overlap = int(binary[yi[valid], xi[valid]].sum())
    union = int(binary.sum()) + int(valid.sum()) - overlap
    return float(overlap / max(union, 1))


def estimate_symmetry_axis(
    mask: np.ndarray,
    *,
    angle_min_deg: float = 45.0,
    angle_max_deg: float = 135.0,
    angle_step_deg: float = 0.75,
    max_offset_px: float = 65.0,
    offset_step_px: float = 8.0,
) -> SymmetryAxis:
    """Estimate a base-up reflection-symmetry axis from one accepted corolla mask.

    The default angle window keeps the candidate family broadly vertical while
    allowing substantial specimen tilt. The returned direction is always ordered
    from the upper/ruler-side endpoint (base) toward the lower lobe/tip side.
    """
    binary = mask.astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if xs.size < 10:
        raise ValueError("Mask is empty or too small for symmetry estimation")

    # Search on a downscaled mask for speed, then convert the winning offset back.
    scale = 0.25
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    best_score = -1.0
    best_angle = 90.0
    best_offset_small = 0.0

    for angle in np.arange(angle_min_deg, angle_max_deg + 1e-9, angle_step_deg):
        for offset in np.arange(
            -max_offset_px * scale,
            max_offset_px * scale + 1e-9,
            offset_step_px * scale,
        ):
            score = _reflect_iou(small, float(angle), float(offset))
            if score > best_score:
                best_score = score
                best_angle = float(angle)
                best_offset_small = float(offset)

    offset_px = best_offset_small / scale
    points = np.column_stack([xs, ys]).astype(float)
    centroid = points.mean(axis=0)
    theta = math.radians(best_angle)
    axis = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    if axis[1] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]], dtype=float)
    origin = centroid + offset_px * normal

    longitudinal = (points - origin) @ axis
    lo, hi = np.percentile(longitudinal, [1.0, 99.0])
    base = origin + lo * axis
    tip = origin + hi * axis

    return SymmetryAxis(
        score_iou=round(float(best_score), 4),
        angle_deg_from_x=round(float(best_angle), 3),
        offset_px=round(float(offset_px), 3),
        base_xy=(round(float(base[0]), 3), round(float(base[1]), 3)),
        tip_xy=(round(float(tip[0]), 3), round(float(tip[1]), 3)),
    )


def rotate_mask_to_symmetry_axis(mask: np.ndarray, axis: SymmetryAxis) -> np.ndarray:
    """Rotate a mask so the reviewed symmetry axis is vertical, base at the top."""
    base = np.asarray(axis.base_xy, dtype=float)
    tip = np.asarray(axis.tip_xy, dtype=float)
    direction = tip - base
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    centre = tuple(((base + tip) / 2.0).tolist())
    matrix = cv2.getRotationMatrix2D(centre, angle - 90.0, 1.0)
    height, width = mask.shape[:2]
    rotated = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
    )
    ys, xs = np.where(rotated > 0)
    if xs.size == 0:
        return rotated
    return rotated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
