"""Campanula-specific interpretations of generic ShimaFlora measurements."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CampanulaGuidePreset:
    """Identify likely guide/oxidation/petal GMM components in Lab a*,b* space."""

    def assign_roles(self, component_centres: np.ndarray) -> dict[str, int]:
        means = np.asarray(component_centres, dtype=float)
        if means.ndim != 2 or means.shape[1] != 2 or len(means) < 3:
            raise ValueError("CampanulaGuidePreset expects >=3 component centres in (a*, b*)")
        magenta_score = means[:, 0] - means[:, 1]
        guide = int(np.argmax(magenta_score))
        remaining = [i for i in range(len(means)) if i != guide]
        yellow_red_score = means[:, 0] + means[:, 1]
        oxidation = int(max(remaining, key=lambda i: yellow_red_score[i]))
        petal = int(min((i for i in remaining if i != oxidation), key=lambda i: np.linalg.norm(means[i])))
        return {"petal": petal, "guide": guide, "oxidation": oxidation}


def _mounted_axis_profile(mask: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Resolve the tube-up axis used by the published Campanula workflow.

    Flattened specimens are mounted with the tube base toward image top. Therefore
    the minimum-area-box side closer to image vertical is biological corolla length;
    it is not necessarily the geometric major axis for fully opened five-lobed flowers.
    The returned profile runs from mounted proximal (top) to distal (bottom).
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2 or not int(m.sum()):
        raise ValueError("mask must be a non-empty 2-D binary array")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    vert1, vert2 = abs(float(e1[1])) / (l1 + 1e-9), abs(float(e2[1])) / (l2 + 1e-9)
    if vert1 >= vert2:
        length_px, width_px, direction = l1, l2, e1
    else:
        length_px, width_px, direction = l2, l1, e2
    u = direction / (np.linalg.norm(direction) + 1e-9)
    if u[1] < 0:
        u = -u
    angle_deg = math.degrees(math.atan2(float(u[1]), float(u[0]))) % 180.0

    h, w = m.shape
    pad = int(max(h, w))
    big = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    centre = (w / 2.0 + pad, h / 2.0 + pad)
    transform = cv2.getRotationMatrix2D(centre, angle_deg - 90.0, 1.0)
    rot = cv2.warpAffine(big, transform, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rot > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    profile = np.zeros(y1 - y0 + 1, dtype=float)
    for k, y in enumerate(range(y0, y1 + 1)):
        row = xs[ys == y]
        profile[k] = float(row.max() - row.min() + 1) if row.size else 0.0
    return length_px, width_px, profile, angle_deg


def _band_median(profile: np.ndarray, lo: float, hi: float) -> float:
    n = len(profile)
    start = max(0, min(n - 1, int(lo * n)))
    stop = max(start + 1, min(n, int(hi * n)))
    return float(np.median(profile[start:stop]))


def campanula_tubular_traits(
    mask: np.ndarray,
    scale_mm_per_px: float = 1.0,
    *,
    width_factor: float = 1.0,
) -> dict[str, float]:
    """Campanula-style flattened tubular-flower traits for tube-up specimens.

    `width_factor=2` reproduces full-flower-equivalent transverse measurements for
    reviewed half-folded specimens. The mounted tube-up assumption is deliberately
    confined to this preset rather than the generic ShimaFlora morphology API.
    """
    if scale_mm_per_px <= 0 or width_factor <= 0:
        raise ValueError("scale_mm_per_px and width_factor must be positive")
    length_px, width_px, prof, angle_deg = _mounted_axis_profile(mask)
    throat = _band_median(prof, 0.04, 0.16) * scale_mm_per_px * width_factor
    mouth = _band_median(prof, 0.72, 0.88) * scale_mm_per_px * width_factor
    length = length_px * scale_mm_per_px
    width = width_px * scale_mm_per_px * width_factor
    return {
        "corolla_length_mm": length,
        "corolla_width_mm": width,
        "throat_width_mm": throat,
        "mouth_width_mm": mouth,
        "corolla_aspect_L_W": length / max(width, 1e-12),
        "tube_flare_W_throat": width / max(throat, 1e-12),
        "mounted_axis_angle_deg": angle_deg,
    }
