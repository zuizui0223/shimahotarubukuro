# -*- coding: utf-8 -*-
"""Extract measured red/green annotation-marker strokes from shimask images.

Strict marker-colour cores are used as seeds. JPEG-blended edge pixels are then
recovered only when connected to those cores. This keeps the complete hand-drawn
line while excluding natural purple nectar guides.
"""
from __future__ import annotations

import cv2
import numpy as np


def _components_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep candidate components that contain or directly touch a strict seed."""
    candidate = candidate.astype(np.uint8)
    seed = seed.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    seed_touch = cv2.dilate(seed, kernel) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    out = np.zeros_like(candidate)
    for index in range(1, count):
        component = labels == index
        if int(stats[index, cv2.CC_STAT_AREA]) >= 3 and np.any(component & seed_touch):
            out[component] = 1
    return out


def stroke_masks(raw: np.ndarray, annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return complete red and green marker strokes at annotation resolution."""
    if annotated.ndim != 3:
        raise ValueError("annotated must be a BGR colour image")
    b, g, r = cv2.split(annotated.astype(np.int16))

    # Measured red cores across all sheets are centred near RGB=(216,18,27).
    # Pale JPEG edge blending reaches roughly RGB=(255,175,175), so recover
    # those pixels only when connected to a strict dark-red core.
    red_seed = (
        (r >= 170)
        & (g <= 90)
        & (b <= 100)
        & (r - g >= 110)
        & (r - b >= 90)
    )
    red_candidate = (
        (r >= 145)
        & (g <= 195)
        & (b <= 195)
        & (r - g >= 45)
        & (r - b >= 35)
    )

    # Green marker uses the same seed-and-grow strategy. Natural leaf tissue is
    # not retained unless it connects to a marker-green core.
    green_seed = (
        (g >= 170)
        & (r <= 100)
        & (b <= 120)
        & (g - r >= 100)
        & (g - b >= 80)
    )
    green_candidate = (
        (g >= 135)
        & (r <= 195)
        & (b <= 195)
        & (g - r >= 40)
        & (g - b >= 35)
    )

    red = _components_touching_seed(red_candidate, red_seed)
    green = _components_touching_seed(green_candidate, green_seed)
    return red, green
