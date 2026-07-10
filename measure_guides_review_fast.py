# -*- coding: utf-8 -*-
"""Fast reviewed runner: full-resolution corollas, downscaled organ line search."""
from __future__ import annotations

import math
import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review

ORGAN_SEARCH_SCALE = 0.40
MAX_HOUGH_LINES = 300


def organs_fast(img: np.ndarray, corolla_mask: np.ndarray, top: int):
    lightness, a_star, b_star = base.channels(img)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    candidate = (
        (lightness < 253)
        & ((chroma > 1.4) | (a_star > 1.3) | (b_star > 1.2))
        & ~((lightness < 145) & (chroma < 12))
    ).astype(np.uint8)
    candidate[:top] = 0
    exclusion = cv2.dilate(
        corolla_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    candidate[exclusion > 0] = 0

    small = cv2.resize(
        candidate,
        None,
        fx=ORGAN_SEARCH_SCALE,
        fy=ORGAN_SEARCH_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_line = max(12, int(round(review.MIN_ORGAN_LENGTH_MM / base.MM_PX * ORGAN_SEARCH_SCALE)))
    max_gap = max(4, int(round(3.0 / base.MM_PX * ORGAN_SEARCH_SCALE)))
    lines = cv2.HoughLinesP(
        small * 255,
        1,
        np.pi / 180,
        threshold=16,
        minLineLength=min_line,
        maxLineGap=max_gap,
    )

    rows = []
    if lines is None:
        return rows

    raw_lines = np.asarray(lines).reshape(-1, 4)
    lengths_sq = (raw_lines[:, 2] - raw_lines[:, 0]) ** 2 + (raw_lines[:, 3] - raw_lines[:, 1]) ** 2
    order = np.argsort(lengths_sq)[::-1][:MAX_HOUGH_LINES]
    raw_lines = raw_lines[order]

    inverse = 1.0 / ORGAN_SEARCH_SCALE
    for sx1, sy1, sx2, sy2 in raw_lines:
        x1, y1, x2, y2 = [int(round(value * inverse)) for value in (sx1, sy1, sx2, sy2)]
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_mm = length_px * base.MM_PX
        if not review.MIN_ORGAN_LENGTH_MM <= length_mm <= 45:
            continue

        tube = np.zeros_like(candidate)
        cv2.line(tube, (x1, y1), (x2, y2), 255, 11)
        pixels = (tube > 0) & (candidate > 0)
        support = int(pixels.sum()) / max(length_px * 11.0, 1.0)
        mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0
        mean_warmth = float(b_star[pixels].mean()) if pixels.any() else 0.0
        if support < 0.10 or mean_chroma < 1.2 or mean_warmth < -1.0:
            continue

        width_mm = max(0.15, min(5.0, support * 11.0 * base.MM_PX))
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        rows.append(dict(
            cx=round((x1 + x2) / 2.0, 2),
            cy=round((y1 + y2) / 2.0, 2),
            length_mm=round(length_mm, 2),
            width_mm=round(width_mm, 2),
            aspect=round(length_mm / max(width_mm, 0.01), 2),
            angle_deg=round(angle, 2),
            organ_type_auto="unclassified_reproductive_organ",
            organ_type_FILL="",
            exclude_FILL="",
        ))
    return review._deduplicate_organs(rows)


def main() -> None:
    review.install_reviewed_overrides()
    v2.organs = organs_fast
    v2.main()


if __name__ == "__main__":
    main()
