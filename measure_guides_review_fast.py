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
MAX_ORGAN_CANDIDATES_PER_SHEET = 8


def _component_rows(
    line_union: np.ndarray,
    candidate: np.ndarray,
    chroma: np.ndarray,
    warmth: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line_union, 8)
    for index in range(1, n):
        if stats[index, cv2.CC_STAT_AREA] < 15:
            continue
        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        length_px = max(rw, rh)
        width_px = min(rw, rh)
        if width_px < 1:
            continue
        length_mm = length_px * base.MM_PX
        width_mm = width_px * base.MM_PX
        aspect = length_px / width_px
        pixels = (component > 0) & (candidate > 0)
        mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0
        mean_warmth = float(warmth[pixels].mean()) if pixels.any() else 0.0
        support = int(pixels.sum()) / max(float(component.sum()), 1.0)
        if not (
            review.MIN_ORGAN_LENGTH_MM <= length_mm <= 40
            and width_mm <= 6.0
            and aspect >= 2.4
            and mean_chroma >= 1.5
            and mean_warmth >= 1.5
            and support >= 0.08
        ):
            continue
        score = mean_warmth * mean_chroma * support * math.sqrt(length_mm)
        rows.append(dict(
            cx=round(cx, 2),
            cy=round(cy, 2),
            length_mm=round(length_mm, 2),
            width_mm=round(width_mm, 2),
            aspect=round(aspect, 2),
            angle_deg=round(angle, 2),
            score=round(score, 3),
            organ_type_auto="unclassified_reproductive_organ",
            organ_type_FILL="",
            exclude_FILL="",
        ))
    return rows


def organs_fast(img: np.ndarray, corolla_mask: np.ndarray, top: int):
    lightness, a_star, b_star = base.channels(img)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    candidate = (
        (lightness < 253)
        & ((chroma > 1.6) | (a_star > 1.5) | (b_star > 1.5))
        & ~((lightness < 145) & (chroma < 12))
    ).astype(np.uint8)
    candidate[:top] = 0
    candidate = review.apply_current_exclusions(candidate)

    exclusion = cv2.dilate(
        corolla_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    candidate[exclusion > 0] = 0
    outside = (corolla_mask == 0).astype(np.uint8)
    distance_from_corolla = cv2.distanceTransform(outside, cv2.DIST_L2, 5)

    small = cv2.resize(
        candidate,
        None,
        fx=ORGAN_SEARCH_SCALE,
        fy=ORGAN_SEARCH_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_line = max(12, int(round(review.MIN_ORGAN_LENGTH_MM / base.MM_PX * ORGAN_SEARCH_SCALE)))
    max_gap = max(3, int(round(1.5 / base.MM_PX * ORGAN_SEARCH_SCALE)))
    lines = cv2.HoughLinesP(
        small * 255,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=min_line,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    raw_lines = np.asarray(lines).reshape(-1, 4)
    lengths_sq = (raw_lines[:, 2] - raw_lines[:, 0]) ** 2 + (raw_lines[:, 3] - raw_lines[:, 1]) ** 2
    raw_lines = raw_lines[np.argsort(lengths_sq)[::-1][:MAX_HOUGH_LINES]]

    inverse = 1.0 / ORGAN_SEARCH_SCALE
    accepted: list[tuple[int, int, int, int]] = []
    for sx1, sy1, sx2, sy2 in raw_lines:
        x1, y1, x2, y2 = [int(round(value * inverse)) for value in (sx1, sy1, sx2, sy2)]
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_mm = length_px * base.MM_PX
        if not review.MIN_ORGAN_LENGTH_MM <= length_mm <= 35:
            continue
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        if not (0 <= cy < candidate.shape[0] and 0 <= cx < candidate.shape[1]):
            continue
        boundary_distance_mm = float(distance_from_corolla[cy, cx]) * base.MM_PX
        if not 0.5 <= boundary_distance_mm <= 25.0:
            continue

        tube = np.zeros_like(candidate)
        cv2.line(tube, (x1, y1), (x2, y2), 255, 11)
        pixels = (tube > 0) & (candidate > 0)
        support = int(pixels.sum()) / max(length_px * 11.0, 1.0)
        mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0
        mean_warmth = float(b_star[pixels].mean()) if pixels.any() else 0.0
        if support >= 0.12 and mean_chroma >= 1.5 and mean_warmth >= 1.5:
            accepted.append((x1, y1, x2, y2))

    if not accepted:
        return []

    line_union = np.zeros_like(candidate)
    for x1, y1, x2, y2 in accepted:
        cv2.line(line_union, (x1, y1), (x2, y2), 255, 15)
    line_union = cv2.morphologyEx(
        line_union,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )

    rows = _component_rows(line_union, candidate, chroma, b_star)
    rows = sorted(rows, key=lambda row: row.get("score", 0), reverse=True)
    return sorted(rows[:MAX_ORGAN_CANDIDATES_PER_SHEET], key=lambda row: (row["cy"], row["cx"]))


def main() -> None:
    review.install_reviewed_overrides()
    v2.organs = organs_fast
    v2.main()


if __name__ == "__main__":
    main()
