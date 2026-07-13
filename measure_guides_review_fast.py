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
MIN_AXIS_ANGLE_FROM_HORIZONTAL = 30.0


def _long_axis_angle(rect_width: float, rect_height: float, angle: float) -> float:
    """Return the long-axis angle in [-90, 90], where 0 is horizontal."""
    axis = angle if rect_width >= rect_height else angle + 90.0
    while axis > 90.0:
        axis -= 180.0
    while axis < -90.0:
        axis += 180.0
    return axis


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
        (cx, cy), (rw, rh), rect_angle = cv2.minAreaRect(contour)
        length_px = max(rw, rh)
        width_px = min(rw, rh)
        if width_px < 1:
            continue
        axis_angle = _long_axis_angle(rw, rh, rect_angle)
        if abs(axis_angle) < MIN_AXIS_ANGLE_FROM_HORIZONTAL:
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
            and mean_chroma >= 1.3
            and mean_warmth >= 1.2
            and support >= 0.07
        ):
            continue
        score = mean_warmth * mean_chroma * support * math.sqrt(length_mm)
        rows.append(dict(
            cx=round(cx, 2),
            cy=round(cy, 2),
            length_mm=round(length_mm, 2),
            width_mm=round(width_mm, 2),
            aspect=round(aspect, 2),
            angle_deg=round(axis_angle, 2),
            score=round(score, 3),
            organ_type_auto="unclassified_reproductive_organ",
            organ_type_FILL="",
            exclude_FILL="",
        ))
    return rows


def _merge_nearby_rows(rows: list[dict]) -> list[dict]:
    """Keep one best row for nearby collinear fragments of the same organ."""
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item.get("score", 0), reverse=True):
        duplicate = False
        for other in kept:
            distance_px = math.hypot(row["cx"] - other["cx"], row["cy"] - other["cy"])
            merge_radius_px = 0.65 * max(row["length_mm"], other["length_mm"]) / base.MM_PX
            angle_diff = abs(row["angle_deg"] - other["angle_deg"])
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if distance_px <= merge_radius_px and angle_diff <= 30.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def organs_fast(
    img: np.ndarray,
    corolla_mask: np.ndarray,
    top: int,
    max_candidates: int = MAX_ORGAN_CANDIDATES_PER_SHEET,
):
    lightness, a_star, b_star = base.channels(img)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    candidate = (
        (lightness < 253)
        & ((chroma > 1.4) | (a_star > 1.3) | (b_star > 1.2))
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
    max_gap = max(3, int(round(2.0 / base.MM_PX * ORGAN_SEARCH_SCALE)))
    lines = cv2.HoughLinesP(
        small * 255,
        1,
        np.pi / 180,
        threshold=16,
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
        line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while line_angle > 90.0:
            line_angle -= 180.0
        while line_angle < -90.0:
            line_angle += 180.0
        if abs(line_angle) < MIN_AXIS_ANGLE_FROM_HORIZONTAL:
            continue

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        if not (0 <= cy < candidate.shape[0] and 0 <= cx < candidate.shape[1]):
            continue
        boundary_distance_mm = float(distance_from_corolla[cy, cx]) * base.MM_PX
        if not 0.5 <= boundary_distance_mm <= 35.0:
            continue

        tube = np.zeros_like(candidate)
        cv2.line(tube, (x1, y1), (x2, y2), 255, 11)
        pixels = (tube > 0) & (candidate > 0)
        support = int(pixels.sum()) / max(length_px * 11.0, 1.0)
        mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0
        mean_warmth = float(b_star[pixels].mean()) if pixels.any() else 0.0
        if support >= 0.10 and mean_chroma >= 1.3 and mean_warmth >= 1.2:
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

    rows = _merge_nearby_rows(_component_rows(line_union, candidate, chroma, b_star))
    rows = sorted(rows, key=lambda row: row.get("score", 0), reverse=True)
    return sorted(rows[:max_candidates], key=lambda row: (row["cy"], row["cx"]))


def organs_review_candidates(
    img: np.ndarray,
    corolla_mask: np.ndarray,
    top: int,
    max_candidates: int = 60,
) -> list[dict]:
    """High-recall candidates for manual O-number review in the app."""
    fast_rows = organs_fast(
        img, corolla_mask, top, max_candidates=max_candidates
    )
    component_rows = v2.organs(img, corolla_mask, top)
    outside = (corolla_mask == 0).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    height, width = corolla_mask.shape
    margin_x = max(20, int(round(width * 0.015)))
    min_y = max(top, int(round(height * 0.08)))

    def spatially_plausible(row, min_length_mm):
        cx = int(round(float(row["cx"])))
        cy = int(round(float(row["cy"])))
        if not (
            margin_x <= cx < width - margin_x
            and min_y <= cy < height - 20
        ):
            return False
        distance_mm = float(distance[cy, cx]) * base.MM_PX
        return (
            min_length_mm <= float(row["length_mm"]) <= 40.0
            and abs(float(row["angle_deg"])) >= 30.0
            and 0.5 <= distance_mm <= 30.0
        )

    filtered_fast = [
        row for row in fast_rows if spatially_plausible(row, 8.0)
    ]
    high_recall = []
    for row in component_rows:
        if not spatially_plausible(row, 5.0):
            continue
        high_recall.append({
            **row,
            "detection_source": "app_high_recall_component",
        })

    combined = [
        {**row, "detection_source": "app_hough_detector"}
        for row in filtered_fast
    ] + high_recall
    return review._deduplicate_organs(combined)[:max_candidates]


def main() -> None:
    review.install_reviewed_overrides()
    v2.organs = organs_fast
    v2.main()


if __name__ == "__main__":
    main()
