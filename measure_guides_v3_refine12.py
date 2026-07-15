# -*- coding: utf-8 -*-
"""Detect detached reproductive organs as line segments near each corolla.

The reviewed shimask images are used only for evaluation. Runtime detection uses
raw scans, the established colour-supported detector, and a geometry-only Hough
fallback for faint straight organs that do not survive connected-component
thresholding.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine9 as refine9  # installs the accepted mask refinements
from measure_guides_v3_core import associate_organ

_BASE_EXTERNAL = refine.external_candidates
_SCALE = 0.5


def _line_response(image: np.ndarray, excluded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-resolution grayscale and edge image for faint linear objects."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * _SCALE)), max(1, int(h * _SCALE))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(24, 24)).apply(gray)
    background = cv2.GaussianBlur(clahe, (0, 0), 5.0)
    dark = cv2.subtract(background, clahe)
    dark = cv2.GaussianBlur(dark, (3, 3), 0)
    edges = cv2.Canny(dark, 2, 8, L2gradient=True)
    small_excluded = cv2.resize((excluded > 0).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    edges[small_excluded > 0] = 0
    return gray, edges


def _segment_mask(shape: tuple[int, int], segment: tuple[float, float, float, float], thickness_px: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = segment
    cv2.line(mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1, thickness_px, cv2.LINE_AA)
    return (mask > 0).astype(np.uint8)


def hough_candidates(image, corolla_union, corollas, top, channels):
    excluded = cv2.dilate((corolla_union > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    excluded[:top] = 1
    _, edges = _line_response(image, excluded)
    small_mm_px = float(base.MM_PX) / _SCALE
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(8, int(round(2.5 / small_mm_px))),
        minLineLength=max(8, int(round(4.0 / small_mm_px))),
        maxLineGap=max(4, int(round(2.2 / small_mm_px))),
    )
    if lines is None:
        return []

    by_flower: dict[int, list[tuple[float, dict]]] = {}
    full_shape = image.shape[:2]
    for packed in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(v) / _SCALE for v in packed)
        length_mm = math.hypot(x2 - x1, y2 - y1) * float(base.MM_PX)
        if not 4.0 <= length_mm <= 38.0:
            continue
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
        axis_score = max(abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle))))
        if axis_score < 0.72:
            continue

        thickness = max(3, int(round(0.75 / float(base.MM_PX))))
        mask = _segment_mask(full_shape, (x1, y1, x2, y2), thickness)
        if int(mask.sum()) == 0:
            continue
        features = refine._global_features(mask, channels, "hough_line")
        if not features:
            continue
        features["cx"] = round((x1 + x2) / 2.0, 2)
        features["cy"] = round((y1 + y2) / 2.0, 2)
        features["rect_length_mm"] = round(length_mm, 3)
        features["median_width_mm"] = round(thickness * float(base.MM_PX), 3)
        features["aspect"] = round(length_mm / max(thickness * float(base.MM_PX), 0.1), 3)
        features["endpoints"] = np.array([[x1, y1], [x2, y2]], dtype=float)

        association = associate_organ(features, corollas)
        distance = float(association["association_distance_mm"])
        if distance > 22.0:
            continue
        cid = int(association["nearest_corolla"])
        if not 0 < cid <= len(corollas):
            continue
        flower = corollas[cid - 1]
        fc_x = float(flower.get("cx", 0.0))
        fc_y = float(flower.get("cy", 0.0))
        dx_mm = (features["cx"] - fc_x) * float(base.MM_PX)
        dy_mm = abs(features["cy"] - fc_y) * float(base.MM_PX)
        # Detached organs in these preparations are normally beside the flower;
        # allow a small left overlap for crowded sheets and horizontal placements.
        if not (-5.0 <= dx_mm <= 30.0 and dy_mm <= 20.0):
            continue

        # Prefer a line to the right, similar vertical level, plausible length,
        # and strong endpoint association. No shimask coordinates are used here.
        score = 0.35
        score += max(0.0, 1.0 - distance / 22.0) * 0.25
        score += max(0.0, 1.0 - dy_mm / 20.0) * 0.16
        score += max(0.0, 1.0 - abs(length_mm - 15.0) / 20.0) * 0.14
        score += 0.08 if dx_mm >= 0.0 else 0.0
        score += axis_score * 0.08
        features.update(
            association,
            candidate_source="hough_line",
            organ_type_auto="style_or_pistil_candidate",
            organ_confidence=round(min(score, 0.94), 3),
            organ_qc_reasons="direct_line_segment_near_corolla",
            association_qc_required=1,
        )
        by_flower.setdefault(cid, []).append((score, features))

    selected: list[dict] = []
    for candidates in by_flower.values():
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected.append(candidates[0][1])
    return selected


def external_candidates(image, corolla_union, corollas, top, channels):
    colour_rows = _BASE_EXTERNAL(image, corolla_union, corollas, top, channels)
    line_rows = hough_candidates(image, corolla_union, corollas, top, channels)
    return refine._deduplicate([*colour_rows, *line_rows])


refine.external_candidates = external_candidates


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
