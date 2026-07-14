# -*- coding: utf-8 -*-
"""Recover faint reproductive organs with efficient directional black-hat filtering.

Reviewed shimask images are used only for scoring; runtime inputs remain raw scans.
The expensive directional response is computed at half resolution and mapped back
before the existing full-resolution geometry and association checks.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine10 as refine10
from measure_guides_v3_core import associate_organ

_DETECTION_SCALE = 0.5


def _roi_line_mask(image: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Detect faint lines cheaply at half resolution, return a full-size mask."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(1, int(round(width * _DETECTION_SCALE))), max(1, int(round(height * _DETECTION_SCALE)))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    response = np.zeros_like(gray)
    mm_per_small_px = float(base.MM_PX) / _DETECTION_SCALE
    length = max(9, int(round(4.0 / mm_per_small_px)))
    if length % 2 == 0:
        length += 1
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, length)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3)),
        np.eye(length, dtype=np.uint8),
        np.fliplr(np.eye(length, dtype=np.uint8)),
    ):
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    small_mask = (response >= 2).astype(np.uint8)
    small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask[excluded > 0] = 0
    return mask


def roi_candidates(image, corolla_union, corollas, top, channels):
    excluded = cv2.dilate(
        (corolla_union > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    excluded[:top] = 1
    line_mask = _roi_line_mask(image, excluded)
    rows = refine10._candidate_rows(line_mask, channels)
    by_flower: dict[int, list[dict]] = {}
    for row in rows:
        length = float(row.get("rect_length_mm", 0.0))
        width = float(row.get("median_width_mm", 999.0))
        aspect = float(row.get("aspect", 0.0))
        if not (4.0 <= length <= 38.0 and width <= 5.5 and aspect >= 2.0):
            continue
        association = associate_organ(row, corollas)
        if float(association["association_distance_mm"]) > 20.0:
            continue
        row.update(association)
        cid = int(row["nearest_corolla"])
        if not (0 < cid <= len(corollas)):
            continue
        flower = corollas[cid - 1]
        dx = float(row["cx"]) - float(flower.get("cx", 0.0))
        dy = abs(float(row["cy"]) - float(flower.get("cy", 0.0)))
        search = 24.0 / float(base.MM_PX)
        if not (-0.35 * search <= dx <= search and dy <= 0.8 * search):
            continue
        score = 0.48
        score += min(aspect / 8.0, 1.0) * 0.18
        score += max(0.0, 1.0 - abs(length - 16.0) / 20.0) * 0.16
        score += max(0.0, 1.0 - float(association["association_distance_mm"]) / 20.0) * 0.16
        row.update(
            candidate_source="roi_blackhat",
            organ_type_auto="style_or_pistil_candidate",
            organ_confidence=round(min(score, 0.88), 3),
            organ_qc_reasons="roi_blackhat_faint_linear",
            association_qc_required=1,
        )
        by_flower.setdefault(cid, []).append(row)
    selected = []
    for records in by_flower.values():
        records.sort(
            key=lambda row: float(row["organ_confidence"]) * float(row["association_confidence"]),
            reverse=True,
        )
        selected.append(records[0])
    return selected


def external_candidates(image, corolla_union, corollas, top, channels):
    # Keep the established colour-supported detector, but skip refine10's old
    # full-resolution faint-line pass.  The new half-resolution ROI detector
    # replaces that expensive path.
    colour_rows = refine10._ORIGINAL_EXTERNAL(image, corolla_union, corollas, top, channels)
    local = roi_candidates(image, corolla_union, corollas, top, channels)
    return refine._deduplicate([*colour_rows, *local])


refine.external_candidates = external_candidates


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
