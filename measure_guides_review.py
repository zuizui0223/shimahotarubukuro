# -*- coding: utf-8 -*-
"""Reviewed overrides for the v2 scanner pipeline.

This module keeps ``measure_guides_v2.py`` reproducible and applies refinements
found during sheet-by-sheet visual QC:

* large area alone never triggers a two-corolla split;
* thin, long spurs attached to an otherwise thick corolla are pruned conservatively;
* reproductive-organ candidates must be at least 8 mm long, while thresholds are
  relaxed enough to retain pale detached stamens/styles.

Run it with the same CLI as measure_guides_v2.py.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2

_ORIGINAL_FOREGROUND = v2.foreground_v2
_ORIGINAL_TRY_SPLIT = v2.try_split

MIN_ORGAN_LENGTH_MM = 8.0
SPUR_CORE_RADIUS_PX = 6
SPUR_REACH_PX = 13
SPUR_MIN_RETAINED_FRAC = 0.90


def prune_thin_spurs(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    source = (mask > 0).astype(np.uint8)
    if source.sum() == 0:
        return source, False

    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    core = (distance >= SPUR_CORE_RADIUS_PX).astype(np.uint8)
    if core.sum() == 0:
        return source, False

    reach = cv2.dilate(
        core,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (SPUR_REACH_PX * 2 + 1, SPUR_REACH_PX * 2 + 1),
        ),
    )
    candidate = source & reach
    retained = float(candidate.sum()) / float(source.sum())
    if retained < SPUR_MIN_RETAINED_FRAC:
        return source, False

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n <= 1:
        return source, False
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = (labels == largest).astype(np.uint8)
    return cleaned, bool(cleaned.sum() < source.sum())


def foreground_reviewed(img: np.ndarray, top: int):
    filled, a, b = _ORIGINAL_FOREGROUND(img, top)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    cleaned = np.zeros_like(filled)
    for index in range(1, n):
        component = (labels == index).astype(np.uint8)
        if stats[index, cv2.CC_STAT_AREA] * base.MM2_PX < 20:
            continue
        pruned, _ = prune_thin_spurs(component)
        cleaned[pruned > 0] = 255
    cleaned[:top] = 0
    return cleaned, a, b


def try_split_reviewed(mask: np.ndarray):
    measured = v2.metrics(mask)
    length_mm = measured["length_px"] * base.MM_PX
    if length_mm <= v2.SPLIT_LEN_MM:
        return [mask], "not_triggered"
    return _ORIGINAL_TRY_SPLIT(mask)


def organs_reviewed(img: np.ndarray, corolla_mask: np.ndarray, top: int):
    lightness, a_star, b_star = base.channels(img)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)

    coloured_or_warm = (chroma > 1.2) | (a_star > 1.2) | (b_star > 1.0)
    dark_neutral_writing = (lightness < 145) & (chroma < 12)
    candidate = ((lightness < 252) & coloured_or_warm & ~dark_neutral_writing).astype(np.uint8)
    candidate[:top] = 0

    exclusion = cv2.dilate(
        corolla_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    candidate[exclusion > 0] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    merged = np.zeros_like(candidate)
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 19)),
        np.eye(15, dtype=np.uint8),
        np.fliplr(np.eye(15, dtype=np.uint8)),
    ):
        merged |= cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    output = []
    for index in range(1, n):
        area_mm2 = stats[index, cv2.CC_STAT_AREA] * base.MM2_PX
        if not 0.8 <= area_mm2 <= 120:
            continue

        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), (rect_width, rect_height), angle = cv2.minAreaRect(contour)
        length_px = max(rect_width, rect_height)
        width_px = min(rect_width, rect_height)
        if width_px < 1:
            continue

        length_mm = length_px * base.MM_PX
        width_mm = width_px * base.MM_PX
        aspect = length_px / width_px
        pixels = component > 0
        mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0

        if not (
            MIN_ORGAN_LENGTH_MM <= length_mm <= 45
            and 0.15 <= width_mm <= 4.5
            and aspect >= 3.5
            and mean_chroma >= 1.0
        ):
            continue

        output.append(
            dict(
                cx=round(cx, 2),
                cy=round(cy, 2),
                length_mm=round(length_mm, 2),
                width_mm=round(width_mm, 2),
                aspect=round(aspect, 2),
                angle_deg=round(angle, 2),
                organ_type_auto="unclassified_reproductive_organ",
                organ_type_FILL="",
                exclude_FILL="",
            )
        )
    return sorted(output, key=lambda row: (row["cy"], row["cx"]))


def install_reviewed_overrides() -> None:
    v2.foreground_v2 = foreground_reviewed
    v2.try_split = try_split_reviewed
    v2.organs = organs_reviewed


def main() -> None:
    install_reviewed_overrides()
    v2.main()


if __name__ == "__main__":
    main()
