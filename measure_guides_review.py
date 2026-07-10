# -*- coding: utf-8 -*-
"""Reviewed overrides for the v2 scanner pipeline.

The v2 implementation remains reproducible; this module contains refinements
confirmed during sheet-by-sheet visual QC.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2

_ORIGINAL_FOREGROUND = v2.foreground_v2
_ORIGINAL_TRY_SPLIT = v2.try_split

MIN_ORGAN_LENGTH_MM = 8.0
SPUR_CORE_RADIUS_PX = 13
SPUR_REACH_PX = 18
SPUR_MIN_RETAINED_FRAC = 0.90
MIN_SPLIT_CENTRE_SEPARATION = 0.90


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


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def try_split_reviewed(mask: np.ndarray):
    measured = v2.metrics(mask)
    length_mm = measured["length_px"] * base.MM_PX
    if length_mm <= v2.SPLIT_LEN_MM:
        return [mask], "not_triggered"

    pieces, status = _ORIGINAL_TRY_SPLIT(mask)
    if status != "auto_split" or len(pieces) != 2:
        return pieces, status

    separation = math.dist(_centroid(pieces[0]), _centroid(pieces[1]))
    equivalent_diameters = [
        2.0 * math.sqrt(float(piece.sum()) / math.pi)
        for piece in pieces
    ]
    separation_ratio = separation / max(float(np.mean(equivalent_diameters)), 1.0)
    if separation_ratio < MIN_SPLIT_CENTRE_SEPARATION:
        # This is one broad body, not two flowers connected by a narrow bridge.
        return [mask], "not_triggered"
    return pieces, status


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _deduplicate_organs(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item["length_mm"], reverse=True):
        duplicate = False
        for other in kept:
            distance = math.hypot(row["cx"] - other["cx"], row["cy"] - other["cy"])
            radius = max(row["length_mm"], other["length_mm"]) / base.MM_PX * 0.35
            if distance <= radius and _angle_difference(row["angle_deg"], other["angle_deg"]) <= 25:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return sorted(kept, key=lambda row: (row["cy"], row["cx"]))


def organs_reviewed(img: np.ndarray, corolla_mask: np.ndarray, top: int):
    """High-recall candidates; identity remains manual (stamen/style/artifact)."""
    lightness, a_star, b_star = base.channels(img)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)

    coloured_or_warm = (chroma > 0.9) | (a_star > 1.0) | (b_star > 0.8)
    dark_neutral_writing = (lightness < 145) & (chroma < 12)
    candidate = ((lightness < 254) & coloured_or_warm & ~dark_neutral_writing).astype(np.uint8)
    candidate[:top] = 0

    exclusion = cv2.dilate(
        corolla_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    candidate[exclusion > 0] = 0

    output: list[dict] = []

    # 1) Connected elongated objects.
    merged = np.zeros_like(candidate)
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31)),
        np.eye(25, dtype=np.uint8),
        np.fliplr(np.eye(25, dtype=np.uint8)),
    ):
        merged |= cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    for index in range(1, n):
        area_mm2 = stats[index, cv2.CC_STAT_AREA] * base.MM2_PX
        if not 0.8 <= area_mm2 <= 150:
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
        if (
            MIN_ORGAN_LENGTH_MM <= length_mm <= 45
            and 0.15 <= width_mm <= 5.0
            and aspect >= 2.5
            and mean_chroma >= 1.0
        ):
            output.append(dict(
                cx=round(cx, 2), cy=round(cy, 2),
                length_mm=round(length_mm, 2), width_mm=round(width_mm, 2),
                aspect=round(aspect, 2), angle_deg=round(angle, 2),
                organ_type_auto="unclassified_reproductive_organ",
                organ_type_FILL="", exclude_FILL="",
            ))

    # 2) Hough-line recovery for pale or interrupted organs.
    line_mask = cv2.morphologyEx(candidate * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_line_px = int(round(MIN_ORGAN_LENGTH_MM / base.MM_PX))
    lines = cv2.HoughLinesP(
        line_mask,
        1,
        np.pi / 180,
        threshold=24,
        minLineLength=min_line_px,
        maxLineGap=int(round(3.0 / base.MM_PX)),
    )
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length_px = math.hypot(x2 - x1, y2 - y1)
            length_mm = length_px * base.MM_PX
            if not MIN_ORGAN_LENGTH_MM <= length_mm <= 45:
                continue
            tube = np.zeros_like(candidate)
            cv2.line(tube, (x1, y1), (x2, y2), 255, 9)
            pixels = (tube > 0) & (candidate > 0)
            support = int(pixels.sum()) / max(length_px * 9.0, 1.0)
            mean_chroma = float(chroma[pixels].mean()) if pixels.any() else 0.0
            if support < 0.08 or mean_chroma < 1.0:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            width_mm = max(0.15, min(5.0, support * 9.0 * base.MM_PX))
            output.append(dict(
                cx=round((x1 + x2) / 2.0, 2), cy=round((y1 + y2) / 2.0, 2),
                length_mm=round(length_mm, 2), width_mm=round(width_mm, 2),
                aspect=round(length_mm / max(width_mm, 0.01), 2),
                angle_deg=round(angle, 2),
                organ_type_auto="unclassified_reproductive_organ",
                organ_type_FILL="", exclude_FILL="",
            ))

    return _deduplicate_organs(output)


def install_reviewed_overrides() -> None:
    v2.foreground_v2 = foreground_reviewed
    v2.try_split = try_split_reviewed
    v2.organs = organs_reviewed


def main() -> None:
    install_reviewed_overrides()
    v2.main()


if __name__ == "__main__":
    main()
