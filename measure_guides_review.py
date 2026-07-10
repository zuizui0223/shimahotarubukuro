# -*- coding: utf-8 -*-
"""Reviewed overrides for the v2 scanner pipeline.

Only corrections verified from the displayed QC overlay are applied automatically.
Unknown sheets remain untouched and continue to require visual review.
"""
from __future__ import annotations

import math
import os
import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2

_ORIGINAL_FOREGROUND = v2.foreground_v2
_ORIGINAL_TRY_SPLIT = v2.try_split
_ORIGINAL_PROCESS_SHEET = v2.process_sheet

MIN_ORGAN_LENGTH_MM = 8.0
MIN_SPLIT_CENTRE_SEPARATION = 0.90

# Normalised polygons (x/W, y/H), verified against the displayed source overlay.
# The Shikine polygon severs/removes the thin foreign strip above circled individual ①
# without entering the main corolla body.
MANUAL_EXCLUSION_POLYGONS = {
    ("shikinejima", "shikine1~4"): [
        [(0.155, 0.374), (0.315, 0.374), (0.295, 0.399), (0.165, 0.399)],
    ],
}

_CURRENT_SHEET: tuple[str, str] | None = None


def apply_current_exclusions(mask: np.ndarray) -> np.ndarray:
    """Apply verified sheet-specific artifact polygons to any binary mask."""
    output = mask.copy()
    if _CURRENT_SHEET not in MANUAL_EXCLUSION_POLYGONS:
        return output
    height, width = output.shape[:2]
    for polygon in MANUAL_EXCLUSION_POLYGONS[_CURRENT_SHEET]:
        points = np.array(
            [[round(x * width), round(y * height)] for x, y in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(output, [points], 0)
    return output


def foreground_reviewed(img: np.ndarray, top: int):
    filled, a, b = _ORIGINAL_FOREGROUND(img, top)
    filled = apply_current_exclusions(filled)
    filled[:top] = 0
    return filled, a, b


def process_sheet_reviewed(path, folder, out_dir, loc_map=None, auto_split=True):
    global _CURRENT_SHEET
    previous = _CURRENT_SHEET
    _CURRENT_SHEET = (folder.lower(), os.path.splitext(os.path.basename(path))[0].lower())
    try:
        return _ORIGINAL_PROCESS_SHEET(path, folder, out_dir, loc_map, auto_split)
    finally:
        _CURRENT_SHEET = previous


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def try_split_reviewed(mask: np.ndarray):
    """Accept a split only when the two bodies are spatially distinct."""
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
        return [mask], "not_triggered"
    return pieces, status


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _deduplicate_organs(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item.get("score", item["length_mm"]), reverse=True):
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


def install_reviewed_overrides() -> None:
    v2.foreground_v2 = foreground_reviewed
    v2.try_split = try_split_reviewed
    v2.process_sheet = process_sheet_reviewed


def main() -> None:
    install_reviewed_overrides()
    v2.main()


if __name__ == "__main__":
    main()
