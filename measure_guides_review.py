# -*- coding: utf-8 -*-
"""Reviewed overrides for the v2 scanner pipeline.

Only corrections verified from displayed QC overlays are applied automatically.
Unknown sheets remain untouched and continue to require visual review.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

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
# Artifact polygons are removed from both corolla and organ searches.
MANUAL_ARTIFACT_POLYGONS = {
    ("shikinejima", "shikine1~4"): [
        # Thin foreign strip above circled individual ①.
        [(0.155, 0.374), (0.315, 0.374), (0.295, 0.399), (0.165, 0.399)],
    ],
}

# These polygons contain reproductive organs or attached foreign material that the
# broad foreground mask had incorrectly absorbed into a corolla. They are removed
# from the corolla mask only; reviewed organ centre-lines are added separately below.
MANUAL_COROLLA_ORGAN_POLYGONS = {
    ("shikinejima", "shikine1~4"): [
        # Between the two flowers of circled individual ③: a style plus a thin bridge/noise.
        [(0.220, 0.495), (0.307, 0.495), (0.307, 0.602), (0.220, 0.602)],
        # Style/pistil attached to the right edge of machine flower C4.
        [(0.536, 0.515), (0.607, 0.515), (0.607, 0.608), (0.536, 0.608)],
    ],
}

# Reviewed centre-lines for obvious pistil/style specimens. Coordinates are
# normalised and therefore independent of the saved overlay scale.
MANUAL_ORGAN_SEGMENTS = {
    ("shikinejima", "shikine1~4"): [
        # Right of C2.
        {"p1": (0.708, 0.372), "p2": (0.740, 0.447), "width_mm": 1.6, "flower_hint": "C2"},
        # Between C3 and C4; belongs to the left flower in the pair.
        {"p1": (0.258, 0.505), "p2": (0.256, 0.590), "width_mm": 1.8, "flower_hint": "C3"},
        # Previously absorbed into the right side of C4.
        {"p1": (0.565, 0.523), "p2": (0.567, 0.595), "width_mm": 1.8, "flower_hint": "C4"},
        # Right of C5.
        {"p1": (0.952, 0.493), "p2": (0.955, 0.567), "width_mm": 1.8, "flower_hint": "C5"},
        # Right of C6.
        {"p1": (0.834, 0.671), "p2": (0.839, 0.744), "width_mm": 1.8, "flower_hint": "C6"},
    ],
}

_CURRENT_SHEET: tuple[str, str] | None = None


def _fill_polygons(mask: np.ndarray, polygons: list[list[tuple[float, float]]]) -> np.ndarray:
    output = mask.copy()
    height, width = output.shape[:2]
    for polygon in polygons:
        points = np.array(
            [[round(x * width), round(y * height)] for x, y in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(output, [points], 0)
    return output


def apply_current_exclusions(
    mask: np.ndarray,
    *,
    include_organ_cuts: bool = False,
) -> np.ndarray:
    """Apply verified sheet-specific exclusions.

    Artifact polygons are always removed. Organ polygons are removed only from
    corolla masks, so the organ detector does not erase the same structures.
    """
    output = _fill_polygons(mask, MANUAL_ARTIFACT_POLYGONS.get(_CURRENT_SHEET, []))
    if include_organ_cuts:
        output = _fill_polygons(
            output,
            MANUAL_COROLLA_ORGAN_POLYGONS.get(_CURRENT_SHEET, []),
        )
    return output


def foreground_reviewed(img: np.ndarray, top: int):
    filled, a, b = _ORIGINAL_FOREGROUND(img, top)
    filled = apply_current_exclusions(filled, include_organ_cuts=True)
    filled[:top] = 0
    return filled, a, b


def manual_organ_rows(
    image_shape: tuple[int, ...],
    sheet: tuple[str, str] | None = None,
) -> list[dict] | None:
    """Return reviewed organ rows for a sheet, or None when no manual review exists."""
    key = sheet if sheet is not None else _CURRENT_SHEET
    segments = MANUAL_ORGAN_SEGMENTS.get(key)
    if segments is None:
        return None

    height, width = image_shape[:2]
    rows: list[dict] = []
    for index, segment in enumerate(segments, start=1):
        x1 = int(round(segment["p1"][0] * width))
        y1 = int(round(segment["p1"][1] * height))
        x2 = int(round(segment["p2"][0] * width))
        y2 = int(round(segment["p2"][1] * height))
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_mm = length_px * base.MM_PX
        width_mm = float(segment["width_mm"])
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        rows.append(
            dict(
                cx=round((x1 + x2) / 2.0, 2),
                cy=round((y1 + y2) / 2.0, 2),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length_mm=round(length_mm, 2),
                width_mm=round(width_mm, 2),
                aspect=round(length_mm / max(width_mm, 1e-6), 2),
                angle_deg=round(angle, 2),
                score=round(1000.0 - index, 3),
                organ_type_auto="pistil_candidate_reviewed",
                organ_type_FILL="",
                exclude_FILL="",
                detection_source="manual_overlay_review",
                flower_hint=segment["flower_hint"],
            )
        )
    return rows


def _draw_reviewed_organ_lines(
    path: str,
    folder: str,
    out_dir: str,
    organs: list[dict],
) -> None:
    manual = [row for row in organs if row.get("detection_source") == "manual_overlay_review"]
    if not manual:
        return

    island, _ = base.ISLANDS.get(folder, (folder, ""))
    stem = os.path.splitext(os.path.basename(path))[0]
    overlay_path = Path(out_dir) / "overlays" / f"{island}_{stem}.png"
    if not overlay_path.exists():
        return

    data = np.fromfile(str(overlay_path), dtype=np.uint8)
    overlay = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if overlay is None:
        return
    source = base.load_bgr(path)
    scale_x = overlay.shape[1] / source.shape[1]
    scale_y = overlay.shape[0] / source.shape[0]

    for index, row in enumerate(manual, start=1):
        p1 = (int(round(row["x1"] * scale_x)), int(round(row["y1"] * scale_y)))
        p2 = (int(round(row["x2"] * scale_x)), int(round(row["y2"] * scale_y)))
        cv2.line(overlay, p1, p2, (0, 140, 255), 5)
        cv2.putText(
            overlay,
            f"P{index} {row['flower_hint']}",
            (p1[0] + 6, max(18, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 140, 255),
            2,
        )

    cv2.imencode(".png", overlay)[1].tofile(str(overlay_path))


def process_sheet_reviewed(path, folder, out_dir, loc_map=None, auto_split=True):
    global _CURRENT_SHEET
    previous = _CURRENT_SHEET
    _CURRENT_SHEET = (folder.lower(), os.path.splitext(os.path.basename(path))[0].lower())
    try:
        rows, organs = _ORIGINAL_PROCESS_SHEET(path, folder, out_dir, loc_map, auto_split)
        _draw_reviewed_organ_lines(path, folder, out_dir, organs)
        return rows, organs
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
