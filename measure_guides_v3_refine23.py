# -*- coding: utf-8 -*-
"""Conservative ridge-vesselness recovery layered on accepted refine19.

Existing organ detections are preserved. Ridge candidates are added only when they
are elongated, locally dark/yellow, inside the organ search band, associated with
a corolla, and not duplicates of an existing organ instance. shimask is never read.

Unlike earlier refinement modules, importing this module does not mutate the shared
pipeline. ``install()`` applies the detector only for an explicit refine23 run,
which keeps the regression-test modules isolated from one another.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.filters import sato

import measure_guides as base
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine19  # noqa: F401 installs accepted boundary + trait chain

MM_PX = float(base.MM_PX)
_ORIGINAL_DETECT = refine13.detect_organs

RIDGE_SIGMAS = (2, 3, 4, 5, 6)
RIDGE_Q = 0.992
RIDGE_MIN_MM = 5.0
RIDGE_MAX_MM = 40.0
RIDGE_MAX_WIDTH_MM = 4.2
RIDGE_MIN_ASPECT = 3.2
RIDGE_ASSOC_MAX_MM = 30.0
RIDGE_DUPLICATE_MM = 6.0


def _candidate_components(union: np.ndarray, top: int, channels) -> tuple[np.ndarray, np.ndarray]:
    light, _a, b, _local_a, _local_b, local_dark, _chroma = channels
    band = refine13._search_band(union, top)
    ys, xs = np.where(band > 0)
    response = np.zeros_like(light, np.float32)
    if len(xs) < 50:
        return np.zeros_like(union, np.uint8), response

    # Sato is the expensive operation. Restrict it to the bounding box of the
    # biologically plausible search band and paste the response back into the
    # full-resolution coordinate system used by the rest of the pipeline.
    pad = 8
    x0, x1 = max(0, int(xs.min()) - pad), min(light.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(light.shape[0], int(ys.max()) + pad + 1)
    darkness = np.clip((255.0 - light[y0:y1, x0:x1]) / 255.0, 0.0, 1.0)
    roi_response = sato(darkness, sigmas=RIDGE_SIGMAS, black_ridges=False)
    response[y0:y1, x0:x1] = roi_response.astype(np.float32)

    values = response[band > 0]
    if values.size < 50 or float(values.max()) <= 0:
        return np.zeros_like(union, np.uint8), response
    threshold = float(np.quantile(values, RIDGE_Q))
    candidate = (
        (band > 0)
        & (response >= threshold)
        & (local_dark > 3.5)
        & (b > 1.5)
        & (light > 105.0)
        & (light < 252.0)
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return candidate, response


def _nearest_corolla(px: float, py: float, contours: list[np.ndarray]) -> tuple[int, float]:
    nearest, dmin = 0, 1e9
    for cid, contour in enumerate(contours, 1):
        if contour.size == 0:
            continue
        distance = -cv2.pointPolygonTest(contour, (px, py), True) * MM_PX
        if distance < dmin:
            nearest, dmin = cid, distance
    return nearest, dmin


def _ridge_rows(union: np.ndarray, corollas: list[dict], top: int, channels) -> list[dict]:
    candidate, response = _candidate_components(union, top, channels)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    contours: list[np.ndarray] = []
    for corolla in corollas:
        cs, _ = cv2.findContours(
            np.asarray(corolla["mask"], np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours.append(max(cs, key=cv2.contourArea) if cs else np.zeros((0, 1, 2), np.int32))

    output: list[dict] = []
    for label in range(1, n):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        upper = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        crop = (labels[upper:upper + height, left:left + width] == label).astype(np.uint8)
        cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        length_px, width_px = max(rw, rh), min(rw, rh)
        length_mm, width_mm = length_px * MM_PX, width_px * MM_PX
        aspect = length_px / max(width_px, 1e-6)
        if not (RIDGE_MIN_MM <= length_mm <= RIDGE_MAX_MM):
            continue
        if width_mm > RIDGE_MAX_WIDTH_MM or aspect < RIDGE_MIN_ASPECT:
            continue
        ys, xs = np.where(labels == label)
        points = refine13._axis_points(ys, xs)
        if not points:
            continue
        centre_x = float(np.mean([p[0] for p in points]))
        centre_y = float(np.mean([p[1] for p in points]))
        nearest, distance = _nearest_corolla(centre_x, centre_y, contours)
        if nearest == 0 or distance > RIDGE_ASSOC_MAX_MM:
            continue
        ridge_score = float(np.median(response[ys, xs]))
        for sample_index, (px, py) in enumerate(points, 1):
            output.append({
                "cx": round(px, 2),
                "cy": round(py, 2),
                "nearest_corolla": nearest,
                "organ_len_mm": round(length_mm, 2),
                "organ_width_mm": round(width_mm, 2),
                "organ_aspect": round(aspect, 2),
                "association_distance_mm": round(distance, 2),
                "candidate_source": "sato_ridge_recovery",
                "organ_type_auto": "stamen_or_style_candidate",
                "organ_type_reason": "thin_high_aspect_ridge",
                "organ_identity_status": "candidate_requires_validation",
                "organ_sample_index": sample_index,
                "organ_sample_count": len(points),
                "measurement_unit": "organ_component_sample",
                "ridge_score": round(ridge_score, 6),
            })
    return output


def detect_organs(union: np.ndarray, corollas: list[dict], top: int, channels) -> list[dict]:
    existing = [dict(row) for row in _ORIGINAL_DETECT(union, corollas, top, channels)]
    existing_points = [
        (float(row["cx"]), float(row["cy"]))
        for row in existing
        if row.get("cx") not in (None, "") and row.get("cy") not in (None, "")
    ]
    next_instance = max([int(row.get("organ_instance_id", 0) or 0) for row in existing] + [0]) + 1
    accepted: list[dict] = []
    grouped: dict[tuple, list[dict]] = {}
    for row in _ridge_rows(union, corollas, top, channels):
        key = (
            int(row["nearest_corolla"]),
            row["organ_len_mm"],
            row["organ_width_mm"],
            row["ridge_score"],
        )
        grouped.setdefault(key, []).append(row)

    for records in grouped.values():
        duplicate = any(
            any(
                math.hypot(float(record["cx"]) - ex, float(record["cy"]) - ey) * MM_PX
                <= RIDGE_DUPLICATE_MM
                for ex, ey in existing_points
            )
            for record in records
        )
        if duplicate:
            continue
        for row in records:
            row["organ_instance_id"] = next_instance
            accepted.append(row)
        next_instance += 1
    return existing + accepted


def install() -> None:
    """Install refine23 for an explicit batch run, without import-time leakage."""
    refine13.detect_organs = detect_organs


def main() -> None:
    install()
    refine13.main()


if __name__ == "__main__":
    main()
