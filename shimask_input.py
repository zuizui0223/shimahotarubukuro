# -*- coding: utf-8 -*-
"""PR25 INPUT-ONLY module: turn the human-reviewed shimask strokes into the exact
objects the existing review pipeline already consumes.

This module builds:
  * red corolla outline  -> corolla components with the SAME dict shape that
    ``measure_guides_v2.corollas`` returns (``mask``/``cx``/``cy``/``m``/...),
  * green organ strokes   -> organ rows with the SAME dict shape that
    ``measure_guides_review.manual_organ_rows`` returns.

It performs NO guide extraction and NO trait measurement. Those stay entirely in
the frozen ``measure_guides_review_spots.py`` / ``measure_guides_review_traits.py``.
The driver injects these builders by monkey-patching ``v2.corollas`` and
``reviewed_organs.organs_reviewed`` (see ``qc_single_sheet_shimask.py``); the whole
downstream pipeline is called unchanged. shimask is the analysis INPUT here (PR25),
not a runtime label for the automatic detector (PR24).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2


# --------------------------------------------------------------------------- #
# Extract ONLY the hand-drawn red/green pen strokes.
# The review strokes are near-pure red/green pen; their colour DOMINANCE is far
# higher than the flower's natural purple guide (high R AND high B -> low red
# dominance) or tan tissue, so a strong-dominance rule isolates the strokes
# cleanly without a raw-image difference.
# --------------------------------------------------------------------------- #
def stroke_masks(annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (red, green) drawn-stroke masks at the annotated-preview resolution."""
    if annotated.ndim != 3:
        raise ValueError("annotated must be a BGR colour image")
    b, g, r = cv2.split(annotated.astype(np.int16))
    red = ((r - np.maximum(g, b) >= 50) & (r >= 110)).astype(np.uint8)
    green = ((g - np.maximum(r, b) >= 45) & (g >= 90)).astype(np.uint8)
    return red, green


def _resize_nn(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 1)
    return (mask | (1 - ff)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Red outline -> corolla components (shape-compatible with v2.corollas output)
# --------------------------------------------------------------------------- #
def red_corolla_components(raw: np.ndarray, annotated: np.ndarray) -> list[dict]:
    """Human red outlines -> filled corolla components in v2.corollas' dict shape."""
    red_small, _ = stroke_masks(annotated)
    red = _resize_nn(red_small, raw.shape[:2])
    # A hand-drawn outline can have small gaps (e.g. over the guide-dense mouth).
    # Seal gaps by closing, fill the enclosed interior, then erode by the same
    # amount so the mask boundary returns onto the drawn red line.
    seal = max(3, int(round(3.0 / float(base.MM_PX))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal, seal))
    sealed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    filled = cv2.erode(_fill_holes(sealed), kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    components: list[dict] = []
    source = 0
    for index in range(1, n):
        if float(stats[index, cv2.CC_STAT_AREA]) * float(base.MM2_PX) < float(base.AREA_MM2_MIN):
            continue
        source += 1
        mask = (labels == index).astype(np.uint8)
        measured = v2.metrics(mask)
        if measured is None:
            continue
        moments = cv2.moments(mask)
        if not moments["m00"]:
            continue
        components.append(dict(
            mask=mask.astype(bool),
            source_component_id=source,
            split_piece=1,
            split_status="shimask_red_outline",
            cx=moments["m10"] / moments["m00"],
            cy=moments["m01"] / moments["m00"],
            m=measured,
        ))
    # Reading order, identical convention to v2.corollas so corolla_id numbering matches.
    components.sort(key=lambda r: (int(r["cy"]) // 170, r["cx"]))
    return components


# --------------------------------------------------------------------------- #
# Green strokes -> organ rows (shape-compatible with manual_organ_rows output)
# --------------------------------------------------------------------------- #
def green_organ_rows(raw: np.ndarray, annotated: np.ndarray) -> list[dict]:
    """Human green strokes -> organ rows in the reviewed-organ dict shape."""
    _, green_small = stroke_masks(annotated)
    green = _resize_nn(green_small, raw.shape[:2])
    seal = max(3, int(round(1.0 / float(base.MM_PX))))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal, seal)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(green, 8)
    rows: list[dict] = []
    order = 0
    for index in range(1, n):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        length_span = max(int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
        if area_px * float(base.MM2_PX) < 0.7 or length_span * float(base.MM_PX) < 3.0:
            continue
        crop = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        length_px = max(rw, rh)
        width_px = max(min(rw, rh), 1.0)
        if rw < rh:
            angle += 90.0
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        cx = float(centroids[index][0]); cy = float(centroids[index][1])
        half = length_px / 2.0
        dx = math.cos(math.radians(angle)) * half
        dy = math.sin(math.radians(angle)) * half
        x1, y1 = int(round(cx - dx)), int(round(cy - dy))
        x2, y2 = int(round(cx + dx)), int(round(cy + dy))
        length_mm = length_px * float(base.MM_PX)
        width_mm = width_px * float(base.MM_PX)
        order += 1
        rows.append(dict(
            cx=round(cx, 2), cy=round(cy, 2),
            x1=x1, y1=y1, x2=x2, y2=y2,
            length_mm=round(length_mm, 2),
            width_mm=round(width_mm, 2),
            aspect=round(length_mm / max(width_mm, 1e-6), 2),
            angle_deg=round(angle, 2),
            score=round(1000.0 - order, 3),
            organ_type_auto="reviewed_reproductive_organ",
            organ_type_FILL="",
            exclude_FILL="",
            detection_source="shimask_green_stroke",
            nearest_corolla_hint="",
            association_confirmed=1,
            visibility_note="human-reviewed green stroke",
        ))
    rows.sort(key=lambda r: (r["cy"], r["cx"]))
    return rows


# --------------------------------------------------------------------------- #
# Confirmation image (1): raw + red + green only (no guides)
# --------------------------------------------------------------------------- #
def write_annotation_overlay(raw: np.ndarray, annotated: np.ndarray, out_path: Path) -> None:
    red_small, green_small = stroke_masks(annotated)
    red = _resize_nn(red_small, raw.shape[:2]) > 0
    green = _resize_nn(green_small, raw.shape[:2]) > 0
    overlay = raw.copy()
    overlay[cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 0, 255)
    overlay[cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", overlay)[1].tofile(str(out_path))
