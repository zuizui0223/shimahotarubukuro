# -*- coding: utf-8 -*-
"""Multiscale refinement for automatic-first floral extraction v3.

This layer keeps the existing v3/refine outputs but improves two residual failure
modes seen on oshima10~13:

* broad styles/ovaries or paper folds that survive the first 2.6-mm opening;
* too many external reproductive-organ candidates around one corolla.

The refinement uses two opening scales and removes a lost component only when its
geometry, attachment neck, or colour is incompatible with an ordinary corolla
lobe. Opening width remains deferred and maximum width remains provisional.
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine

_SMALL_OPENING_MM = 2.6
_LARGE_OPENING_MM = 4.0
_CURRENT_CHANNELS: tuple[np.ndarray, ...] | None = None

_ORIGINAL_PROCESS_SHEET = refine.process_sheet
_ORIGINAL_EXTERNAL_CANDIDATES = refine.external_candidates


def _odd_kernel_diameter(mm: float) -> int:
    diameter = max(9, int(round(float(mm) / float(base.MM_PX))))
    if diameter % 2 == 0:
        diameter += 1
    return diameter


def _largest(mask: np.ndarray) -> np.ndarray:
    q = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(q, 8)
    if n <= 1:
        return q
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def _component_geometry(component: np.ndarray) -> tuple[float, float, float]:
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0, 0.0, 0.0
    rect_width, rect_height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
    length = max(float(rect_width), float(rect_height)) * float(base.MM_PX)
    width = min(float(rect_width), float(rect_height)) * float(base.MM_PX)
    aspect = length / max(width, 1e-6)
    return length, width, aspect


def _lost_components(mask: np.ndarray, opening_mm: float) -> list[dict]:
    q = (np.asarray(mask) > 0).astype(np.uint8)
    diameter = _odd_kernel_diameter(opening_mm)
    opened = cv2.morphologyEx(
        q,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    )
    opened = _largest(opened)
    if not opened.any():
        return []

    difference = q & (1 - opened)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(difference, 8)
    touching = cv2.dilate(opened, np.ones((3, 3), np.uint8))
    output: list[dict] = []
    for label in range(1, n):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        area_mm2 = area_px * float(base.MM2_PX)
        if area_mm2 < 0.35:
            continue
        component = (labels == label).astype(np.uint8)
        length_mm, width_mm, aspect = _component_geometry(component)
        if length_mm <= 0:
            continue
        output.append(
            {
                "mask": component,
                "area_px": area_px,
                "area_mm2": area_mm2,
                "length_mm": length_mm,
                "width_mm": width_mm,
                "aspect": aspect,
                "neck_pixels": int(np.count_nonzero(component & touching)),
                "opening_mm": opening_mm,
            }
        )
    return output


def _colour_summary(component: np.ndarray) -> tuple[float, float]:
    if _CURRENT_CHANNELS is None:
        return 99.0, 99.0
    _, _, b, _, _, _, chroma = _CURRENT_CHANNELS
    selected = component > 0
    if not selected.any():
        return 0.0, 0.0
    return float(np.median(chroma[selected])), float(np.mean(b[selected]))


def _should_remove(record: dict) -> bool:
    area_mm2 = float(record["area_mm2"])
    length_mm = float(record["length_mm"])
    width_mm = float(record["width_mm"])
    aspect = float(record["aspect"])
    neck_pixels = int(record["neck_pixels"])
    median_chroma, mean_b = _colour_summary(record["mask"])

    # Paper creases are pale/neutral even when an aggressive close joins them to
    # a faded corolla. Keep the area ceiling generous because C17/C18 contain
    # long bottom folds.
    paper_like = (
        (median_chroma < 5.6 or mean_b < 4.8)
        and length_mm >= 2.3
        and area_mm2 <= 160.0
    )

    # Classical style/pistil or filament-like appendage.
    elongated = (
        length_mm >= 4.5
        and aspect >= 2.6
        and width_mm <= 6.0
        and area_mm2 <= 85.0
    )

    # C7-like broad style/ovary: not extremely elongated, but connected through
    # a narrow neck and much smaller than the corolla body. Ordinary lobe tips
    # normally have a wider contact or a lower aspect ratio.
    weak_neck = (
        length_mm >= 4.5
        and aspect >= 1.7
        and neck_pixels <= 34
        and area_mm2 <= 40.0
        and width_mm <= 6.5
    )

    # Very long side projections are implausible lobes even when the opening
    # leaves a slightly wider contact band.
    very_long_projection = (
        length_mm >= 8.0
        and aspect >= 1.9
        and neck_pixels <= 52
        and area_mm2 <= 65.0
    )
    return bool(paper_like or elongated or weak_neck or very_long_projection)


def detach_thin_appendages(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Remove multiscale narrow-neck appendages while retaining normal lobes."""
    q = (np.asarray(mask) > 0).astype(np.uint8)
    if not q.any():
        return q, []

    ys, xs = np.where(q > 0)
    margin = _odd_kernel_diameter(_LARGE_OPENING_MM)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(q.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(q.shape[0], int(ys.max()) + margin + 1)
    crop = q[y0:y1, x0:x1]

    candidates: list[dict] = []
    for opening_mm in (_SMALL_OPENING_MM, _LARGE_OPENING_MM):
        for record in _lost_components(crop, opening_mm):
            full = np.zeros_like(q)
            full[y0:y1, x0:x1] = record["mask"]
            record = dict(record)
            record["mask"] = full
            if _should_remove(record):
                candidates.append(record)

    # Merge the same appendage found at both scales. Larger-scale masks are
    # useful because they include the ovary/side blob that a 2.6-mm opening can
    # leave behind.
    candidates.sort(
        key=lambda row: (float(row["opening_mm"]), int(row["area_px"])),
        reverse=True,
    )
    remove_union = np.zeros_like(q)
    removed: list[np.ndarray] = []
    for record in candidates:
        component = record["mask"].astype(np.uint8)
        new_pixels = component & (1 - remove_union)
        if int(new_pixels.sum()) * float(base.MM2_PX) < 0.25:
            continue
        remove_union |= component
        removed.append(component)

    # Safety guard: a refinement must never erase a large fraction of the
    # corolla. If this trips, retain only the original first-stage result.
    if int(remove_union.sum()) > int(q.sum()) * 0.16:
        return refine.detach_thin_appendages(q)

    cleaned = _largest(q & (1 - remove_union))
    return cleaned, removed


def external_candidates(*args, **kwargs) -> list[dict]:
    """Keep at most one strong external candidate for each corolla."""
    candidates = _ORIGINAL_EXTERNAL_CANDIDATES(*args, **kwargs)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        organ_confidence = float(row.get("organ_confidence", 0.0))
        association_confidence = float(row.get("association_confidence", 0.0))
        chroma = float(row.get("median_chroma", 0.0))
        length_mm = float(row.get("rect_length_mm", 0.0))
        aspect = float(row.get("aspect", 0.0))
        combined = organ_confidence * association_confidence
        if chroma < 5.8:
            continue
        if length_mm < 4.5 or length_mm > 38.0 or aspect < 2.25:
            continue
        if combined < 0.46:
            continue
        grouped[int(row["nearest_corolla"])].append(row)

    selected: list[dict] = []
    for records in grouped.values():
        records.sort(
            key=lambda row: (
                float(row.get("organ_confidence", 0.0))
                * float(row.get("association_confidence", 0.0)),
                float(row.get("median_chroma", 0.0)),
            ),
            reverse=True,
        )
        selected.append(records[0])
    return selected


def process_sheet(path: str, folder: str, out_dir: str):
    """Set per-sheet colour context, then run the existing refined pipeline."""
    global _CURRENT_CHANNELS
    image = base.load_bgr(path)
    _CURRENT_CHANNELS = refine._lab_channels(image)
    try:
        return _ORIGINAL_PROCESS_SHEET(path, folder, out_dir)
    finally:
        _CURRENT_CHANNELS = None


# The original process_sheet resolves these functions from its module globals at
# call time, so replacing them keeps the established CSV/overlay writer intact.
refine.detach_thin_appendages = detach_thin_appendages
refine.external_candidates = external_candidates
refine.process_sheet = process_sheet


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
