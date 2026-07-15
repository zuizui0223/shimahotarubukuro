# -*- coding: utf-8 -*-
"""Accepted entry point: refine17 (0.7 mm edge erosion + refine14 traits) plus
lobe/tube area partitioning and corolla perimeter.

Rationale: after correcting the shimask evaluator to score only the hand-drawn
review strokes (not the flower's natural nectar-guide colour), the true corolla
boundary recall is ~0.76 (not ~0.55). A fresh full-20-sheet re-optimisation of the
edge treatment against the corrected ground truth confirmed that a smooth fixed
0.7 mm erosion is optimal (boundary F1 0.80); pixel-level tissue snapping and
per-corolla adaptive erosion both scored lower because the reviewed boundary is a
smooth stroke. So the boundary is kept as refine17 and this stage only adds
morphometric traits used by Campanula floral studies.

Adds per corolla (flattened-scan proxies, status-flagged):
- lobe_area_mm2 / tube_area_mm2 / lobe_area_frac  (split at the lobe sinus)
- corolla_perimeter_mm

shimask remains strictly evaluation-only.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine17 as refine17  # sets 0.7 mm erosion + installs refine14

MM_PX = float(base.MM_PX)
MM2_PX = float(base.MM2_PX)

_PREV_RECOMPUTE = refine._recompute_traits  # refine14's trait-enhanced recompute


def corolla_area_traits(mask: np.ndarray) -> dict:
    """Partition the base-to-tip oriented corolla into lobe and tube areas."""
    q = np.asarray(mask) > 0
    if int(q.sum()) < 20:
        return {
            "lobe_area_mm2": "", "tube_area_mm2": "", "lobe_area_frac": "",
            "corolla_perimeter_mm": "", "area_partition_status": "not_measurable",
        }
    oriented, _ = base.orient_base_tip(q, np.zeros_like(q, dtype=bool))
    height = int(oriented.shape[0])
    widths = oriented.sum(axis=1).astype(float)
    geometry = base.geometry(oriented)
    tube_depth_px = float(np.clip(geometry.get("tube_depth", 0.0), 0.0, max(height - 1, 0)))
    throat_row = int(np.clip(round((height - 1) - tube_depth_px), 0, height - 1))

    tube_area_px = float(widths[throat_row:].sum())   # base/tube end (below the sinus)
    lobe_area_px = float(widths[:throat_row].sum())    # lobed end (above the sinus)
    total_px = float(widths.sum())

    contours, _ = cv2.findContours(q.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter_px = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0

    reliable = throat_row > 0 and lobe_area_px > 0 and tube_area_px > 0
    return {
        "lobe_area_mm2": round(lobe_area_px * MM2_PX, 1) if reliable else "",
        "tube_area_mm2": round(tube_area_px * MM2_PX, 1) if reliable else "",
        "lobe_area_frac": round(lobe_area_px / total_px, 3) if reliable and total_px else "",
        "corolla_perimeter_mm": round(perimeter_px * MM_PX, 1) if perimeter_px else "",
        "area_partition_status": "automatic_provisional" if reliable else "review_required",
    }


def _recompute_with_areas(image: np.ndarray, mask: np.ndarray, row: dict, channels) -> dict:
    updated = _PREV_RECOMPUTE(image, mask, row, channels)
    updated.update(corolla_area_traits(mask))
    return updated


refine._recompute_traits = _recompute_with_areas


def main() -> None:
    refine17.main()


if __name__ == "__main__":
    main()
