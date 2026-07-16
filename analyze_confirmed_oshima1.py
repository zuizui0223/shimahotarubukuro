#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the one-sheet confirmed analysis with an oshima1-specific mask sanity gate."""
from __future__ import annotations

import cv2

import analyze_confirmed_single_sheet as single
import measure_guides as base

_original_components = single._confirmed_components


def _oshima1_components(red, raw_shape):
    components = _original_components(red, raw_shape)
    plausible = []
    for component in components:
        mask = component["mask"].astype("uint8")
        area_mm2 = float(mask.sum()) * float(base.MM2_PX)
        metrics = component["m"]
        length_mm = float(metrics["length_px"]) * float(base.MM_PX)
        width_mm = float(metrics["width_px"]) * float(base.MM_PX)
        if not (base.AREA_MM2_MIN <= area_mm2 <= base.AREA_MM2_MAX):
            continue
        if not (15.0 <= length_mm <= 65.0 and 12.0 <= width_mm <= 60.0):
            continue
        plausible.append(component)

    # oshima1 has four reviewed flowers.  Keep the four largest biologically
    # plausible closed red outlines; small red annotation loops/text are excluded.
    plausible.sort(key=lambda row: int(row["mask"].sum()), reverse=True)
    selected = plausible[:4]
    if len(selected) != 4:
        raise RuntimeError(
            f"oshima1 expected 4 plausible confirmed corollas; "
            f"raw_closed={len(components)} plausible={len(plausible)}"
        )
    selected.sort(key=lambda row: (int(float(row["cy"])) // 170, float(row["cx"])))
    for source_id, component in enumerate(selected, start=1):
        component["source_component_id"] = source_id
    return selected


single._confirmed_components = _oshima1_components

if __name__ == "__main__":
    single.main()
