# -*- coding: utf-8 -*-
"""Exclude obvious narrow non-corolla fragments before v3 measurement.

v2 already defines a conservative fragment rule: area below 150 mm2 *and* width
below 15 mm.  The pipeline previously kept such objects as low-confidence flower
rows, forcing pointless review.  This entrypoint applies the established rule at
the component stage, before IDs and trait rows are written.
"""
from __future__ import annotations

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_v3_refine as refine
import measure_guides_v3_refine8 as refine8  # noqa: F401  (installs waist splitter)

_ORIGINAL_COROLLAS = v2.corollas


def corollas_without_fragments(filled, auto_split=True):
    components = _ORIGINAL_COROLLAS(filled, auto_split)
    kept = []
    for component in components:
        measured = component.get("m") or v2.metrics(component["mask"])
        if measured is None:
            continue
        area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
        width_mm = float(measured["width_px"]) * float(base.MM_PX)
        if v2.is_fragment(area_mm2, width_mm):
            continue
        kept.append(component)
    return kept


v2.corollas = corollas_without_fragments


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
