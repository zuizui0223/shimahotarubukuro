#!/usr/bin/env python3
"""Area-based nectar-guide measurement per reviewed corolla.

Only the guide-pixel classifier lives here; corolla ROIs and all size/organ
measurements are unchanged.  Guide colour is learned without manual spot outlines by
a global three-component CIELAB Gaussian mixture (petal / oxidation / purple guide).
A pixel contributes to guide area when P(guide) >= 0.5.  This conservative posterior
majority rule retains faint small purple marks while rejecting ambiguous or oxidised
pixels, and avoids hand-tuned RGB/HSV thresholds.

Writes ``results_shimask_all/guide_traits.csv``.  The fitted component centres are
written to ``guide_gmm_components.csv`` for an auditable Methods record.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import guide_colour_model as gcm
import measure_guides as base
import remeasure_medial as rm
import shimask_input
from run_all_shimask_confirmed import find_raw

# Preserve the pre-existing minimum pixel count used only for the binary
# has_nectar_guide convenience flag. Continuous guide coverage is the analysis trait.
GUIDE_PRESENT_MIN_PX = 150


def guide_mask(raw: np.ndarray, piece: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    """Return (GMM guide mask, bbox origin, reviewed ROI area in pixels)."""
    return gcm.segment_piece(raw, piece)


def main() -> None:
    # Fit once across all 218 reviewed corollas, equally sampling each flower. Later
    # stages load the saved parameters and therefore use exactly the same classifier.
    params = gcm.fit_global_model()
    labels = sorted(
        p for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows = []
    for label_path in labels:
        sheet = label_path.stem
        _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
        raw = base.load_bgr(str(raw_path))
        ann = base.load_bgr(str(label_path))
        comps = shimask_input.red_corolla_components(
            raw, ann, strokes=shimask_input.stroke_masks(raw, ann)
        )
        for cid, comp in enumerate(comps, 1):
            pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
            suffixes = [""] if len(pieces) == 1 else ["a", "b"]
            for suffix, piece in zip(suffixes, pieces):
                guide, _origin, roi_pixels = gcm.segment_piece(raw, piece, params)
                guide_pixels = int(guide.sum())
                rows.append({
                    "sheet": sheet,
                    "corolla_id": f"{cid}{suffix}",
                    "guide_coverage_pct": (
                        round(100.0 * guide_pixels / roi_pixels, 2) if roi_pixels else 0.0
                    ),
                    "has_nectar_guide": int(guide_pixels >= GUIDE_PRESENT_MIN_PX),
                })
        print(
            f"[{sheet}] {sum(1 for row in rows if row['sheet'] == sheet)} corollas",
            flush=True,
        )

    out = Path("results_shimask_all/guide_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; adaptive GMM coverage)")


if __name__ == "__main__":
    main()
