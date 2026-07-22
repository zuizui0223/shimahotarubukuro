#!/usr/bin/env python3
"""Area-based nectar-guide measurement per reviewed corolla.

Purple/magenta guide pixels are detected inside each human-reviewed corolla ROI on
the ruler-calibrated raw scan. The publication pipeline retains only guide coverage
(the guide area as a percentage of the corolla ROI): discrete spot counts and their
derived density are intentionally not measured because adjacent spots merge and
single spots split with threshold, resolution and fading.

Writes ``results_shimask_all/guide_traits.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from run_all_shimask_confirmed import find_raw


def guide_mask(raw: np.ndarray, piece: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    """Return (guide mask, bbox origin, reviewed ROI area in pixels)."""
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    roi = piece[y0:y1, x0:x1] > 0
    b, g, r = cv2.split(sub.astype(int))
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    guide = (((r - g) > 18) & ((b - g) > -10) &
             (hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 205) & roi)
    guide = cv2.morphologyEx(
        guide.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    return guide, (x0, y0), int(roi.sum())


def main() -> None:
    labels = sorted(
        p for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows = []
    for lp in labels:
        sheet = lp.stem
        _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
        raw = base.load_bgr(str(raw_path))
        ann = base.load_bgr(str(lp))
        comps = shimask_input.red_corolla_components(
            raw, ann, strokes=shimask_input.stroke_masks(raw, ann)
        )
        for cid, comp in enumerate(comps, 1):
            pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
            suffixes = [""] if len(pieces) == 1 else ["a", "b"]
            for suffix, piece in zip(suffixes, pieces):
                guide, _origin, roi_px = guide_mask(raw, piece)
                n_px = int(guide.sum())
                rows.append({
                    "sheet": sheet,
                    "corolla_id": f"{cid}{suffix}",
                    "guide_coverage_pct": round(100.0 * n_px / roi_px, 2) if roi_px else 0.0,
                })
        print(f"[{sheet}] {sum(1 for r in rows if r['sheet'] == sheet)} corollas", flush=True)

    out = Path("results_shimask_all/guide_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; coverage only)")


if __name__ == "__main__":
    main()
