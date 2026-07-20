#!/usr/bin/env python3
"""Nectar-guide traits per corolla: coverage, spot count and spot density.

Detects the purple/magenta nectar-guide spots inside each reviewed corolla ROI on
the raw scan and records how much of the corolla they cover, how many discrete
spots there are, and their density. Guide-less corollas (mostly Shikinejima) come
out at zero, which is itself a trait. Writes results_shimask_all/guide_traits.csv.
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

MM = float(base.MM_PX)
MIN_SPOT_PX = 3  # ignore single-pixel noise when counting discrete spots


def guide_mask(raw: np.ndarray, piece: np.ndarray) -> tuple[np.ndarray, int]:
    """Binary guide mask inside the corolla bbox, and the ROI area in pixels."""
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    roi = piece[y0:y1, x0:x1] > 0
    b, g, r = cv2.split(sub.astype(int))
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    spot = (((r - g) > 18) & ((b - g) > -10) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 205)) & roi
    spot = cv2.morphologyEx(spot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return spot, int(roi.sum())


def main() -> None:
    labels = sorted(p for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows = []
    for lp in labels:
        sheet = lp.stem
        _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
        raw = base.load_bgr(str(raw_path))
        ann = base.load_bgr(str(lp))
        comps = shimask_input.red_corolla_components(raw, ann, strokes=shimask_input.stroke_masks(raw, ann))
        for cid, comp in enumerate(comps, 1):
            pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
            suffixes = [""] if len(pieces) == 1 else ["a", "b"]
            for suffix, piece in zip(suffixes, pieces):
                spot, roi_px = guide_mask(raw, piece)
                n_px = int(spot.sum())
                count, _lab, stats, _c = cv2.connectedComponentsWithStats(spot, 8)
                n_spots = int(sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_SPOT_PX))
                area_mm2 = roi_px * MM * MM
                rows.append({
                    "sheet": sheet, "corolla_id": f"{cid}{suffix}",
                    "guide_px": n_px, "n_guide_spots": n_spots,
                    "guide_coverage_pct": round(100.0 * n_px / roi_px, 2) if roi_px else 0.0,
                    "guide_density_per_cm2": round(n_spots / (area_mm2 / 100.0), 1) if area_mm2 else 0.0,
                })
        print(f"[{sheet}] {sum(1 for r in rows if r['sheet'] == sheet)} corollas", flush=True)

    out = Path("results_shimask_all/guide_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas)")


if __name__ == "__main__":
    main()
