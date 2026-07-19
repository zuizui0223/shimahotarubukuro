#!/usr/bin/env python3
"""Combine iPhone-mask and hand-mask corolla traits into one final table.

The iPhone-registered silhouette (results_shimask_all/iphone_traits.csv) is used as
the primary ROI wherever it matched a corolla; the reviewed hand-mask measurement
(results_shimask_all/medial_traits.csv) is the fallback for corollas without an
iPhone match and for split merged pairs (which the per-corolla iPhone masks handle
as separate objects, not the merged component). Fold state and nectar-guide columns
always come from the reviewed hand table.

Writes results_shimask_all/corolla_traits_final.csv with a roi_source column and
the iPhone match IoU, plus a low_iou_match QC flag where the registration was weak.
"""
from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results_shimask_all")
LOW_IOU = 0.75


def main() -> None:
    hand = list(csv.DictReader((RESULTS / "medial_traits.csv").open(encoding="utf-8-sig")))
    iphone = {(r["sheet"], r["corolla_id"]): r
              for r in csv.DictReader((RESULTS / "iphone_traits.csv").open(encoding="utf-8-sig"))}
    guide = {(r["sheet"], r["corolla_id"]): r
             for r in csv.DictReader((RESULTS / "guide_traits.csv").open(encoding="utf-8-sig"))}

    rows = []
    n_iphone = 0
    for h in hand:
        key = (h["sheet"], h["corolla_id"])
        factor = 1.0 if h["fold_state"] == "opened_full" else 2.0
        ip = iphone.get(key)
        qc = [f for f in h["qc_flag"].split("|") if f] if h["qc_flag"] else []
        if ip is not None:
            n_iphone += 1
            source, iou = "iphone", float(ip["match_iou"])
            length = float(ip["corolla_length_mm"])
            width = float(ip["corolla_width_obs_mm"])
            area = float(ip["corolla_area_obs_mm2"])
            # The hand ROI issues are resolved by the clean iPhone silhouette.
            qc = [f for f in qc if f not in
                  ("roi_area_reconstructed", "roi_open_outline", "roi_trimmed_to_petal", "irregular_roi")]
            if iou < LOW_IOU:
                qc.append("low_iou_match")
        else:
            source, iou = "hand", ""
            length = float(h["corolla_length_mm"])
            width = float(h["corolla_width_obs_mm"])
            area = float(h["corolla_area_obs_mm2"])
        rows.append({
            "sheet": h["sheet"], "corolla_id": h["corolla_id"], "fold_state": h["fold_state"],
            "corolla_length_mm": round(length, 2), "corolla_width_obs_mm": round(width, 2),
            "corolla_area_obs_mm2": round(area, 1),
            "corolla_width_fulleq_mm": round(width * factor, 2),
            "corolla_area_fulleq_mm2": round(area * factor, 1),
            "roi_source": source, "match_iou": iou,
            "nectar_guide_px": h["nectar_guide_px"], "has_nectar_guide": h["has_nectar_guide"],
            "guide_coverage_pct": g["guide_coverage_pct"] if (g := guide.get(key)) else "",
            "n_guide_spots": g["n_guide_spots"] if g else "",
            "guide_density_per_cm2": g["guide_density_per_cm2"] if g else "",
            "qc_flag": "|".join(qc),
        })

    out = RESULTS / "corolla_traits_final.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; {n_iphone} iPhone, {len(rows) - n_iphone} hand)")


if __name__ == "__main__":
    main()
