#!/usr/bin/env python3
"""Combine the reviewed corolla, guide and reproductive-organ measurements.

The iPhone-registered silhouette is the primary size ROI wherever it matched a
corolla. The reviewed hand ROI is the fallback for unmatched corollas and split
merged pairs. Fold state and guide presence come from the reviewed hand table;
area-based guide coverage comes from ``guide_traits.csv``.

Writes ``results_shimask_all/corolla_traits_final.csv`` with one row per corolla.
"""
from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results_shimask_all")
LOW_IOU = 0.75


def main() -> None:
    hand = list(csv.DictReader((RESULTS / "medial_traits.csv").open(encoding="utf-8-sig")))
    iphone = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "iphone_traits.csv").open(encoding="utf-8-sig"))
    }
    guide = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "guide_traits.csv").open(encoding="utf-8-sig"))
    }
    organ = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "organ_traits.csv").open(encoding="utf-8-sig"))
    }

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
            qc = [
                f for f in qc
                if f not in (
                    "roi_area_reconstructed",
                    "roi_open_outline",
                    "roi_trimmed_to_petal",
                    "irregular_roi",
                )
            ]
            if iou < LOW_IOU:
                qc.append("low_iou_match")
        else:
            source, iou = "hand", ""
            length = float(h["corolla_length_mm"])
            width = float(h["corolla_width_obs_mm"])
            area = float(h["corolla_area_obs_mm2"])

        g = guide.get(key)
        o = organ.get(key)
        rows.append({
            "sheet": h["sheet"],
            "corolla_id": h["corolla_id"],
            "fold_state": h["fold_state"],
            "corolla_length_mm": round(length, 2),
            "corolla_width_obs_mm": round(width, 2),
            "corolla_area_obs_mm2": round(area, 1),
            "corolla_width_fulleq_mm": round(width * factor, 2),
            "corolla_area_fulleq_mm2": round(area * factor, 1),
            "roi_source": source,
            "match_iou": iou,
            "has_nectar_guide": h["has_nectar_guide"],
            "guide_coverage_pct": g["guide_coverage_pct"] if g else "",
            "organ_length_mm": o["organ_length_mm"] if o else "",
            "qc_flag": "|".join(qc),
        })

    out = RESULTS / "corolla_traits_final.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas; {n_iphone} iPhone, {len(rows) - n_iphone} hand)")


if __name__ == "__main__":
    main()
