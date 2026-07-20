#!/usr/bin/env python3
"""Reproductive-organ length per corolla from the reviewer's green strokes.

Each corolla has one green line drawn beside it marking the reproductive organ
(stamens / the three-branched pistil); the organ is not classified - the green line
is measured directly. shimask_input.green_organ_rows recovers each green stroke and
its skeleton length; here each stroke is assigned to the nearest corolla and the
length recorded. Writes results_shimask_all/organ_traits.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from run_all_shimask_confirmed import find_raw


def main() -> None:
    labels = sorted(p for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows = []
    for lp in labels:
        sheet = lp.stem
        _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
        raw = base.load_bgr(str(raw_path))
        ann = base.load_bgr(str(lp))
        strokes = shimask_input.stroke_masks(raw, ann)
        comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
        # Two corollas whose hand outlines touch flood-fill into one component; split
        # them so each half carries its OWN organ. Without this, a merged pair keeps
        # only one of its two stamens (and under the wrong a/b id), which is how the
        # organ beside oshima4 #17 and the other split pairs went missing.
        pieces = []  # one entry per measured corolla, matching the a/b ids elsewhere
        for cid0, comp in enumerate(comps):
            parts = rm.split_merged_pair(comp["mask"].astype(np.uint8))
            suffixes = [""] if len(parts) == 1 else ["a", "b"]
            for suffix, part in zip(suffixes, parts):
                ys, xs = np.where(part > 0)
                pieces.append({"id": f"{cid0 + 1}{suffix}",
                               "cx": float(xs.mean()), "cy": float(ys.mean()),
                               "h": float(ys.max() - ys.min())})
        band = float(np.median([p["h"] for p in pieces])) * 0.7 if pieces else 200.0
        greens = shimask_input.green_organ_rows(raw, ann, strokes=strokes)
        # Each organ line is drawn to the right of its corolla at the same row, so
        # assign it to the nearest corolla (or split half) to its left within one row;
        # fall back to the plain nearest centroid if none qualify.
        best: dict[str, dict] = {}
        for g in greens:
            gx, gy = float(g["cx"]), float(g["cy"])
            cand = [p for p in pieces if p["cx"] < gx and abs(p["cy"] - gy) < band]
            pool = cand if cand else pieces
            p = min(pool, key=lambda p: (gx - p["cx"]) ** 2 + (gy - p["cy"]) ** 2)
            pid = p["id"]
            if pid not in best or g["endpoint_distance_mm"] > best[pid]["endpoint_distance_mm"]:
                best[pid] = g
        for p in pieces:
            g = best.get(p["id"])
            # The green line is straight, so its end-to-end distance is the organ
            # length; the skeleton path length over-counts on the thick stroke.
            rows.append({
                "sheet": sheet, "corolla_id": p["id"],
                "organ_length_mm": g["endpoint_distance_mm"] if g else "",
                "organ_skeleton_mm": g["length_mm"] if g else "",
                "organ_width_mm": g["width_mm"] if g else "",
                "has_organ": int(g is not None),
            })
        n = sum(1 for r in rows if r["sheet"] == sheet and r["has_organ"])
        print(f"[{sheet}] {n}/{len(pieces)} corollas with a green organ stroke", flush=True)

    out = Path("results_shimask_all/organ_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} corollas)")


if __name__ == "__main__":
    main()
