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


# Manual organ->corolla pins for cases the nearest-left heuristic gets wrong - a big
# open flower whose stamen is drawn above/left can be nearer a neighbour and, being
# the longer of two candidates, steal that neighbour's slot. (sheet, corolla_id) ->
# (approx_cx, approx_cy) of the correct stroke; the nearest stroke to that point is
# pinned to that corolla and removed from the automatic pool.
ORGAN_ASSIGN: dict[tuple[str, str], tuple[float, float]] = {
    ("niijiama1-2", "16"): (1219, 2890),  # #168's own vertical stamen, just to its right
    ("niijiama1-2", "17"): (1410, 2821),  # #169's horizontal stamen, drawn above-left
    ("kozu1", "9"): (1562, 2138),         # #208's stamen, drawn just below it (was labelled 209)
    ("kozu1", "10"): (641, 2238),         # #209's own stamen, to its right (was discarded)
}


def build_pieces(comps) -> list[dict]:
    """One entry per measured corolla (split merged pairs into a/b), with centroid."""
    pieces = []
    for cid0, comp in enumerate(comps):
        parts = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        suffixes = [""] if len(parts) == 1 else ["a", "b"]
        for suffix, part in zip(suffixes, parts):
            ys, xs = np.where(part > 0)
            pieces.append({"id": f"{cid0 + 1}{suffix}",
                           "cx": float(xs.mean()), "cy": float(ys.mean()),
                           "h": float(ys.max() - ys.min())})
    return pieces


def associate_organs(sheet: str, pieces: list[dict], greens: list[dict]) -> dict[str, dict]:
    """Map corolla_id -> its green organ stroke.

    Manual pins (ORGAN_ASSIGN) are applied first and lock both the stroke and the
    corolla; the rest use the heuristic: each organ line is drawn beside its corolla
    at the same row, so assign it to the nearest corolla to its left within one row,
    falling back to the plain nearest centroid, keeping the longest stroke per corolla.
    """
    band = float(np.median([p["h"] for p in pieces])) * 0.7 if pieces else 200.0
    best: dict[str, dict] = {}
    used = set()
    pinned_ids = {cid for (sh, cid) in ORGAN_ASSIGN if sh == sheet}
    for (sh, cid), (px, py) in ORGAN_ASSIGN.items():
        if sh != sheet or not greens:
            continue
        gi = min(range(len(greens)),
                 key=lambda i: (greens[i]["cx"] - px) ** 2 + (greens[i]["cy"] - py) ** 2)
        best[cid] = greens[gi]
        used.add(gi)
    for i, g in enumerate(greens):
        if i in used:
            continue
        gx, gy = float(g["cx"]), float(g["cy"])
        avail = [p for p in pieces if p["id"] not in pinned_ids]
        cand = [p for p in avail if p["cx"] < gx and abs(p["cy"] - gy) < band]
        pool = cand if cand else avail
        if not pool:
            continue
        p = min(pool, key=lambda p: (gx - p["cx"]) ** 2 + (gy - p["cy"]) ** 2)
        pid = p["id"]
        if pid not in best or g["endpoint_distance_mm"] > best[pid]["endpoint_distance_mm"]:
            best[pid] = g
    return best


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
        pieces = build_pieces(comps)
        greens = shimask_input.green_organ_rows(raw, ann, strokes=strokes)
        best = associate_organs(sheet, pieces, greens)
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
