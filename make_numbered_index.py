#!/usr/bin/env python3
"""Numbered index overlays for manual petal/coordinate alignment.

For each of the 20 sheets this draws, on the raw scan, a clean numbered index of
everything that was analysed: every corolla gets a large circled number C{n} at its
centroid with its ROI outline, and its reproductive-organ stroke (the reviewer's
green line) is highlighted and labelled O{n} with the SAME number, so corolla and
organ correspond at a glance. A light millimetre grid gives a coordinate reference
for aligning petals/coordinates by hand.

Numbering matches corolla_traits_final.csv / organ_traits.csv (reading order, split
pairs as {n}a/{n}b). Writes results_shimask_all/numbered_index/<sheet>.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from make_overlays import organ_for_corolla
from run_all_shimask_confirmed import find_raw

MM = float(base.MM_PX)
OUT = Path("results_shimask_all/numbered_index")

C_ROI = (70, 200, 70)      # green corolla outline
C_NUM = (30, 30, 200)      # red corolla number
C_ORGAN = (210, 90, 30)    # blue organ stroke + number
C_GRID = (210, 210, 210)
C_AXIS = (120, 120, 120)


def final_ids() -> dict:
    p = Path("results_shimask_all/corolla_traits_final.csv")
    return {(r["sheet"], r["corolla_id"]): r for r in csv.DictReader(p.open(encoding="utf-8-sig"))}


def draw_grid(img):
    """Light 10 mm grid with labelled axes (mm) for a coordinate reference."""
    h, w = img.shape[:2]
    step = int(round(10.0 / MM))  # 10 mm
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), C_GRID, 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(x * MM)}", (x + 3, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, C_AXIS, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), C_GRID, 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(y * MM)}", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, C_AXIS, 1, cv2.LINE_AA)


def circled_number(img, cx, cy, text, colour, r=34):
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, colour, 3, cv2.LINE_AA)
    fs = 1.5 if len(text) <= 2 else 1.15
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                fs, colour, 3, cv2.LINE_AA)


def process_sheet(sheet: str, final: dict) -> Path:
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    organ = organ_for_corolla(raw, ann, strokes, comps)
    img = raw.copy()
    draw_grid(img)

    for cid0, comp in enumerate(comps):
        pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            corolla_id = f"{cid0 + 1}{suffix}"
            ys, xs = np.where(piece)
            x0, y0 = int(xs.min()), int(ys.min())
            trimmed = (sheet, corolla_id) in rm.TRIM_TO_PETAL
            mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
            solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
            cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.polylines(img, [c + [x0, y0]], True, C_ROI, 2, cv2.LINE_AA)
            circled_number(img, int(xs.mean()), int(ys.mean()), f"{corolla_id}", C_NUM)

            gr = organ.get(cid0) if suffix in ("", "a") else None
            if gr is not None:
                p1 = (int(gr["x1"]), int(gr["y1"]))
                p2 = (int(gr["x2"]), int(gr["y2"]))
                cv2.line(img, p1, p2, C_ORGAN, 3, cv2.LINE_AA)
                mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                circled_number(img, mx + 46, my, f"O{cid0 + 1}", C_ORGAN, r=30)

    # Legend box.
    lx, ly = 40, raw.shape[0] - 150
    cv2.rectangle(img, (lx - 15, ly - 35), (lx + 430, ly + 95), (255, 255, 255), -1)
    cv2.rectangle(img, (lx - 15, ly - 35), (lx + 430, ly + 95), C_AXIS, 2)
    cv2.putText(img, f"{sheet}  -  numbered index", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (20, 20, 20), 2, cv2.LINE_AA)
    circled_number(img, lx + 20, ly + 45, "n", C_NUM, r=22)
    cv2.putText(img, "corolla (C n) + ROI outline", (lx + 55, ly + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(img, (lx + 5, ly + 82), (lx + 38, ly + 82), C_ORGAN, 3, cv2.LINE_AA)
    cv2.putText(img, "organ (O n), grid = 10 mm", (lx + 55, ly + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sheet}.png"
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    final = final_ids()
    sheets = sorted(p.stem for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for sheet in sheets:
        out = process_sheet(sheet, final)
        print(f"[{sheet}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
