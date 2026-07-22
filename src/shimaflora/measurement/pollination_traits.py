#!/usr/bin/env python3
"""Pollination-relevant 2-D floral morphometrics per reviewed corolla.

The retained measurements follow the mechanical-fit traits used for bellflowers by
Nagano et al. (2014, Ecology & Evolution 4:3819; doi:10.1002/ece3.1191), limited to
quantities that flattened, pressed specimens support reproducibly. Absolute corolla
size comes from ``corolla_traits_final.csv``; throat and mouth widths, lobe incision
and shape ratios are measured from the reviewed hand ROI in the scan coordinate
frame.

``throat_width_mm`` and ``mouth_width_mm`` are widths of the equally flattened
specimens, not recovered 3-D entrance diameters. The reviewed green stroke is
reported as reproductive-organ length; it is not reclassified as style or stamen by
image analysis.

Writes ``results_shimask_all/pollination_traits.csv``.
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


def upright(mask_local: np.ndarray, angle_deg: float):
    """Rotate so the resolved corolla-length direction is vertical."""
    h, w = mask_local.shape
    cx, cy = w / 2.0, h / 2.0
    pad = int(max(h, w))
    big = cv2.copyMakeBorder(mask_local, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    M = cv2.getRotationMatrix2D((cx + pad, cy + pad), angle_deg - 90.0, 1.0)
    rot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
    return rot, M


def width_profile(rot: np.ndarray):
    """Per-row ROI width from the mounted tube base to the lobe tips."""
    ys, xs = np.where(rot > 0)
    y0, y1 = ys.min(), ys.max()
    prof = np.zeros(y1 - y0 + 1)
    for k, y in enumerate(range(y0, y1 + 1)):
        row = xs[ys == y]
        prof[k] = (row.max() - row.min() + 1) if row.size else 0.0
    return prof, y0, y1


def lobe_incision_mm(rot: np.ndarray) -> float:
    """Depth of the scalloped distal margin in the flattened display."""
    ys, xs = np.where(rot > 0)
    y0, y1 = ys.min(), ys.max()
    mid = y0 + 0.5 * (y1 - y0)
    edge = {}
    for x, y in zip(xs, ys):
        if y >= mid and y > edge.get(x, -1):
            edge[x] = y
    vals = np.array(list(edge.values()), float)
    if vals.size < 8:
        return 0.0
    return float(np.percentile(vals, 98) - np.percentile(vals, 15)) * MM


def load_final() -> dict:
    path = Path("results_shimask_all/corolla_traits_final.csv")
    return {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader(path.open(encoding="utf-8-sig"))
    }


def load_organ() -> dict:
    path = Path("results_shimask_all/organ_traits.csv")
    return {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader(path.open(encoding="utf-8-sig"))
    }


def measure_sheet(sheet: str, final: dict, organ: dict) -> list[dict]:
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    rows = []
    for cid0, comp in enumerate(comps):
        pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            corolla_id = f"{cid0 + 1}{suffix}"
            trimmed = (sheet, corolla_id) in rm.TRIM_TO_PETAL
            mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
            solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
            measured = rm.medial_axis(mask_local)
            rot, _ = upright(solid, float(measured["angle_deg"]))
            prof, _y0, _y1 = width_profile(rot)
            height = max(len(prof), 1)

            def band(lo: float, hi: float) -> float:
                seg = prof[int(lo * height):max(int(lo * height) + 1, int(hi * height))]
                return float(np.median(seg)) if seg.size else 0.0

            throat_px = band(0.04, 0.16)
            mouth_px = band(0.72, 0.88)
            fr = final.get((sheet, corolla_id))
            opened = (fr["fold_state"] == "opened_full") if fr else rm.is_full_open(sheet, cid0 + 1)
            width_factor = 1.0 if opened else 2.0
            corolla_length = float(fr["corolla_length_mm"]) if fr else float(measured["length_mm"])
            corolla_width = (
                float(fr["corolla_width_fulleq_mm"])
                if fr else float(measured["width_mm"]) * width_factor
            )
            throat = throat_px * MM * width_factor
            mouth = mouth_px * MM * width_factor
            lobe = lobe_incision_mm(rot)

            og = organ.get((sheet, corolla_id)) or organ.get((sheet, str(cid0 + 1)))
            organ_length = (
                float(og["organ_length_mm"])
                if og and og.get("organ_length_mm") else float("nan")
            )

            def rnd(value: float, digits: int = 2):
                return round(value, digits) if value == value else ""

            rows.append({
                "sheet": sheet,
                "corolla_id": corolla_id,
                "fold_state": fr["fold_state"] if fr else "",
                "corolla_length_mm": rnd(corolla_length),
                "corolla_width_fulleq_mm": rnd(corolla_width),
                "throat_width_mm": rnd(throat),
                "mouth_width_mm": rnd(mouth),
                "corolla_aspect_L_W": rnd(corolla_length / corolla_width) if corolla_width else "",
                "tube_flare_W_throat": rnd(corolla_width / throat) if throat else "",
                "lobe_incision_mm": rnd(lobe),
                "lobe_incision_ratio": rnd(lobe / corolla_length) if corolla_length else "",
                "organ_corolla_ratio": (
                    rnd(organ_length / corolla_length)
                    if corolla_length and organ_length == organ_length else ""
                ),
            })
    return rows


def main() -> None:
    final = load_final()
    organ = load_organ()
    sheets = sorted(
        p.stem for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    all_rows = []
    for sheet in sheets:
        rows = measure_sheet(sheet, final, organ)
        all_rows.extend(rows)
        print(f"[{sheet}] {len(rows)} corollas", flush=True)

    out = Path("results_shimask_all/pollination_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {out}  ({len(all_rows)} corollas)")


if __name__ == "__main__":
    main()
