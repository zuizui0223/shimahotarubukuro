#!/usr/bin/env python3
"""Pollination-relevant floral traits per corolla, beyond raw size.

Motivated by Nagano, Abe, Kikvidze & Kikuchi (2014, Ecology & Evolution 4:3819,
doi:10.1002/ece3.1191), who showed that the floral size of the bumblebee-pollinated
bellflower Campanula punctata tracks the size of the LOCAL bumblebee, measuring
corolla length (CL), corolla width (CW), corolla entrance diameter (CE) and style
length (SL) - the traits that set the mechanical fit between a Bombus body and the
bell. This module measures the analogues that a flattened, pressed corolla can still
support, plus a block of nectar-guide *conspicuousness* traits.

The nectar-guide block exists to test the reviewer's hypothesis: the Izu Islands
lack Bombus except Oshima (Bombus ardens), and purple nectar guides are a signal
that directs bumblebees into the bell. Where the pollinator that reads that signal
is gone, the guide is a cost with no payoff, so - if divergence has run past neutral
drift - guide investment (coverage, spot number, and especially chromatic contrast)
should be reduced on the bumblebee-free islands relative to Oshima. These columns let
that be tested against corolla size (an allometry/neutral null) rather than assumed.

Geometry is taken from the reviewed hand ROI (which lives in the scan's own
coordinate frame); absolute size (CL, CW) is carried from corolla_traits_final.csv so
the headline numbers stay the iPhone-registered ones. Writes
results_shimask_all/pollination_traits.csv.

CAVEAT: specimens are pressed flat, so the true 3-D corolla-entrance diameter cannot
be recovered. throat_width_mm is the proximal (tube-region) width of the flattened
corolla - a proxy for CE, not CE itself - and is only comparable among these equally
flattened specimens.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from run_all_shimask_confirmed import find_raw

MM = float(base.MM_PX)


def upright(mask_local: np.ndarray, angle_deg: float):
    """Rotate the ROI so its length axis is vertical; return (rot_mask, M)."""
    h, w = mask_local.shape
    cx, cy = w / 2.0, h / 2.0
    pad = int(max(h, w))
    big = cv2.copyMakeBorder(mask_local, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    M = cv2.getRotationMatrix2D((cx + pad, cy + pad), angle_deg - 90.0, 1.0)
    rot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
    return rot, M


def width_profile(rot: np.ndarray):
    """Per-row corolla width (mask x-extent) from top (base) to bottom (tip)."""
    ys, xs = np.where(rot > 0)
    y0, y1 = ys.min(), ys.max()
    prof = np.zeros(y1 - y0 + 1)
    for k, y in enumerate(range(y0, y1 + 1)):
        row = xs[ys == y]
        prof[k] = (row.max() - row.min() + 1) if row.size else 0.0
    return prof, y0, y1


def lobe_incision_mm(rot: np.ndarray) -> float:
    """Depth of the scalloped distal margin (notches between corolla lobes).

    For each column, the tip-most (bottom) mask row is the petal edge. Over the
    distal half, deep lobe notches sit well above the lobe tips; the spread between
    the tip extreme and the 15th-percentile edge is the incision depth.
    """
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


def guide_and_color(raw: np.ndarray, piece: np.ndarray):
    """Guide spot mask (bbox frame), bbox origin, and conspicuousness metrics.

    Returns (spot_bbox, (x0, y0), contrast_dE, saturation) where contrast_dE is the
    CIELab distance between the mean guide colour and the mean surrounding petal
    colour (how much the guide stands out), and saturation is the guides' mean HSV
    chroma. Both are NaN when the corolla has no detectable guide.
    """
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    inside = piece[y0:y1, x0:x1] > 0
    b, g, r = cv2.split(sub.astype(int))
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    spot = (((r - g) > 18) & ((b - g) > -10) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 205)) & inside
    spot = cv2.morphologyEx(spot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0

    if int(spot.sum()) < 30:
        return spot.astype(np.uint8), (x0, y0), float("nan"), float("nan")
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(float)
    # Petal background = tissue inside the ROI that is not guide and not right beside
    # a guide (dilate to exclude the transition halo).
    halo = cv2.dilate(spot.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bg = inside & ~halo
    if int(bg.sum()) < 50:
        bg = inside & ~spot
    guide_lab = lab[spot].mean(0)
    bg_lab = lab[bg].mean(0)
    contrast = float(np.linalg.norm(guide_lab - bg_lab))
    saturation = float(hsv[:, :, 1][spot].mean())
    return spot.astype(np.uint8), (x0, y0), contrast, saturation


def load_final() -> dict:
    path = Path("results_shimask_all/corolla_traits_final.csv")
    return {(r["sheet"], r["corolla_id"]): r
            for r in csv.DictReader(path.open(encoding="utf-8-sig"))}


def load_organ() -> dict:
    path = Path("results_shimask_all/organ_traits.csv")
    return {(r["sheet"], r["corolla_id"]): r
            for r in csv.DictReader(path.open(encoding="utf-8-sig"))}


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
            m = rm.medial_axis(mask_local)
            angle = float(m["angle_deg"])

            rot, _ = upright(solid, angle)
            prof, y0r, y1r = width_profile(rot)
            H = max(len(prof), 1)

            def band(lo, hi):  # median width over a length fraction band [lo, hi]
                seg = prof[int(lo * H):max(int(lo * H) + 1, int(hi * H))]
                return float(np.median(seg)) if seg.size else 0.0

            throat_px = band(0.04, 0.16)   # proximal tube region (skip the apex taper)
            mouth_px = band(0.72, 0.88)    # distal rim (below the bulge, above lobe tips)

            fr = final.get((sheet, corolla_id))
            opened = (fr["fold_state"] == "opened_full") if fr else rm.is_full_open(sheet, cid0 + 1)
            wfac = 1.0 if opened else 2.0   # folding halves transverse width, not length
            cl = float(fr["corolla_length_mm"]) if fr else float(m["length_mm"])
            cw = float(fr["corolla_width_fulleq_mm"]) if fr else float(m["width_mm"]) * wfac
            throat = throat_px * MM * wfac
            mouth = mouth_px * MM * wfac
            lobe = lobe_incision_mm(rot)

            og = organ.get((sheet, corolla_id)) or organ.get((sheet, str(cid0 + 1)))
            sl = float(og["organ_length_mm"]) if og and og.get("organ_length_mm") else float("nan")

            spot, (bx0, by0), contrast, sat = guide_and_color(raw, piece)
            # Guide position along the corolla length (0 = base, 1 = tip).
            reach = cent = float("nan")
            if int(spot.sum()) >= 30:
                full = np.zeros(raw.shape[:2], np.uint8)
                gy, gx = np.where(spot)
                full[gy + by0, gx + bx0] = 1
                sub_full = rm.crop_to_mask(piece)  # frame matching mask_local
                # Rotate the guide mask into the same upright frame as the profile.
                gloc = np.zeros_like(mask_local)
                ys_p, xs_p = np.where(piece)
                px0, py0 = xs_p.min(), ys_p.min()
                ggy, ggx = np.where(spot)
                yy, xx = ggy + (by0 - py0), ggx + (bx0 - px0)
                ok = (yy >= 0) & (yy < gloc.shape[0]) & (xx >= 0) & (xx < gloc.shape[1])
                gloc[yy[ok], xx[ok]] = 1
                grot, _ = upright(gloc, angle)
                rys, _ = np.where(grot > 0)
                if rys.size:
                    reach = float(np.clip((rys.max() - y0r) / H, 0, 1))
                    cent = float(np.clip((rys.mean() - y0r) / H, 0, 1))

            def rnd(v, n=2):
                return round(v, n) if v == v else ""  # NaN -> blank

            rows.append({
                "sheet": sheet, "corolla_id": corolla_id,
                "fold_state": fr["fold_state"] if fr else "",
                "corolla_length_mm": rnd(cl), "corolla_width_fulleq_mm": rnd(cw),
                "throat_width_mm": rnd(throat),
                "mouth_width_mm": rnd(mouth),
                "corolla_aspect_L_W": rnd(cl / cw) if cw else "",
                "tube_flare_W_throat": rnd(cw / throat) if throat else "",
                "lobe_incision_mm": rnd(lobe),
                "lobe_incision_ratio": rnd(lobe / cl) if cl else "",
                "style_length_mm": rnd(sl),
                "style_corolla_ratio": rnd(sl / cl) if (cl and sl == sl) else "",
                "guide_coverage_pct": fr["guide_coverage_pct"] if fr else "",
                "n_guide_spots": fr["n_guide_spots"] if fr else "",
                "guide_reach_frac": rnd(reach),
                "guide_centroid_frac": rnd(cent),
                "guide_contrast_dE": rnd(contrast, 1),
                "guide_saturation": rnd(sat, 1),
                "has_nectar_guide": fr["has_nectar_guide"] if fr else "",
            })
    return rows


def main() -> None:
    final = load_final()
    organ = load_organ()
    sheets = sorted(p.stem for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    all_rows = []
    for sheet in sheets:
        rows = measure_sheet(sheet, final, organ)
        all_rows.extend(rows)
        print(f"[{sheet}] {len(rows)} corollas", flush=True)
    out = Path("results_shimask_all/pollination_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out}  ({len(all_rows)} corollas)")


if __name__ == "__main__":
    main()
