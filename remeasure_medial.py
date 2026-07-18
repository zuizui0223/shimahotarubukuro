#!/usr/bin/env python3
"""Re-measure corolla length/width/area on a spot+symmetry medial axis.

Motivation
----------
The existing flat-trait orientation rotates the silhouette's longest (PCA) axis
to vertical and calls that "length". For opened, flattened corollas the longest
silhouette dimension is the lobe-to-lobe spread, so length and width come out
swapped, and on tilted specimens length is over-estimated.

This module instead finds the flower's bilateral-symmetry medial axis from the
reviewed corolla mask AND its nectar-guide spot pattern, then measures:
- corolla length  : extent ALONG the medial axis (tube base -> lobe tip)
- corolla width   : max extent PERPENDICULAR to the medial axis
- corolla area    : filled mask area

Half-folded (2.5-lobe) vs fully-open (5-lobe) is taken from a reviewed
ground-truth table, not auto-detected. To compare both on one basis, folded
individuals also get full-flower-equivalent columns (width x2, area x2).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import shimask_input
import measure_guides_symmetry_axis as sym
from run_all_shimask_confirmed import find_raw, IMAGE_SUFFIXES

MM_PX = float(base.MM_PX)

# Reviewed fold ground truth. "ALL" = every corolla on the sheet is fully open.
FULL_OPEN: dict[str, object] = {
    "kozu1": "ALL",
    "kozu2": "ALL",
    "shikine1": {1, 4, 5, 6},
    "toshima6-8": {3},
    "toshima3-6": {5},
}


def is_full_open(sheet: str, corolla_id: int) -> bool:
    spec = FULL_OPEN.get(sheet)
    if spec == "ALL":
        return True
    if isinstance(spec, set):
        return corolla_id in spec
    return False


def detect_guide_spots(raw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Purple/magenta nectar-guide spots inside the corolla mask (cropped frame)."""
    ys, xs = np.where(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    sub = raw[y0:ys.max() + 1, x0:xs.max() + 1]
    m = mask[y0:ys.max() + 1, x0:xs.max() + 1].astype(bool)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = cv2.split(sub.astype(int))
    spot = ((((r - g) > 18) & ((b - g) > -10) & (S > 50) & (V < 210)) & m).astype(np.uint8)
    spot = cv2.morphologyEx(spot, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return spot, m.astype(np.uint8), (y0, x0)


def _sym_score(mask_s: np.ndarray, spot_s: np.ndarray, angle: float, offset: float) -> float:
    mask_iou = sym._reflect_iou(mask_s, angle, offset)
    if spot_s.sum() > 12:
        spot_iou = sym._reflect_iou(cv2.dilate(spot_s, np.ones((3, 3), np.uint8)), angle, offset)
        return 0.45 * mask_iou + 0.55 * spot_iou
    return mask_iou


def _top_edge_midpoint(mask_s: np.ndarray) -> np.ndarray:
    """Midpoint of the topmost (tube-base) edge, on the downscaled mask."""
    ys, xs = np.where(mask_s > 0)
    y_top = ys.min()
    band = ys <= y_top + max(2, int(0.06 * (ys.max() - y_top)))
    return np.array([xs[band].mean(), float(y_top)])


def medial_axis(mask_local: np.ndarray, spot_local: np.ndarray, *, anchor_top: bool = False) -> dict[str, object]:
    """Symmetry-axis measurement on the cropped corolla frame.

    The reflection search runs on a downscaled mask over a near-vertical angle
    window (specimens are mounted upright, so the axis is always near-vertical;
    this stops the search from locking onto a wrong diagonal). When ``anchor_top``
    is set (folded corollas), the axis is forced through the top-edge midpoint -
    the tube-base centre lies on the flower axis and pins the otherwise-drifting
    offset, removing residual tilt on narrow folded silhouettes. Opened corollas
    are wide fans whose top-edge midpoint is not on the axis, so they keep the
    free offset search.
    """
    target_h = 150.0
    scale = min(1.0, target_h / mask_local.shape[0])
    ms = cv2.resize(mask_local, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ss = cv2.resize(spot_local, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    angle_min, angle_max = 62.0, 118.0
    best = (-1.0, 90.0, 0.0)
    if anchor_top:
        cs = np.where(ms > 0)
        centroid_s = np.array([cs[1].mean(), cs[0].mean()])
        p_top = _top_edge_midpoint(ms)
        for angle in np.arange(angle_min, angle_max + 0.01, 1.0):
            theta = math.radians(angle)
            normal = np.array([-math.sin(theta), math.cos(theta)])
            offset = float((p_top - centroid_s) @ normal)  # line passes through p_top
            score = _sym_score(ms, ss, float(angle), offset)
            if score > best[0]:
                best = (score, float(angle), offset)
    else:
        for angle in np.arange(angle_min, angle_max + 0.01, 1.0):
            for offset in np.arange(-18.0, 18.01, 2.0):
                score = _sym_score(ms, ss, float(angle), float(offset))
                if score > best[0]:
                    best = (score, float(angle), float(offset))
        _, ang0, off0 = best
        for offset in np.arange(off0 - 2.0, off0 + 2.01, 0.5):
            score = _sym_score(ms, ss, ang0, float(offset))
            if score > best[0]:
                best = (score, ang0, float(offset))
    _, ang, off_small = best
    offset_px = off_small / scale
    # Report symmetry quality as the best achievable at the chosen angle over any
    # offset, so anchoring (which fixes the offset) is not penalised as "low
    # symmetry" -- the score reflects how bilaterally symmetric the flower is.
    score = max(
        _sym_score(ms, ss, ang, float(o)) for o in np.arange(-18.0, 18.01, 2.0)
    )

    ys, xs = np.where(mask_local > 0)
    centroid = np.array([xs.mean(), ys.mean()])
    theta = math.radians(ang)
    axis = np.array([math.cos(theta), math.sin(theta)])
    if axis[1] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    origin = centroid + offset_px * normal

    pts = np.column_stack([xs, ys]).astype(float)
    lon = (pts - origin) @ axis
    lat = (pts - origin) @ normal
    lo, hi = lon.min(), lon.max()
    length_mm = (hi - lo) * MM_PX

    # widest perpendicular cross-section, scanning bins along the axis
    n_bins = 26
    edges = np.linspace(lo, hi, n_bins + 1)
    width_px = 0.0
    w_center = w_lo = w_hi = 0.0
    for i in range(n_bins):
        sel = (lon >= edges[i]) & (lon < edges[i + 1])
        if int(sel.sum()) < 5:
            continue
        span = float(lat[sel].max() - lat[sel].min())
        if span > width_px:
            width_px = span
            w_center = (edges[i] + edges[i + 1]) / 2.0
            w_lo, w_hi = float(lat[sel].min()), float(lat[sel].max())
    width_mm = width_px * MM_PX
    area_mm2 = float((mask_local > 0).sum()) * MM_PX * MM_PX

    base_pt = origin + lo * axis
    tip_pt = origin + hi * axis
    wc = origin + w_center * axis
    return {
        "sym_score": round(float(score), 4),
        "angle_deg": round(float(ang), 2),
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 1),
        "base_xy": (float(base_pt[0]), float(base_pt[1])),
        "tip_xy": (float(tip_pt[0]), float(tip_pt[1])),
        "w0_xy": tuple((wc + w_lo * normal).tolist()),
        "w1_xy": tuple((wc + w_hi * normal).tolist()),
    }


LENGTH_MERGE_MM = 55.0  # a single corolla never reaches this; longer => two merged


def split_merged_pair(mask: np.ndarray) -> list[np.ndarray]:
    """Split a mask that joins two corollas at a narrow neck.

    Two adjacent hand outlines can touch, so the flood-fill yields one long
    component with a strong central constriction. Only masks whose PCA-major
    length exceeds a single corolla and that show a deep central neck are split;
    a half-folded single corolla (normal length, folded waist) is left intact.
    """
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m)
    pts = np.column_stack([xs, ys]).astype(float)
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    axis = vt[0]
    if axis[1] < 0:
        axis = -axis
    lon = (pts - mean) @ axis
    lo, hi = lon.min(), lon.max()
    if (hi - lo) * MM_PX < LENGTH_MERGE_MM:
        return [m]
    n_bins = 60
    idx = np.clip(((lon - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    prof = np.array([(idx == i).sum() for i in range(n_bins)], dtype=float)
    c0, c1 = int(n_bins * 0.28), int(n_bins * 0.72)
    neck = c0 + int(np.argmin(prof[c0:c1]))
    top_max = prof[:neck].max() if neck > 0 else 0.0
    bot_max = prof[neck:].max() if neck < n_bins else 0.0
    if not (min(top_max, bot_max) > 0 and prof[neck] < 0.55 * min(top_max, bot_max)):
        return [m]
    t_neck = lo + (neck + 0.5) / n_bins * (hi - lo)
    side = lon < t_neck
    out = []
    for sel in (side, ~side):
        piece = np.zeros_like(m)
        piece[ys[sel], xs[sel]] = 1
        count, labels, stats, _ = cv2.connectedComponentsWithStats(piece, 8)
        if count > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            piece = (labels == k).astype(np.uint8)
        out.append(piece)
    return out


def measure_sheet(sheet: str, raw_path: Path, shimask_path: Path) -> list[dict[str, object]]:
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(shimask_path))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    rows: list[dict[str, object]] = []
    for cid, comp in enumerate(comps, start=1):
        full_mask = comp["mask"].astype(np.uint8)
        pieces = split_merged_pair(full_mask)
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            spot_local, mask_local, _ = detect_guide_spots(raw, piece)
            opened = is_full_open(sheet, cid)
            m = medial_axis(mask_local, spot_local, anchor_top=not opened)
            fold = "opened_full" if opened else "folded_half"
            factor = 1.0 if opened else 2.0
            score = float(m["sym_score"])
            angle = float(m["angle_deg"])
            qc = []
            # The axis search is constrained near-vertical; a result pinned to the
            # window edge means the natural symmetry wanted a more diagonal axis,
            # so the constrained axis is only approximate -> manual review.
            if angle <= 64.0 or angle >= 116.0:
                qc.append("axis_review")
            # Folded silhouettes legitimately have low reflection symmetry (torn
            # edges, one-sided spots) yet their anchored axis is still correct, so
            # only a very low score points at a genuinely malformed mask.
            if score < 0.30:
                qc.append("low_symmetry")
            if float(m["length_mm"]) > LENGTH_MERGE_MM:
                qc.append("length_outlier")
            if suffix:
                qc.append("split_from_merged_pair")
            rows.append({
                "sheet": sheet,
                "corolla_id": f"{cid}{suffix}",
                "fold_state": fold,
                "corolla_length_mm": m["length_mm"],
                "corolla_width_obs_mm": m["width_mm"],
                "corolla_area_obs_mm2": m["area_mm2"],
                "corolla_width_fulleq_mm": round(float(m["width_mm"]) * factor, 2),
                "corolla_area_fulleq_mm2": round(float(m["area_mm2"]) * factor, 1),
                "medial_angle_deg": m["angle_deg"],
                "symmetry_score": m["sym_score"],
                "qc_flag": "|".join(qc),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("shimask"))
    parser.add_argument("--raw-root", type=Path, default=Path("shimahotarubukuro"))
    parser.add_argument("--out", type=Path, default=Path("results_shimask_all/medial_traits.csv"))
    args = parser.parse_args()

    labels = sorted(p for p in args.labels.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    all_rows: list[dict[str, object]] = []
    for label in labels:
        sheet = label.stem
        _, raw = find_raw(sheet, args.raw_root)
        rows = measure_sheet(sheet, raw, label)
        all_rows.extend(rows)
        print(f"[{sheet}] {len(rows)} corollas "
              f"(open={sum(r['fold_state']=='opened_full' for r in rows)})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    n_qc = sum(1 for r in all_rows if r["qc_flag"])
    print(f"Wrote {len(all_rows)} rows to {args.out}  (qc-flagged={n_qc})")


if __name__ == "__main__":
    main()
