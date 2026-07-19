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


# Hand-set orientation (degrees) for corollas whose minAreaRect box is judged wrong
# on review; keyed by (sheet, id). Empty by default - the ROI box handles every
# corolla (e.g. kozu2 C3's diagonal falls straight out of the box), so overrides are
# only added if a specific specimen needs one.
MANUAL_AXIS: dict[tuple[str, str], float] = {}


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




def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask by flood-filling the background."""
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    filled = padded | (1 - flood)
    return filled[1:-1, 1:-1].astype(np.uint8)


def _solid_roi(mask_u8: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (solid corolla region, filled_by_hull).

    A properly segmented corolla is already solid, so filling its holes is a no-op.
    A hollow outline (an open hand-drawn ring the pipeline could not seal) has no
    enclosed interior to fill; bridge the gap by morphological closing, widening the
    kernel only until the interior fills, then flood-fill. If even that fails - a
    side of the outline was left open - fall back to the convex hull so the corolla
    is still measured (length/width are unaffected; area is a slight overestimate
    across the concave lobe notches). ``filled_by_hull`` flags that last case.
    """
    filled = _fill_holes(mask_u8)
    base_area = int(mask_u8.sum())
    if int(filled.sum()) >= 1.3 * base_area:
        return filled, False  # a closed ring/holes filled straight away
    # Already a solid blob? A real corolla occupies most of its convex hull, whereas
    # a thin outline occupies only a small fraction of it.
    cnt = max(cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    hull_pts = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull_pts), 1.0)
    if base_area >= 0.45 * hull_area:
        return filled, False
    # Thin outline: bridge the gap by closing, widening the kernel only as needed.
    # Accept a seal only once the filled region covers most of the corolla (its
    # convex hull), so a partial bridge that encloses a small pocket is rejected.
    for k in (15, 25, 41, 61, 81, 101):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sealed = _fill_holes(cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel))
        if int(sealed.sum()) >= 0.55 * hull_area:
            return sealed, False
    # A side was left open and closing cannot seal it: fall back to the convex hull.
    hull = np.zeros_like(mask_u8)
    cv2.fillConvexPoly(hull, hull_pts, 1)
    return hull, True


def _upright_top_mid(mask_local: np.ndarray, ang: float) -> np.ndarray:
    """Top-edge midpoint measured after tilt correction.

    The raw silhouette is still tilted, so its topmost edge is skewed off the
    tube-base centre. Rotate the mask so the resolved axis is vertical, take the
    midpoint of the top edge in that upright frame, then map it back to the
    original image - this is the true tube-base centre the axis should run through.
    """
    ys, xs = np.where(mask_local > 0)
    centre = (float(xs.mean()), float(ys.mean()))
    M = cv2.getRotationMatrix2D(centre, ang - 90.0, 1.0)
    h, w = mask_local.shape
    rot = cv2.warpAffine(mask_local, M, (w, h), flags=cv2.INTER_NEAREST)
    ry, rx = np.where(rot > 0)
    y_top, y_bot = ry.min(), ry.max()
    band = ry <= y_top + max(2, int(0.06 * (y_bot - y_top)))
    x0 = rx[band].mean()
    inv = cv2.invertAffineTransform(M)
    p = inv @ np.array([x0, float(y_top), 1.0])
    return np.array([p[0], p[1]])


def medial_axis(
    mask_local: np.ndarray,
    spot_local: np.ndarray,
    *,
    force_angle: float | None = None,
) -> dict[str, object]:
    """ROI-based corolla dimensions from the minimum-area oriented bounding box.

    Corolla length and width are read straight off the ROI (the corolla mask):
    ``cv2.minAreaRect`` fits the tightest rotated rectangle to the silhouette and
    its two side lengths are the corolla dimensions - the side nearer vertical is
    the length (base->tip; specimens are mounted tube-up) and the other is the
    width. This depends only on the ROI shape, so it is stable and needs no fragile
    per-flower symmetry/axis search. ``force_angle`` overrides the orientation for a
    corolla whose box is judged wrong on review; length and width are then the
    extents along and across that fixed direction. The reported start point is the
    ROI's top-edge midpoint (the tube-base centre).

    ``spot_local`` is accepted for call-site compatibility and is not used.
    """
    # Repair the ROI first: a few annotations trace only the corolla outline with a
    # gap too wide for the pipeline's 3 mm seal, so the interior never filled and the
    # mask is a hollow ring whose area is wrong (length/width, taken from the outline
    # extent, are still fine). _solid_roi fills the interior, widening the gap-seal
    # only as much as each mask needs; already-solid masks pass through unchanged.
    mask_u8, filled_by_hull = _solid_roi((mask_local > 0).astype(np.uint8))
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    ys, xs = np.where(mask_u8 > 0)
    pts = np.column_stack([xs, ys]).astype(float)

    if force_angle is not None:
        ang = float(force_angle) % 180.0
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        if u[1] < 0:
            u = -u
        n = np.array([-u[1], u[0]])
        lon_all, lat_all = pts @ u, pts @ n
        length_px = float(lon_all.max() - lon_all.min())
        width_px = float(lat_all.max() - lat_all.min())
    else:
        centre, (rw, rh), angle = cv2.minAreaRect(cnt)
        box = cv2.boxPoints((centre, (rw, rh), angle))
        e1, e2 = box[1] - box[0], box[2] - box[1]
        l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        # Assign the side nearer vertical as the length (upright-mounted flowers).
        vert1, vert2 = abs(e1[1]) / (l1 + 1e-9), abs(e2[1]) / (l2 + 1e-9)
        if vert1 >= vert2:
            length_px, width_px, ldir = l1, l2, e1
        else:
            length_px, width_px, ldir = l2, l1, e2
        u = ldir / (np.linalg.norm(ldir) + 1e-9)
        if u[1] < 0:
            u = -u
        n = np.array([-u[1], u[0]])
        ang = math.degrees(math.atan2(u[1], u[0])) % 180.0

    length_mm = length_px * MM_PX
    width_mm = width_px * MM_PX
    area_mm2 = float(mask_u8.sum()) * MM_PX * MM_PX
    # Fill ratio: how much of the oriented box the ROI occupies (a torn or badly
    # segmented mask sits low; scalloped lobes keep normal corollas around 0.6-0.8).
    fill_ratio = float(mask_u8.sum()) / (length_px * width_px + 1e-9)

    # Start point = ROI top-edge midpoint (tube-base centre); draw base->tip along
    # the length direction and place the width bar at the widest cross-section.
    p_top = _upright_top_mid(mask_local, ang)
    lon, lat = (pts - p_top) @ u, (pts - p_top) @ n
    base_pt = p_top + float(lon.min()) * u
    tip_pt = p_top + float(lon.max()) * u
    n_bins = 26
    edges = np.linspace(float(lon.min()), float(lon.max()), n_bins + 1)
    w_center = w_lo = w_hi = 0.0
    best_w = 0.0
    for i in range(n_bins):
        sel = (lon >= edges[i]) & (lon < edges[i + 1])
        if int(sel.sum()) < 5:
            continue
        span = float(lat[sel].max() - lat[sel].min())
        if span > best_w:
            best_w = span
            w_center = (edges[i] + edges[i + 1]) / 2.0
            w_lo, w_hi = float(lat[sel].min()), float(lat[sel].max())
    wc = p_top + w_center * u
    return {
        "fill_ratio": round(float(fill_ratio), 4),
        "filled_by_hull": filled_by_hull,
        "angle_deg": round(float(ang), 2),
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 1),
        "base_xy": (float(base_pt[0]), float(base_pt[1])),
        "tip_xy": (float(tip_pt[0]), float(tip_pt[1])),
        "w0_xy": tuple((wc + w_lo * n).tolist()),
        "w1_xy": tuple((wc + w_hi * n).tolist()),
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
            manual = MANUAL_AXIS.get((sheet, f"{cid}{suffix}"))
            m = medial_axis(mask_local, spot_local, force_angle=manual)
            fold = "opened_full" if opened else "folded_half"
            factor = 1.0 if opened else 2.0
            qc = []
            if manual is not None:
                # Orientation set by hand after review (the ROI box was judged wrong).
                qc.append("manual_axis")
            if float(m["length_mm"]) > LENGTH_MERGE_MM:
                qc.append("length_outlier")
            if m.get("filled_by_hull"):
                # Outline left open on one side; area came from the convex hull and
                # is a slight overestimate (length/width are still fine).
                qc.append("roi_open_outline")
            elif float(m["fill_ratio"]) < 0.45:
                # ROI fills little of its box -> possibly torn or mis-segmented mask.
                qc.append("irregular_roi")
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
                "roi_angle_deg": m["angle_deg"],
                "roi_fill_ratio": m["fill_ratio"],
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
