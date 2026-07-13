#!/usr/bin/env python3
"""Human-review package for the single sheet oshima10~13 (canonical ruler-at-top).

Starts from the accepted PRE-QC corolla masks (green/magenta outlines in the
committed reviewed overlay) and the committed corolla ids. It does NOT re-segment
and does NOT invent axes. Only these reviewed decisions are applied:

  * C10  : axis corrected onto the central-petal midline (manual visual review;
           see measure_guides_review_axis_overrides.py). Mask unchanged.
  * C11  : excluded (corolla damaged / petals missing).
  * C17  : kept; only the horizontally elongated bottom-crease noise removed.
  * C18  : kept; only the horizontally elongated bottom-crease noise removed.
  * C1,C2,C4,C6,C10,C14 : visible attached pistil recorded.

All other corolla masks and axes are left at PRE-QC.
"""
from __future__ import annotations
import csv, math, os
import numpy as np, cv2

import measure_guides as base
import measure_guides_symmetry_axis as sym
import measure_guides_review_axis_overrides as ovr

RAW = "shimahotarubukuro/oshima/oshima10~13.jpg"
OVERLAY = "results_single/oshima10-13/overlays/Oshima_oshima10~13.png"
TRAITS = "results/reviewed/oshima10-13/traits.csv"
FOLDER, SHEET = "oshima", "oshima10~13"
OUT = "results/reviewed/oshima10-13/review"
os.makedirs(OUT, exist_ok=True)

PISTIL_IDS = [1, 2, 4, 6, 10, 14]
EXCLUDE = {11: "corolla_damaged_missing_petals"}
NOISE_IDS = {17: "horizontal_bottom_crease_noise", 18: "horizontal_bottom_crease_noise"}


def _png(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def load():
    raw = base.load_bgr(RAW)
    ov = cv2.imdecode(np.fromfile(OVERLAY, np.uint8), cv2.IMREAD_COLOR)
    return raw, ov


def committed_centroids():
    rows = list(csv.DictReader(open(TRAITS, encoding="utf-8-sig")))
    return {int(r["corolla_id"]): (float(r["cx"]), float(r["cy"])) for r in rows}


def corolla_masks(raw, ov):
    """Return {corolla_id: filled mask (raw canonical)} by matching accepted outlines
    to the committed corolla centroids. C17/C18 share one split component, so the
    merged bottom blob is divided at the mid-x between their committed centroids."""
    rh, rw = raw.shape[:2]
    oh, ow = ov.shape[:2]
    sx, sy = rw / ow, rh / oh
    b, g, r = cv2.split(ov)
    green = ((g > 165) & (r < 135) & (b < 150)).astype(np.uint8) * 255
    mag = ((r > 150) & (b > 150) & (g < 120)).astype(np.uint8) * 255
    outl = cv2.morphologyEx(cv2.bitwise_or(green, mag), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(outl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
    cen = committed_centroids()

    scaled = []
    for c in cnts:
        cc = c.astype(np.float32).copy(); cc[:, 0, 0] *= sx; cc[:, 0, 1] *= sy
        cc = np.rint(cc).astype(np.int32)
        m = cv2.moments(cc)
        if m["m00"] <= 0:
            continue
        scaled.append(((m["m10"] / m["m00"], m["m01"] / m["m00"]), cc))

    masks = {}
    used = set()
    # assign each committed id to its nearest not-yet-used contour centroid
    for cid in sorted(cen):
        cx, cy = cen[cid]
        best, bd = None, 1e18
        for i, (ctr, cc) in enumerate(scaled):
            if i in used:
                continue
            d = (ctr[0] - cx) ** 2 + (ctr[1] - cy) ** 2
            if d < bd:
                bd, best = d, i
        if best is None:
            continue
        used.add(best)
        mask = np.zeros((rh, rw), np.uint8)
        cv2.drawContours(mask, [scaled[best][1]], -1, 1, -1)
        masks[cid] = mask
    # C17 and C18 are an auto-split pair joined at the bottom by the paper-crease
    # noise: their accepted outline is the single largest MAGENTA (flagged) blob.
    # Take that blob directly and divide it at the mid-x between the two committed
    # centroids, instead of the ambiguous nearest-contour assignment above.
    mcnts, _ = cv2.findContours(mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mcnts = [c for c in mcnts if cv2.contourArea(c) > 1000]
    if mcnts:
        big = max(mcnts, key=cv2.contourArea).astype(np.float32).copy()
        big[:, 0, 0] *= sx; big[:, 0, 1] *= sy
        union = np.zeros((rh, rw), np.uint8)
        cv2.drawContours(union, [np.rint(big).astype(np.int32)], -1, 1, -1)
        midx = int(round((cen[17][0] + cen[18][0]) / 2))
        left = union.copy(); left[:, midx:] = 0
        right = union.copy(); right[:, :midx] = 0
        masks[17], masks[18] = left, right
    return masks


def remove_horizontal_noise(mask):
    """Keep the corolla body; drop only thin, horizontally elongated appendages
    (bottom paper-crease strips). Body = vertical-opening (survives only where the
    mask is tall); noise = mask minus the body component holding the corolla."""
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    body = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    # regrow the body a little so we do not shave the true lobe edges
    body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    body = cv2.bitwise_and(body, mask)
    n, lbl, st, ct = cv2.connectedComponentsWithStats(body, 8)
    if n <= 1:
        return mask, np.zeros_like(mask)
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    body = (lbl == keep).astype(np.uint8)
    cleaned = cv2.bitwise_and(mask, body)
    noise = cv2.bitwise_and(mask, cv2.bitwise_not(cleaned))
    return cleaned, noise


def main():
    raw, ov = load()
    masks = corolla_masks(raw, ov)
    ids = sorted(masks)

    # ---- axes: PRE-QC for all, override for C10 ----
    axis_rows = []
    reviewed_axes = {}
    for cid in ids:
        pre = sym.estimate_symmetry_axis(masks[cid])
        o = ovr.get_axis_override(FOLDER, SHEET, cid)
        if o is not None:
            rb = (float(o["base_x"]), float(o["base_y"]))
            rt = (float(o["tip_x"]), float(o["tip_y"]))
            source = "reviewed_override"
        else:
            rb, rt = pre.base_xy, pre.tip_xy
            source = "preqc_symmetry"
        reviewed_axes[cid] = (rb, rt)
        axis_rows.append({
            "island": "Oshima", "sheet": SHEET, "corolla_id": cid,
            "preqc_base_x": round(pre.base_xy[0], 3), "preqc_base_y": round(pre.base_xy[1], 3),
            "preqc_tip_x": round(pre.tip_xy[0], 3), "preqc_tip_y": round(pre.tip_xy[1], 3),
            "reviewed_base_x": round(rb[0], 3), "reviewed_base_y": round(rb[1], 3),
            "reviewed_tip_x": round(rt[0], 3), "reviewed_tip_y": round(rt[1], 3),
            "axis_source": source,
            "changed_from_preqc": "yes" if source == "reviewed_override" else "no",
        })
    _write_csv(os.path.join(OUT, "reviewed_axes.csv"), axis_rows)

    # ---- C17/C18 noise removal ----
    mask_corr_rows = []
    cleaned_masks = dict(masks)
    for cid, reason in NOISE_IDS.items():
        if cid not in masks:
            continue
        cleaned, noise = remove_horizontal_noise(masks[cid])
        cleaned_masks[cid] = cleaned
        removed = int(noise.sum()); kept = int(cleaned.sum())
        mask_corr_rows.append({
            "island": "Oshima", "sheet": SHEET, "corolla_id": cid,
            "correction": "remove_horizontal_noise", "reason": reason,
            "kept_area_px": kept, "removed_noise_px": removed,
            "removed_fraction": round(removed / max(kept + removed, 1), 4),
            "mask_geometry": "corolla_body_retained",
        })
        _mask_before_after(raw, masks[cid], cleaned, noise, cid)
    _write_csv(os.path.join(OUT, "reviewed_mask_corrections.csv"), mask_corr_rows)

    # ---- exclusions ----
    _write_csv(os.path.join(OUT, "reviewed_exclusions.csv"), [
        {"island": "Oshima", "sheet": SHEET, "corolla_id": cid, "excluded": "yes", "reason": reason}
        for cid, reason in EXCLUDE.items()
    ])

    # ---- human_review.csv ----
    hr = []
    for cid in ids:
        if cid in EXCLUDE:
            status, note = "excluded", EXCLUDE[cid]
        elif cid in NOISE_IDS:
            status, note = "retained_noise_removed", NOISE_IDS[cid]
        else:
            status, note = "retained", ""
        hr.append({
            "island": "Oshima", "sheet": SHEET, "corolla_id": cid,
            "review_status": status,
            "visible_pistil_attached": "yes" if cid in PISTIL_IDS else "",
            "reproductive_organ_present": "yes" if cid in PISTIL_IDS else "",
            "organ_type_reviewed": "pistil" if cid in PISTIL_IDS else "",
            "nearest_corolla_reviewed": cid if cid in PISTIL_IDS else "",
            "association_confirmed": "yes" if cid in PISTIL_IDS else "",
            "axis_reviewed": "yes" if cid == 10 else "",
            "note": note,
        })
    _write_csv(os.path.join(OUT, "human_review.csv"), hr)

    # ---- C10 BEFORE / AFTER ----
    _c10_before_after(raw, masks[10], axis_rows)

    # ---- full-sheet reviewed QC ----
    _full_sheet_qc(raw, cleaned_masks, reviewed_axes)

    print(f"corollas={len(ids)} axes_written={len(axis_rows)} "
          f"mask_corrections={len(mask_corr_rows)} excluded={list(EXCLUDE)} -> {OUT}")


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close(); return
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _crop_box(mask, mgn):
    ys, xs = np.where(mask > 0)
    return max(0, xs.min() - mgn), max(0, ys.min() - mgn), xs.max() + mgn, ys.max() + mgn


def _c10_before_after(raw, mask, axis_rows):
    row = next(r for r in axis_rows if r["corolla_id"] == 10)
    x0, y0, x1, y1 = _crop_box(mask, 80)
    x1 = min(raw.shape[1], x1); y1 = min(raw.shape[0], y1)
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    sc = 2.4
    for tag, bx, by, tx, ty, col in [
        ("before", row["preqc_base_x"], row["preqc_base_y"], row["preqc_tip_x"], row["preqc_tip_y"], (0, 0, 235)),
        ("after", row["reviewed_base_x"], row["reviewed_base_y"], row["reviewed_tip_x"], row["reviewed_tip_y"], (0, 180, 0)),
    ]:
        crop = raw[y0:y1, x0:x1].copy()
        big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        P = lambda px, py: (int(round((px - x0) * sc)), int(round((py - y0) * sc)))
        cv2.drawContours(big, [((cnt - [x0, y0]) * sc).astype(np.int32)], -1, (0, 200, 0), 1)
        cv2.arrowedLine(big, P(bx, by), P(tx, ty), col, 3, cv2.LINE_AA, tipLength=0.05)
        cv2.circle(big, P(bx, by), 9, col, -1)
        cv2.circle(big, P(tx, ty), 7, col, 2)
        cv2.putText(big, f"C10 {tag.upper()}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        _png(os.path.join(OUT, f"c10_{tag}.png"), big)


def _mask_before_after(raw, before, after, noise, cid):
    x0, y0, x1, y1 = _crop_box(before, 60)
    x1 = min(raw.shape[1], x1); y1 = min(raw.shape[0], y1)
    sc = 1.4
    panels = []
    for tag, m, extra in [("BEFORE", before, None), ("AFTER", after, noise)]:
        crop = raw[y0:y1, x0:x1].copy()
        ov = crop.copy()
        ov[m[y0:y1, x0:x1] > 0] = (0, 180, 0)
        vis = cv2.addWeighted(crop, 0.55, ov, 0.45, 0)
        if extra is not None:
            vis[extra[y0:y1, x0:x1] > 0] = (0, 0, 235)  # removed noise in red
        cnts, _ = cv2.findContours(m[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 120, 0), 2)
        cv2.putText(vis, f"C{cid} {tag}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        panels.append(cv2.resize(vis, None, fx=sc, fy=sc))
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
    _png(os.path.join(OUT, f"c{cid}_mask_before_after.png"), np.hstack(panels))


def _full_sheet_qc(raw, masks, axes):
    canvas = raw.copy()
    for cid in sorted(masks):
        m = masks[cid]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        excluded = cid in EXCLUDE
        col = (0, 0, 230) if excluded else ((255, 0, 200) if cid in NOISE_IDS else (0, 190, 0))
        cv2.drawContours(canvas, cnts, -1, col, 3)
        ys, xs = np.where(m > 0); cx, cy = int(xs.mean()), int(ys.mean())
        rb, rt = axes[cid]
        if not excluded:
            cv2.arrowedLine(canvas, (int(rb[0]), int(rb[1])), (int(rt[0]), int(rt[1])),
                            (0, 0, 220) if cid == 10 else (60, 60, 60), 3, cv2.LINE_AA, tipLength=0.03)
        label = f"C{cid}"
        if excluded:
            label += " EXCL"
        if cid == 10:
            label += " axis*"
        if cid in NOISE_IDS:
            label += " denoised"
        if cid in PISTIL_IDS:
            label += " +pistil"
        cv2.putText(canvas, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    sc = 1700 / max(canvas.shape[:2])
    _png(os.path.join(OUT, "full_sheet_reviewed_qc.png"),
         cv2.resize(canvas, None, fx=sc, fy=sc))


if __name__ == "__main__":
    main()
