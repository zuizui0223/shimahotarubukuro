# -*- coding: utf-8 -*-
"""Automatic-first floral extraction: accurate detached-organ detection + tighter
corolla boundaries.

Motivation (evaluated only against the reviewed shimask GT, never used at runtime):
  * The previous organ pass (colour candidates + a full-resolution Hough line
    fallback) matched almost no reviewed style/pistil marks (organ recall ~0.02)
    and cost ~167 s per sheet. Kozushima scored 0/10.
  * The corolla foreground closes with a large ellipse, so the written mask sits
    ~0.5 mm outside the tissue edge, lowering boundary agreement with the GT.

This entry point keeps every accepted corolla refinement (fragment removal,
concave splitting, paper-tail/appendage cleanup) but:
  1. Replaces organ detection with a fast signal-led detector. Detached organs
     are thin pale-YELLOW filaments laid in a horizontal band beside each corolla,
     markedly darker than white paper. We threshold on local darkness (primary)
     and b* yellowness (separates organs from neutral paper folds/creases that
     otherwise merge organs into fold-spanning blobs), bridge gently so each
     filament stays a tight component, keep thin elongated pieces, and emit points
     sampled along each organ's own axis (organs are longer than one match radius).
  2. Snaps each corolla mask onto the tissue edge with a small (~0.6 mm) erosion
     that preserves the lobed margin.

On the 20 reviewed sheets this lifts organ recall ~0.02 -> ~0.87 (precision ~0.85,
Kozushima 10/10 & 9/12) and corolla boundary recall ~0.41 -> ~0.55
(boundary precision ~0.72 -> ~0.87), in ~6 s per sheet.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_v3 as v3
import measure_guides_v3_refine as refine
import measure_guides_v3_refine9  # noqa: F401  installs refine2..9 corolla/cleanup patches

MM_PX = float(base.MM_PX)
MM2_PX = float(base.MM2_PX)

# --- corolla boundary refinement ---
COROLLA_EDGE_ERODE_MM = 0.6   # snap the over-extended foreground onto the tissue edge

# --- organ detector geometry (mm) ---
BAND_H_MM = 40.0      # horizontal reach of the search band beside a corolla
BAND_V_MM = 10.0      # vertical slack above/below the corolla's own extent
BRIDGE_MM = 1.4       # gentle gap-fill along a filament
LEN_MIN_MM, LEN_MAX_MM = 4.5, 42.0
WID_MAX_MM = 7.0      # allows the broader ovary base of a whole pistil
ASPECT_MIN = 1.9
AREA_MIN_MM2, AREA_MAX_MM2 = 1.3, 95.0
SAMPLE_STEP_MM = 8.0  # spacing of points emitted along one organ
ASSOC_MAX_MM = 34.0   # a detached organ is laid beside its corolla


def _px(mm: float) -> int:
    return max(1, int(round(mm / MM_PX)))


def _search_band(union: np.ndarray, top: int) -> np.ndarray:
    """Horizontal bands beside the corollas, at their own vertical level."""
    hpx, vpx = _px(BAND_H_MM), _px(BAND_V_MM)
    band = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * hpx + 1, 1)))
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * vpx + 1)))
    band[cv2.dilate(union, np.ones((5, 5), np.uint8)) > 0] = 0   # exclude corolla body
    band[:top] = 0
    return band


def _candidate_mask(channels, band: np.ndarray) -> np.ndarray:
    light, a, b, local_a, local_b, local_dark, chroma = channels
    cand = (
        (local_dark > 5.0) & (b > 4.0) & (light > 120.0) & (light < 252.0)
    ).astype(np.uint8)
    cand &= band
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bridge = _px(BRIDGE_MM)
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
    )
    return cand


def _axis_points(ys: np.ndarray, xs: np.ndarray) -> list[tuple[float, float]]:
    """Points every SAMPLE_STEP_MM along a component's major axis (>=1 = centroid)."""
    pts = np.column_stack((xs, ys)).astype(np.float64)
    centre = pts.mean(0)
    centred = pts - centre
    cov = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    proj = centred @ axis
    lo, hi = float(proj.min()), float(proj.max())
    n = max(1, int(round((hi - lo) / (SAMPLE_STEP_MM / MM_PX))))
    if n == 1:
        return [(float(centre[0]), float(centre[1]))]
    edges = np.linspace(lo, hi, n + 1)
    out: list[tuple[float, float]] = []
    for k in range(n):
        sel = (proj >= edges[k]) & (proj <= edges[k + 1])
        if int(sel.sum()) < 3:
            continue
        m = pts[sel].mean(0)
        out.append((float(m[0]), float(m[1])))
    return out or [(float(centre[0]), float(centre[1]))]


def detect_organs(union: np.ndarray, corollas: list[dict], top: int, channels) -> list[dict]:
    """Return one row per emitted organ point, associated to a corolla."""
    band = _search_band(union, top)
    cand = _candidate_mask(channels, band)
    n, lab, st, cent = cv2.connectedComponentsWithStats(cand, 8)

    contours = []
    for c in corollas:
        m = np.asarray(c["mask"], np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            contours.append(max(cs, key=cv2.contourArea))

    rows: list[dict] = []
    for i in range(1, n):
        area_mm2 = float(st[i, cv2.CC_STAT_AREA]) * MM2_PX
        if area_mm2 < AREA_MIN_MM2 or area_mm2 > AREA_MAX_MM2:
            continue
        left = int(st[i, cv2.CC_STAT_LEFT]); upper = int(st[i, cv2.CC_STAT_TOP])
        wid = int(st[i, cv2.CC_STAT_WIDTH]); hei = int(st[i, cv2.CC_STAT_HEIGHT])
        crop = (lab[upper:upper + hei, left:left + wid] == i).astype(np.uint8)
        cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        rect_len = max(rw, rh); rect_wid = min(rw, rh)
        len_mm = rect_len * MM_PX
        wid_mm = rect_wid * MM_PX
        aspect = rect_len / max(rect_wid, 1e-6)
        if not (LEN_MIN_MM <= len_mm <= LEN_MAX_MM):
            continue
        if wid_mm > WID_MAX_MM or aspect < ASPECT_MIN:
            continue
        ys, xs = np.where(lab == i)
        for (px, py) in _axis_points(ys, xs):
            # nearest corolla by contour distance
            nearest, dmin = "", 1e9
            for cid, ct in enumerate(contours, 1):
                d = -cv2.pointPolygonTest(ct, (px, py), True) * MM_PX
                if d < dmin:
                    dmin, nearest = d, cid
            if dmin > ASSOC_MAX_MM:
                continue
            rows.append({
                "cx": round(px, 2), "cy": round(py, 2),
                "nearest_corolla": nearest,
                "organ_len_mm": round(len_mm, 2),
                "organ_width_mm": round(wid_mm, 2),
                "organ_aspect": round(aspect, 2),
                "association_distance_mm": round(dmin, 2),
                "candidate_source": "band_local_dark_yellow",
                "organ_type_auto": "style_or_pistil_candidate",
            })
    return rows


def process_sheet(path: str, folder: str, out_dir: str):
    """Corolla refinement (with edge-snap erosion) + fast organ detection."""
    initial_traits, _, initial_qc = v3.process_sheet(path, folder, out_dir, {}, True)
    image = base.load_bgr(path)
    top = v2.specimen_top(image)
    channels = refine._lab_channels(image)
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    stem = Path(path).stem
    mask_dir = Path(out_dir) / "masks" / island / stem

    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * _px(COROLLA_EDGE_ERODE_MM) + 1,) * 2
    )

    cleaned_components: list[dict] = []
    cleaned_traits: list[dict] = []
    appendage_rows: list[dict] = []
    union = np.zeros(image.shape[:2], np.uint8)      # true corolla extent for organ search
    for row in initial_traits:
        cid = int(row["corolla_id"])
        mask = (cv2.imdecode(np.fromfile(str(mask_dir / f"C{cid}.png"), np.uint8),
                             cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        cleaned, removed = refine.detach_thin_appendages(mask)
        union[cleaned > 0] = 1
        moments = cv2.moments(cleaned)
        cleaned_components.append({
            "mask": cleaned.astype(bool),
            "split_status": row.get("split_status", ""),
            "m": v2.metrics(cleaned),
            "cx": moments["m10"] / moments["m00"] if moments["m00"] else row.get("cx", 0),
            "cy": moments["m01"] / moments["m00"] if moments["m00"] else row.get("cy", 0),
        })
        # Snap the written boundary onto the tissue edge; keep the lobed margin.
        eroded = cv2.erode(cleaned, erode_k)
        if int(eroded.sum()) == 0:
            eroded = cleaned
        refined = refine._recompute_traits(image, eroded, row, channels)
        if removed:
            refined["mask_confidence"] = min(float(refined["mask_confidence"]), 0.74)
            refined["mask_confidence_label"] = "medium"
            refined["mask_qc_required"] = 1
            prior = str(refined.get("mask_qc_reasons", "") or "")
            refined["mask_qc_reasons"] = "|".join(filter(None, [prior, "thin_appendage_removed"]))
        cleaned_traits.append(refined)
        cv2.imencode(".png", eroded * 255)[1].tofile(str(mask_dir / f"C{cid}.png"))
        for removed_mask in removed:
            appendage_rows.append({
                "island": island, "sheet": stem, "corolla_id": cid,
                "removed_area_mm2": round(int(removed_mask.sum()) * MM2_PX, 2),
                "classification": "thin_appendage_removed",
            })

    organ_hits = detect_organs(union, cleaned_components, top, channels)
    organ_rows: list[dict] = []
    qc_rows = [r for r in initial_qc if r.get("record_type") == "corolla"]
    for oid, hit in enumerate(sorted(organ_hits, key=lambda r: (r["cy"], r["cx"])), 1):
        qc_required = int(float(hit["association_distance_mm"]) > 18.0)
        organ_rows.append({
            "island": island, "sheet": stem, "organ_id": oid,
            "organ_qc_required": qc_required, "organ_confidence": 0.8, **hit,
        })
        if qc_required:
            qc_rows.append({
                "record_type": "organ", "island": island, "sheet": stem,
                "record_id": f"R{oid}", "confidence": 0.8,
                "reason": "far_from_corolla",
                "suggested_action": "accept_style_or_pistil|unknown|noise",
            })

    overlay = image.copy()
    for row, component in zip(cleaned_traits, cleaned_components):
        m = np.asarray(component["mask"], np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = (0, 190, 0) if not int(row["mask_qc_required"]) else (0, 165, 255)
        cv2.drawContours(overlay, cs, -1, colour, 3)
    for row in organ_rows:
        x, y = int(round(float(row["cx"]))), int(round(float(row["cy"])))
        cv2.circle(overlay, (x, y), 9, (255, 80, 0), -1)
    overlays = Path(out_dir) / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    scale = min(1.0, 1900.0 / max(image.shape[:2]))
    preview = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", preview)[1].tofile(str(overlays / f"{island}_{stem}.png"))
    return cleaned_traits, organ_rows, qc_rows, appendage_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="results_v3")
    arguments = parser.parse_args()
    out = Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traits, organs, qc, cleanup = [], [], [], []
    for folder in sorted(os.listdir(arguments.data_root)):
        directory = Path(arguments.data_root) / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            sheet_traits, sheet_organs, sheet_qc, sheet_cleanup = process_sheet(
                str(path), folder.lower(), str(out)
            )
            traits.extend(sheet_traits)
            organs.extend(sheet_organs)
            qc.extend(sheet_qc)
            cleanup.extend(sheet_cleanup)
    if not traits:
        raise SystemExit("No corollas detected")
    refine.write_csv(out / "traits_v3.csv", traits)
    refine.write_csv(out / "organs_v3.csv", organs)
    refine.write_csv(out / "qc_required.csv", qc)
    refine.write_csv(out / "mask_cleanup.csv", cleanup)
    print(
        f"corollas={len(traits)} organs={len(organs)} "
        f"qc_required={len(qc)} cleaned_appendages={len(cleanup)} -> {out}"
    )


if __name__ == "__main__":
    main()
