# -*- coding: utf-8 -*-
"""Create a non-destructive rescue package for v3 extraction failures.

Normal measurements remain untouched. Missing organ associations and low-confidence
corollas are exported separately for secondary inspection. shimask is never read.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine

MM_PX = float(base.MM_PX)
MM2_PX = float(base.MM2_PX)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def find_raw(root: Path, stem: str) -> Path:
    hits = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} and p.stem == stem]
    if not hits:
        raise FileNotFoundError(stem)
    return sorted(hits, key=lambda p: (len(p.parts), p.as_posix()))[0]


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def relaxed_candidates(image: np.ndarray, target: np.ndarray, union: np.ndarray) -> list[dict]:
    light, a, b, local_a, local_b, local_dark, chroma = refine._lab_channels(image)
    reach = max(5, int(round(42.0 / MM_PX)))
    band = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    band[cv2.dilate(union, np.ones((5, 5), np.uint8)) > 0] = 0
    cand = (((local_dark > 2.2) & (light < 253.0) & (b > 0.5)) |
            ((local_dark > 4.0) & (light < 248.0) & (chroma > 2.0))).astype(np.uint8)
    cand &= band
    close = max(3, int(round(0.8 / MM_PX))) | 1
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contour = max(contours, key=cv2.contourArea) if contours else None
    out = []
    for label in range(1, n):
        area_mm2 = float(stats[label, cv2.CC_STAT_AREA]) * MM2_PX
        if not 0.25 <= area_mm2 <= 140.0:
            continue
        x, y, w, h = [int(stats[label, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        crop = (labels[y:y+h, x:x+w] == label).astype(np.uint8)
        cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        rw, rh = cv2.minAreaRect(max(cs, key=cv2.contourArea))[1]
        length_mm = max(rw, rh) * MM_PX; width_mm = min(rw, rh) * MM_PX
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        if not (2.5 <= length_mm <= 48.0 and width_mm <= 9.0 and aspect >= 1.35):
            continue
        cx, cy = map(float, cents[label])
        distance = -cv2.pointPolygonTest(target_contour, (cx, cy), True) * MM_PX if target_contour is not None else 999.0
        selected = labels == label
        darkness = float(np.mean(local_dark[selected])); yellow = float(np.mean(b[selected]))
        score = 1.5 * min(aspect / 8.0, 1.0) + min(darkness / 12.0, 1.0) + 0.5 * min(max(yellow, 0.0) / 10.0, 1.0) + math.exp(-distance / 16.0)
        out.append(dict(cx=round(cx,2), cy=round(cy,2), x=x, y=y, width_px=w, height_px=h,
                        organ_len_mm=round(length_mm,2), organ_width_mm=round(width_mm,2),
                        aspect=round(aspect,2), distance_mm=round(distance,2), rescue_score=round(score,3)))
    return sorted(out, key=lambda r: r["rescue_score"], reverse=True)[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="shimahotarubukuro")
    parser.add_argument("--results", default="results_shimask")
    parser.add_argument("--out-dir", default="results_v3/rescue")
    args = parser.parse_args()
    root, results, out = Path(args.data_root), Path(args.results), Path(args.out_dir)
    traits, organs = read_csv(results / "traits_v3.csv"), read_csv(results / "organs_v3.csv")
    instances = {(r.get("sheet"), r.get("organ_instance_id")) for r in organs}
    associated = {(r.get("sheet"), int(float(r["nearest_corolla"]))) for r in organs if r.get("nearest_corolla") not in (None, "")}
    candidate_rows, manual_rows, mask_rows = [], [], []
    for sheet in sorted({r["sheet"] for r in traits}):
        raw_path = find_raw(root, sheet); image = base.load_bgr(str(raw_path)); shape = image.shape[:2]
        rows = [r for r in traits if r["sheet"] == sheet]; island = rows[0]["island"]
        masks = {int(r["corolla_id"]): load_mask(results / "masks" / island / sheet / f"C{int(r['corolla_id'])}.png", shape) for r in rows}
        union = np.zeros(shape, np.uint8)
        for m in masks.values(): union |= m
        for row in rows:
            cid = int(row["corolla_id"]); key = (sheet, cid); target = masks[cid]
            if int(float(row.get("mask_qc_required", 0) or 0)) or float(row.get("mask_confidence", 1) or 1) < 0.75:
                ys, xs = np.where(target > 0)
                mask_rows.append(dict(island=island, sheet=sheet, corolla_id=cid, reason=row.get("mask_qc_reasons", ""),
                                      current_mask=f"masks/{island}/{sheet}/C{cid}.png", corrected_mask_path="",
                                      action="accept|replace_mask_png|exclude|split"))
            if key in associated:
                continue
            candidates = relaxed_candidates(image, target, union)
            overlay = image.copy()
            cs, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cs, -1, (0, 255, 255), 5)
            for rank, cand in enumerate(candidates, 1):
                cand.update(island=island, sheet=sheet, corolla_id=cid, candidate_rank=rank, status="rescue_only_not_measured")
                candidate_rows.append(cand)
                cv2.rectangle(overlay, (cand["x"], cand["y"]), (cand["x"]+cand["width_px"], cand["y"]+cand["height_px"]), (255,0,255), 4)
                cv2.putText(overlay, f"R{rank}", (cand["x"], max(30, cand["y"]-8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 3)
            manual_rows.append(dict(island=island, sheet=sheet, corolla_id=cid, selected_candidate_rank="",
                                    organ_type="pistil|stamen|unknown", x1="", y1="", x2="", y2="", width_mm="",
                                    action="select_candidate|enter_endpoints|not_present|unmeasurable", notes=""))
            ys, xs = np.where(target > 0); pad = max(30, int(round(45/MM_PX)))
            x0,x1=max(0,int(xs.min())-pad),min(shape[1],int(xs.max())+pad); y0,y1=max(0,int(ys.min())-pad),min(shape[0],int(ys.max())+pad)
            crop = overlay[y0:y1, x0:x1]
            (out / "rescue_overlays").mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out / "rescue_overlays" / f"{sheet}_C{cid}.jpg"))
    write_csv(out / "rescue_candidates.csv", candidate_rows, ["island","sheet","corolla_id","candidate_rank","cx","cy","organ_len_mm","organ_width_mm","aspect","distance_mm","rescue_score","status"])
    write_csv(out / "manual_rescue_template.csv", manual_rows, ["island","sheet","corolla_id","selected_candidate_rank","organ_type","x1","y1","x2","y2","width_mm","action","notes"])
    write_csv(out / "mask_rescue_template.csv", mask_rows, ["island","sheet","corolla_id","reason","current_mask","corrected_mask_path","action"])
    print(f"organ rescue targets={len(manual_rows)} candidates={len(candidate_rows)} mask rescue targets={len(mask_rows)}")


if __name__ == "__main__":
    main()
