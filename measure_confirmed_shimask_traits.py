# -*- coding: utf-8 -*-
"""Measure floral traits from human-confirmed shimask annotations.

This is the analysis-grade path for the reviewed dataset. Red annotation strokes
provide the confirmed corolla outlines; green strokes provide confirmed
reproductive-organ traces. Raw scan pixels are still used for ruler calibration
and nectar-guide colour segmentation. Every output row carries explicit
provenance and must not be described as fully automatic extraction.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
from evaluate_shimask_labels import IMAGE_SUFFIXES, load_bgr
from evaluate_v3_against_shimask_v2 import find_raw
from export_shimask_ground_truth import close_and_fill_boundaries, skeletonize
from measure_guides_review_spots import spot_candidate_masks
from measure_guides_review_traits import calibrate_ruler, measure_flat_traits
from shimask_annotation_diff import annotation_masks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _lab_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, a, b = cv2.split(lab)
    return light, a - 128.0, b - 128.0


def _skeleton_trace_length_px(skeleton: np.ndarray) -> float:
    """Sum each undirected 8-neighbour skeleton edge exactly once.

    The older shared helper divided this edge sum by two even though right,
    down, and diagonal comparisons already count every undirected edge once.
    That halved confirmed organ lengths. This local implementation preserves
    the reviewed trace geometry without changing the legacy evaluation export.
    """
    q = np.asarray(skeleton) > 0
    if not q.any():
        return 0.0
    horizontal = int(np.sum(q[:, 1:] & q[:, :-1]))
    vertical = int(np.sum(q[1:, :] & q[:-1, :]))
    diag1 = int(np.sum(q[1:, 1:] & q[:-1, :-1]))
    diag2 = int(np.sum(q[1:, :-1] & q[:-1, 1:]))
    return float(horizontal + vertical + (diag1 + diag2) * np.sqrt(2.0))


def _confirmed_organs(green: np.ndarray, mm_per_px: float, sheet: str, island: str) -> list[dict]:
    cleaned = cv2.morphologyEx((green > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    rows: list[dict] = []
    organ_id = 0
    for label in range(1, n):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 20:
            continue
        component = (labels == label).astype(np.uint8)
        skel = skeletonize(component)
        length_px = _skeleton_trace_length_px(skel)
        if length_px < 5:
            continue
        organ_id += 1
        width_mm = (area_px / max(length_px, 1.0)) * mm_per_px
        rows.append({
            "island": island,
            "sheet": sheet,
            "confirmed_organ_id": organ_id,
            "cx": round(float(centroids[label][0]), 2),
            "cy": round(float(centroids[label][1]), 2),
            "organ_length_mm": round(length_px * mm_per_px, 3),
            "mean_width_mm": round(width_mm, 3),
            "annotation_area_px": area_px,
            "organ_identity": "human_confirmed_reproductive_organ_untyped",
            "measurement_status": "confirmed_from_green_annotation",
            "provenance": "shimask_human_review",
        })
    return rows


def measure_sheet(label_path: Path, labels_root: Path, raw_root: Path, overlay_dir: Path) -> tuple[list[dict], list[dict]]:
    raw_path = find_raw(label_path, raw_root)
    raw = load_bgr(raw_path)
    annotated = load_bgr(label_path)
    red_small, green_small = annotation_masks(annotated, raw)
    red = cv2.resize(red_small, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    green = cv2.resize(green_small, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)

    folder = raw_path.parent.name.lower()
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    sheet = raw_path.stem
    top = v2.specimen_top(raw)
    scale = calibrate_ruler(raw, top)
    mm_per_px = float(scale["mm_per_px"])
    _light, a_channel, b_channel = _lab_channels(raw)

    corolla_rows: list[dict] = []
    overlay = raw.copy()
    masks = close_and_fill_boundaries(red)
    for corolla_id, mask in enumerate(masks, 1):
        corolla = mask.astype(bool)
        strong, weak, guide = spot_candidate_masks(a_channel, b_channel, corolla)
        traits, _, _, _ = measure_flat_traits(corolla, guide.astype(bool), mm_per_px)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter_px = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
        guide_px = int((guide.astype(bool) & corolla).sum())
        corolla_px = int(corolla.sum())
        row = {
            "island": island,
            "sheet": sheet,
            "confirmed_corolla_id": corolla_id,
            **scale,
            **traits,
            "confirmed_perimeter_mm": round(perimeter_px * mm_per_px, 3),
            "guide_area_mm2": round(guide_px * mm_per_px * mm_per_px, 3),
            "guide_cov_pct": round(100.0 * guide_px / max(corolla_px, 1), 3),
            "guide_present": int(guide_px > 0),
            "strong_guide_px": int(strong.sum()),
            "weak_recovered_guide_px": int(weak.sum()),
            "mask_status": "human_confirmed_red_outline",
            "measurement_status": "analysis_grade_confirmed_mask",
            "provenance": "shimask_human_review_plus_raw_scan_colour",
        }
        corolla_rows.append(row)
        if contours:
            cv2.drawContours(overlay, [max(contours, key=cv2.contourArea)], -1, (0, 180, 0), 4)
        overlay[guide > 0] = (255, 0, 255)

    organ_rows = _confirmed_organs(green, mm_per_px, sheet, island)
    for organ in organ_rows:
        cv2.circle(overlay, (int(round(organ["cx"])), int(round(organ["cy"]))), 12, (0, 255, 0), -1)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    scale_preview = min(1.0, 1900.0 / max(overlay.shape[:2]))
    preview = cv2.resize(overlay, None, fx=scale_preview, fy=scale_preview, interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(overlay_dir / f"{island}_{sheet}.jpg"))
    return corolla_rows, organ_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="shimask")
    parser.add_argument("--raw-root", default="shimahotarubukuro")
    parser.add_argument("--out-dir", default="results_confirmed_traits")
    args = parser.parse_args()

    labels_root = Path(args.labels)
    raw_root = Path(args.raw_root)
    out = Path(args.out_dir)
    all_corollas: list[dict] = []
    all_organs: list[dict] = []
    label_files = sorted(p for p in labels_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for label_path in label_files:
        corollas, organs = measure_sheet(label_path, labels_root, raw_root, out / "overlays")
        all_corollas.extend(corollas)
        all_organs.extend(organs)

    write_csv(out / "confirmed_corolla_traits.csv", all_corollas)
    write_csv(out / "confirmed_reproductive_organs.csv", all_organs)
    (out / "PROVENANCE.txt").write_text(
        "Human-reviewed shimask red outlines and green organ traces are treated as confirmed annotations.\n"
        "Raw scans are used only for ruler calibration and nectar-guide colour segmentation.\n"
        "These outputs are analysis-grade reviewed measurements, not fully automatic predictions.\n",
        encoding="utf-8",
    )
    print(f"sheets={len(label_files)} confirmed_corollas={len(all_corollas)} confirmed_organs={len(all_organs)}")


if __name__ == "__main__":
    main()
