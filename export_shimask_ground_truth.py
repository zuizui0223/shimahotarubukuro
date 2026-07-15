# -*- coding: utf-8 -*-
"""Export canonical measurements from reviewed shimask annotations.

This script is for labelled sheets only. It converts annotation-only red strokes
into filled corolla masks and green strokes into skeleton lengths. The outputs are
used as canonical measurements and for training/calibration; they are never read by
the runtime extractor for an unseen image.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
from evaluate_v3_against_shimask_v2 import find_raw, normalise_stem
from evaluate_shimask_labels import IMAGE_SUFFIXES, load_bgr
from shimask_annotation_diff import annotation_masks

MM_PX = float(base.MM_PX)
MM2_PX = float(base.MM2_PX)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        skel |= img & (1 - opened)
        img = cv2.erode(img, kernel)
    return skel


def skeleton_length_px(skel: np.ndarray) -> float:
    q = skel > 0
    if not q.any():
        return 0.0
    horizontal = np.sum(q[:, 1:] & q[:, :-1])
    vertical = np.sum(q[1:, :] & q[:-1, :])
    diag1 = np.sum(q[1:, 1:] & q[:-1, :-1])
    diag2 = np.sum(q[1:, :-1] & q[:-1, 1:])
    return float(horizontal + vertical + (diag1 + diag2) * np.sqrt(2.0)) / 2.0


def close_and_fill_boundaries(red: np.ndarray) -> list[np.ndarray]:
    # Join small gaps in hand-drawn outlines, then fill external contours.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    joined = cv2.morphologyEx((red > 0).astype(np.uint8), cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        m = np.zeros_like(red, np.uint8)
        cv2.drawContours(m, [contour], -1, 1, -1)
        masks.append(m)
    return sorted(masks, key=lambda m: (np.where(m > 0)[0].mean(), np.where(m > 0)[1].mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="shimask")
    parser.add_argument("--raw-root", default="shimahotarubukuro")
    parser.add_argument("--out-dir", default="results_v3/shimask_ground_truth_measurements")
    args = parser.parse_args()
    labels_root, raw_root, out = Path(args.labels), Path(args.raw_root), Path(args.out_dir)
    mask_root = out / "corolla_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    corolla_rows, organ_rows = [], []
    label_files = sorted(p for p in labels_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for label_path in label_files:
        raw_path = find_raw(label_path, raw_root)
        raw, annotated = load_bgr(raw_path), load_bgr(label_path)
        red_small, green_small = annotation_masks(annotated, raw)
        scale_x = raw.shape[1] / annotated.shape[1]
        scale_y = raw.shape[0] / annotated.shape[0]
        red = cv2.resize(red_small, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        green = cv2.resize(green_small, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        sheet = raw_path.stem
        sheet_dir = mask_root / sheet
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for cid, mask in enumerate(close_and_fill_boundaries(red), 1):
            ys, xs = np.where(mask > 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
            cv2.imencode(".png", mask * 255)[1].tofile(str(sheet_dir / f"C{cid}.png"))
            corolla_rows.append({
                "sheet": sheet, "corolla_id": cid,
                "area_mm2": round(mask.sum() * MM2_PX, 3),
                "perimeter_mm": round(perimeter * MM_PX, 3),
                "bbox_x": int(xs.min()), "bbox_y": int(ys.min()),
                "bbox_width": int(xs.max() - xs.min() + 1), "bbox_height": int(ys.max() - ys.min() + 1),
                "source": "shimask_annotation_ground_truth",
            })
        cleaned = cv2.morphologyEx((green > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cleaned, 8)
        oid = 0
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            component = (labels == label).astype(np.uint8)
            skel = skeletonize(component)
            length_px = skeleton_length_px(skel)
            if length_px < 5:
                continue
            oid += 1
            width_mm = (area / max(length_px, 1.0)) * MM_PX
            organ_rows.append({
                "sheet": sheet, "organ_gt_id": oid,
                "cx": round(float(cents[label][0]), 2), "cy": round(float(cents[label][1]), 2),
                "organ_length_mm": round(length_px * MM_PX, 3),
                "mean_width_mm": round(width_mm, 3),
                "annotation_area_px": area,
                "source": "shimask_annotation_ground_truth",
            })
    write_csv(out / "ground_truth_corollas.csv", corolla_rows)
    write_csv(out / "ground_truth_organs.csv", organ_rows)
    print(f"sheets={len(label_files)} corollas={len(corolla_rows)} organs={len(organ_rows)}")


if __name__ == "__main__":
    main()
