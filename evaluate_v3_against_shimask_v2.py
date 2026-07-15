# -*- coding: utf-8 -*-
"""Evaluate v3 using only review strokes added to shimask previews."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
from evaluate_shimask_labels import IMAGE_SUFFIXES, load_bgr, write_csv
from evaluate_v3_against_shimask import (
    boundary,
    find_raw,
    green_centres,
    normalise_stem,
    overlap_with_tolerance,
    read_rows,
    resize_mask,
    union_masks,
)
from shimask_annotation_diff import annotation_masks


def organ_instances(rows: list[dict]) -> list[list[tuple[float, float]]]:
    """Group axis sample points that belong to the same detected organ.

    ``organs_v3.csv`` can contain several rows per biological candidate so that
    a long curved organ is represented at multiple points. Those rows must count
    as one prediction during precision/recall evaluation.
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    for index, row in enumerate(rows):
        if row.get("cx") in (None, "") or row.get("cy") in (None, ""):
            continue
        instance_id = row.get("organ_instance_id")
        if instance_id in (None, ""):
            instance_id = row.get("organ_id")
        if instance_id in (None, ""):
            instance_id = f"row-{index}"
        grouped.setdefault(str(instance_id), []).append(
            (float(row["cx"]), float(row["cy"]))
        )
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="shimask")
    parser.add_argument("--raw-root", default="shimahotarubukuro")
    parser.add_argument("--predictions", default="results_shimask")
    parser.add_argument("--out-dir", default="results_v3/shimask_evaluation")
    args = parser.parse_args()

    labels_root = Path(args.labels)
    raw_root = Path(args.raw_root)
    predictions = Path(args.predictions)
    out = Path(args.out_dir)
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    traits = read_rows(predictions / "traits_v3.csv")
    organs = read_rows(predictions / "organs_v3.csv")
    metrics: list[dict] = []
    mapping: list[dict] = []

    label_files = sorted(
        path for path in labels_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for label_path in label_files:
        raw_path = find_raw(label_path, raw_root)
        raw = load_bgr(raw_path)
        annotated = load_bgr(label_path)
        red_small, green_small = annotation_masks(annotated, raw)
        red = resize_mask(red_small, raw.shape[:2])
        green = resize_mask(green_small, raw.shape[:2])

        sheet = raw_path.stem
        trait_rows = [row for row in traits if row.get("sheet") == sheet]
        organ_rows = [row for row in organs if row.get("sheet") == sheet]
        if not trait_rows:
            raise RuntimeError(f"No prediction rows for matched raw scan {raw_path}")
        island = trait_rows[0]["island"]
        predicted = union_masks(predictions / "masks" / island / sheet, raw.shape[:2])
        pred_boundary = boundary(predicted)

        scale = raw.shape[1] / max(annotated.shape[1], 1)
        tolerance = max(3, int(round(5.0 * scale)))
        red_recall = overlap_with_tolerance(red, pred_boundary, tolerance)
        boundary_precision = overlap_with_tolerance(pred_boundary, red, tolerance)

        gt_centres = green_centres(green)
        pred_instances = organ_instances(organ_rows)
        pred_sample_points = [point for instance in pred_instances for point in instance]
        match_radius = 10.0 / float(base.MM_PX)
        matched_gt = sum(
            any(
                math.hypot(gx - px, gy - py) <= match_radius
                for instance in pred_instances
                for px, py in instance
            )
            for gx, gy in gt_centres
        )
        matched_instances = sum(
            any(
                math.hypot(gx - px, gy - py) <= match_radius
                for px, py in instance
                for gx, gy in gt_centres
            )
            for instance in pred_instances
        )
        organ_recall = matched_gt / len(gt_centres) if gt_centres else float("nan")
        organ_precision = (
            matched_instances / len(pred_instances) if pred_instances else float("nan")
        )

        metrics.append({
            "label_file": label_path.relative_to(labels_root).as_posix(),
            "raw_file": raw_path.relative_to(raw_root).as_posix(),
            "sheet": sheet,
            "corolla_count": len(trait_rows),
            "organ_candidates": len(pred_instances),
            "organ_sample_points": len(pred_sample_points),
            "gt_green_objects": len(gt_centres),
            "gt_red_pixels": int(red.sum()),
            "gt_green_pixels": int(green.sum()),
            "red_boundary_recall": round(red_recall, 4) if not math.isnan(red_recall) else "",
            "pred_boundary_precision": round(boundary_precision, 4) if not math.isnan(boundary_precision) else "",
            "organ_recall_10mm": round(organ_recall, 4) if not math.isnan(organ_recall) else "",
            "organ_precision_10mm": round(organ_precision, 4) if not math.isnan(organ_precision) else "",
        })
        mapping.append({
            "label_file": label_path.relative_to(labels_root).as_posix(),
            "raw_file": raw_path.relative_to(raw_root).as_posix(),
        })

        overlay = raw.copy()
        contours, _ = cv2.findContours(predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 5)
        red_contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, red_contours, -1, (0, 0, 255), 3)
        cv2.drawContours(overlay, green_contours, -1, (0, 255, 0), 3)
        for instance in pred_instances:
            for x, y in instance:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 12, (255, 0, 255), -1)
        preview_scale = min(1.0, 1800.0 / max(overlay.shape[:2]))
        preview = cv2.resize(overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            str(overlay_dir / f"{normalise_stem(label_path.stem)}.jpg")
        )

    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "mapping.csv", mapping)
    numeric = lambda key: [float(row[key]) for row in metrics if row[key] not in (None, "")]
    summary = {
        "sheets": len(metrics),
        "mean_red_boundary_recall": round(float(np.mean(numeric("red_boundary_recall"))), 4),
        "mean_pred_boundary_precision": round(float(np.mean(numeric("pred_boundary_precision"))), 4),
        "mean_organ_recall_10mm": round(float(np.mean(numeric("organ_recall_10mm"))), 4),
        "mean_organ_precision_10mm": round(float(np.mean(numeric("organ_precision_10mm"))), 4),
    }
    (out / "SUMMARY.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
