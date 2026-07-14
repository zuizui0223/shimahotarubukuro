# -*- coding: utf-8 -*-
"""Compare v3 outputs with reviewed red/green shimask annotations.

The annotated previews are never used by the extractor. Red strokes are treated
as reviewed corolla boundaries and green strokes as reviewed organ locations.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
from evaluate_shimask_labels import IMAGE_SUFFIXES, label_masks, load_bgr, write_csv


def normalise_stem(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = text.replace("niijiama", "niijima")
    text = text.replace("～", "~").replace("-", "~")
    return re.sub(r"[^a-z0-9~()]", "", text)


def find_raw(label_path: Path, raw_root: Path) -> Path:
    target = normalise_stem(label_path.stem)
    candidates = [
        path for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    exact = [path for path in candidates if normalise_stem(path.stem) == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        exact.sort(key=lambda path: (len(path.parts), path.as_posix()))
        return exact[0]

    # Conservative fallback: require matching island/name prefix and number set.
    prefix = re.match(r"[a-z]+", target)
    numbers = re.findall(r"\d+", target)
    ranked: list[tuple[int, Path]] = []
    for path in candidates:
        stem = normalise_stem(path.stem)
        score = 0
        if prefix and stem.startswith(prefix.group(0)):
            score += 10
        if re.findall(r"\d+", stem) == numbers:
            score += 20
        score -= abs(len(stem) - len(target))
        ranked.append((score, path))
    ranked.sort(key=lambda item: (item[0], item[1].as_posix()), reverse=True)
    if not ranked or ranked[0][0] < 20:
        raise FileNotFoundError(f"No raw scan match for {label_path}")
    return ranked[0][1]


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def union_masks(mask_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, np.uint8)
    for path in sorted(mask_dir.glob("C*.png")):
        mask = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        union[mask > 0] = 1
    return union


def boundary(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def overlap_with_tolerance(reference: np.ndarray, prediction: np.ndarray, radius: int) -> float:
    selected = reference > 0
    if not selected.any():
        return float("nan")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    expanded = cv2.dilate(prediction.astype(np.uint8), kernel) > 0
    return float(expanded[selected].mean())


def green_centres(mask: np.ndarray) -> list[tuple[float, float]]:
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    centres: list[tuple[float, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        length = max(int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]))
        if area < 20 or length < 12:
            continue
        centres.append((float(centroids[label][0]), float(centroids[label][1])))
    return centres


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
        red_small, green_small = label_masks(annotated)
        red = resize_mask(red_small, raw.shape[:2])
        green = resize_mask(green_small, raw.shape[:2])

        sheet = raw_path.stem
        trait_rows = [row for row in traits if row.get("sheet") == sheet]
        organ_rows = [row for row in organs if row.get("sheet") == sheet]
        if not trait_rows:
            raise RuntimeError(f"No prediction rows for matched raw scan {raw_path}")
        island = trait_rows[0]["island"]
        mask_dir = predictions / "masks" / island / sheet
        predicted = union_masks(mask_dir, raw.shape[:2])
        pred_boundary = boundary(predicted)

        scale = raw.shape[1] / max(annotated.shape[1], 1)
        tolerance = max(3, int(round(5.0 * scale)))
        red_recall = overlap_with_tolerance(red, pred_boundary, tolerance)
        boundary_precision = overlap_with_tolerance(pred_boundary, red, tolerance)

        gt_centres = green_centres(green)
        pred_centres = [
            (float(row["cx"]), float(row["cy"]))
            for row in organ_rows
            if row.get("cx") not in (None, "") and row.get("cy") not in (None, "")
        ]
        match_radius = 10.0 / float(base.MM_PX)
        matched_gt = sum(
            any(math.hypot(gx - px, gy - py) <= match_radius for px, py in pred_centres)
            for gx, gy in gt_centres
        )
        matched_pred = sum(
            any(math.hypot(gx - px, gy - py) <= match_radius for gx, gy in gt_centres)
            for px, py in pred_centres
        )
        organ_recall = matched_gt / len(gt_centres) if gt_centres else float("nan")
        organ_precision = matched_pred / len(pred_centres) if pred_centres else float("nan")

        metrics.append(
            {
                "label_file": label_path.relative_to(labels_root).as_posix(),
                "raw_file": raw_path.relative_to(raw_root).as_posix(),
                "sheet": sheet,
                "corolla_count": len(trait_rows),
                "organ_candidates": len(pred_centres),
                "gt_green_objects": len(gt_centres),
                "red_boundary_recall": round(red_recall, 4),
                "pred_boundary_precision": round(boundary_precision, 4),
                "organ_recall_10mm": round(organ_recall, 4) if not math.isnan(organ_recall) else "",
                "organ_precision_10mm": round(organ_precision, 4) if not math.isnan(organ_precision) else "",
            }
        )
        mapping.append(
            {
                "label_file": label_path.relative_to(labels_root).as_posix(),
                "raw_file": raw_path.relative_to(raw_root).as_posix(),
            }
        )

        overlay = raw.copy()
        contours, _ = cv2.findContours(predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 5)
        red_contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, red_contours, -1, (0, 0, 255), 3)
        cv2.drawContours(overlay, green_contours, -1, (0, 255, 0), 3)
        for x, y in pred_centres:
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
