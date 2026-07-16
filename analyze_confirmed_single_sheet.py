#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure one reviewed sheet directly from red and green annotations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
from evaluate_shimask_labels import load_bgr
from measure_confirmed_shimask_traits import (
    _confirmed_organs,
    confirmed_corolla_masks,
    simple_corolla_metrics,
)
from measure_guides_review_spots import spot_candidate_masks
from measure_guides_review_traits import calibrate_ruler
from shimask_annotation_diff import annotation_masks


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _lab_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    _light, a, b = cv2.split(lab)
    return a - 128.0, b - 128.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shimask", type=Path, required=True)
    parser.add_argument("--folder", default="oshima")
    parser.add_argument("--out-dir", type=Path, default=Path("results_confirmed_single"))
    args = parser.parse_args()

    raw = load_bgr(args.raw)
    annotated = load_bgr(args.shimask)
    red_small, green_small = annotation_masks(annotated, raw)
    red = cv2.resize(red_small.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    green = cv2.resize(green_small.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)

    masks = confirmed_corolla_masks(red)
    if not masks:
        raise RuntimeError("No closed red corolla outlines were recovered")

    scale_info = calibrate_ruler(raw, v2.specimen_top(raw))
    mm_per_px = float(scale_info["mm_per_px"])
    island = base.ISLANDS.get(args.folder.lower(), (args.folder, ""))[0]
    a_channel, b_channel = _lab_channels(raw)

    trait_rows: list[dict] = []
    spot_rows: list[dict] = []
    overlay = raw.copy()

    for corolla_id, mask in enumerate(masks, start=1):
        corolla = mask.astype(bool)
        strong, weak, guide = spot_candidate_masks(a_channel, b_channel, corolla)
        guide = guide.astype(bool) & corolla
        guide_px = int(guide.sum())
        area_px = int(corolla.sum())
        metrics = simple_corolla_metrics(mask, mm_per_px)
        trait_rows.append({
            "island": island,
            "sheet": args.raw.stem,
            "corolla_id": corolla_id,
            "mask_source": "human_confirmed_red_outline",
            **scale_info,
            **metrics,
            "guide_area_px": guide_px,
            "guide_area_mm2": round(guide_px * mm_per_px * mm_per_px, 3),
            "guide_cov_pct": round(100.0 * guide_px / max(area_px, 1), 3),
            "guide_present": int(guide_px > 0),
            "strong_guide_px": int((strong.astype(bool) & corolla).sum()),
            "weak_guide_px": int((weak.astype(bool) & corolla).sum()),
            "provenance": "shimask_human_review_plus_raw_scan_colour",
        })
        spot_rows.append({
            "island": island,
            "sheet": args.raw.stem,
            "corolla_id": corolla_id,
            "guide_area_px": guide_px,
            "guide_area_mm2": round(guide_px * mm_per_px * mm_per_px, 3),
            "guide_cov_pct": round(100.0 * guide_px / max(area_px, 1), 3),
            "guide_present": int(guide_px > 0),
        })
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cv2.drawContours(overlay, [max(contours, key=cv2.contourArea)], -1, (0, 0, 255), 4)
        overlay[guide] = (255, 0, 255)

    organ_rows = _confirmed_organs(green, mm_per_px, args.raw.stem, island)
    if not organ_rows:
        raise RuntimeError("No green reproductive-organ traces were recovered")
    overlay[green > 0] = (0, 255, 0)

    _write_csv(args.out_dir / "confirmed_reviewed_traits.csv", trait_rows)
    _write_csv(args.out_dir / "confirmed_reviewed_spots.csv", spot_rows)
    _write_csv(args.out_dir / "confirmed_reproductive_organs.csv", organ_rows)
    _write_csv(args.out_dir / "scale_calibration.csv", [scale_info])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError("Could not encode annotation overlay")
    encoded.tofile(str(args.out_dir / "annotation_overlay_red_mask_green_organs.jpg"))

    (args.out_dir / "PROVENANCE.txt").write_text(
        "red_closed_outline=filled_directly_as_corolla_mask\n"
        "green_trace=skeletonized_and_measured_directly_without_morphological_closing\n"
        "inferred_lobes_tube_throat_mouth=not_measured\n"
        "raw_scan=ruler_calibration_and_nectar_guide_colour_only\n",
        encoding="utf-8",
    )
    print(
        f"confirmed_corollas={len(trait_rows)} confirmed_organs={len(organ_rows)} "
        f"scale={mm_per_px} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
