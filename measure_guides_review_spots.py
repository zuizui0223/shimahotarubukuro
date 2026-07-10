# -*- coding: utf-8 -*-
"""Reviewed nectar-guide spot extraction and visual QC.

This module recomputes spot masks from the raw scan after the reviewed corolla
corrections have been installed. It writes:

- one spot-overlay image per sheet;
- one binary diagnostic mask per sheet;
- a corolla-level spot summary;
- one row per accepted spot component.

Small connected components below ``MIN_SPOT_AREA_MM2`` are treated as scan noise.
Brown/degraded tissue is measured independently and is not silently relabelled as
a nectar-guide spot.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review

MIN_SPOT_AREA_MM2 = 0.02


def _accepted_spots(raw_spots: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Remove tiny connected components and return accepted component metrics."""
    raw = raw_spots.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    accepted = np.zeros_like(raw, dtype=np.uint8)
    components: list[dict[str, float]] = []

    for index in range(1, n):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        area_mm2 = area_px * base.MM2_PX
        if area_mm2 < MIN_SPOT_AREA_MM2:
            continue
        accepted[labels == index] = 1
        equivalent_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
        components.append(
            {
                "label_index": float(index),
                "cx": float(centroids[index][0]),
                "cy": float(centroids[index][1]),
                "area_px": float(area_px),
                "area_mm2": float(area_mm2),
                "equivalent_diameter_mm": float(equivalent_diameter_mm),
            }
        )

    components.sort(key=lambda row: (row["cy"], row["cx"]))
    return accepted, components


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def write_spot_qc(
    image_path: str | Path,
    folder: str,
    out_dir: str | Path,
    *,
    auto_split: bool = True,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Write reviewed spot outputs and return corolla summaries and spot rows."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    folder_key = folder.lower()
    stem = image_path.stem
    sheet_key = (folder_key, stem.lower())
    island, _ = base.ISLANDS.get(folder_key, (folder_key, ""))

    image = base.load_bgr(str(image_path))
    height, width = image.shape[:2]
    top = v2.specimen_top(image)

    previous_sheet = review._CURRENT_SHEET
    review._CURRENT_SHEET = sheet_key
    try:
        filled, a_channel, b_channel = v2.foreground_v2(image, top)
        components = v2.corollas(filled, auto_split=auto_split)
    finally:
        review._CURRENT_SHEET = previous_sheet

    brown = (a_channel > 6) & ((a_channel - b_channel) < -15)
    overlay = image.copy()
    mask_panel = np.full_like(image, 255)
    summaries: dict[int, dict[str, object]] = {}
    spot_rows: list[dict[str, object]] = []

    for corolla_id, component in enumerate(components, start=1):
        corolla = component["mask"].astype(bool)
        raw_spots = base.spot_segment(a_channel, b_channel, corolla)
        accepted, spot_components = _accepted_spots(raw_spots)
        accepted_bool = accepted.astype(bool) & corolla
        corolla_area_px = int(corolla.sum())
        accepted_area_px = int(accepted_bool.sum())
        corolla_area_mm2 = corolla_area_px * base.MM2_PX
        accepted_area_mm2 = accepted_area_px * base.MM2_PX
        coverage_pct = 100.0 * accepted_area_px / max(corolla_area_px, 1)
        density_cm2 = len(spot_components) / max(corolla_area_mm2 / 100.0, 1e-9)
        brown_overlap_px = int((accepted_bool & brown).sum())
        brown_overlap_pct = 100.0 * brown_overlap_px / max(accepted_area_px, 1)
        areas = np.array([row["area_mm2"] for row in spot_components], dtype=float)

        summary = {
            "corolla_id": corolla_id,
            "n_spots": len(spot_components),
            "guide_area_mm2": round(accepted_area_mm2, 3),
            "guide_cov_pct": round(coverage_pct, 3),
            "spot_density_cm2": round(density_cm2, 3),
            "mean_spot_area_mm2": round(float(areas.mean()), 4) if areas.size else 0.0,
            "median_spot_area_mm2": round(float(np.median(areas)), 4) if areas.size else 0.0,
            "max_spot_area_mm2": round(float(areas.max()), 4) if areas.size else 0.0,
            "brown_overlap_pct_of_spots": round(brown_overlap_pct, 3),
            "guide_present": int(coverage_pct >= 0.5),
            "spot_qc_source": "raw_scan_reviewed_corolla",
        }
        summaries[corolla_id] = summary

        difference = (a_channel - b_channel).astype(np.float32)
        for spot_id, spot in enumerate(spot_components, start=1):
            label_index = int(spot["label_index"])
            # Reconstruct this spot from its seed location within the accepted mask.
            seed_x = min(max(int(round(spot["cx"])), 0), width - 1)
            seed_y = min(max(int(round(spot["cy"])), 0), height - 1)
            # Connected-components is recomputed on the accepted mask to avoid using
            # labels from components that were rejected as tiny noise.
            n_acc, labels_acc, _, _ = cv2.connectedComponentsWithStats(accepted, 8)
            accepted_label = int(labels_acc[seed_y, seed_x]) if n_acc > 1 else 0
            spot_mask = labels_acc == accepted_label if accepted_label > 0 else np.zeros_like(accepted, dtype=bool)
            values = difference[spot_mask]
            brown_pixels = int((spot_mask & brown).sum())
            spot_rows.append(
                {
                    "island": island,
                    "sheet": stem,
                    "corolla_id": corolla_id,
                    "spot_id": spot_id,
                    "cx": round(spot["cx"], 2),
                    "cy": round(spot["cy"], 2),
                    "area_mm2": round(spot["area_mm2"], 4),
                    "equivalent_diameter_mm": round(spot["equivalent_diameter_mm"], 4),
                    "mean_a_minus_b": round(float(values.mean()), 3) if values.size else "",
                    "median_a_minus_b": round(float(np.median(values)), 3) if values.size else "",
                    "brown_overlap_pct": round(100.0 * brown_pixels / max(int(spot_mask.sum()), 1), 3),
                    "exclude_FILL": "",
                }
            )

        # Raw-image overlay: cyan accepted spots, orange brown tissue, green corolla.
        colour_layer = overlay.copy()
        colour_layer[accepted_bool] = (255, 255, 0)
        colour_layer[brown & corolla & ~accepted_bool] = (0, 140, 255)
        overlay = cv2.addWeighted(overlay, 0.70, colour_layer, 0.30, 0)

        contours, _ = cv2.findContours(corolla.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
        spot_contours, _ = cv2.findContours(accepted.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, spot_contours, -1, (255, 255, 0), 1)

        mask_panel[corolla] = (230, 230, 230)
        mask_panel[accepted_bool] = (0, 0, 0)
        mask_panel[brown & corolla & ~accepted_bool] = (0, 140, 255)
        cv2.drawContours(mask_panel, contours, -1, (0, 180, 0), 2)

        x = int(round(component["cx"]))
        y = int(round(component["cy"]))
        label = f"C{corolla_id} cov={coverage_pct:.1f}% n={len(spot_components)}"
        cv2.putText(
            overlay,
            label,
            (max(2, x - 120), max(top + 22, y - 65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (180, 0, 180),
            2,
        )
        cv2.putText(
            mask_panel,
            f"C{corolla_id}",
            (x - 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 0, 180),
            2,
        )

    cv2.line(overlay, (0, top), (width - 1, top), (180, 0, 180), 2)
    cv2.line(mask_panel, (0, top), (width - 1, top), (180, 0, 180), 2)
    scale = min(1.0, 1900.0 / max(height, width))
    output_size = (int(round(width * scale)), int(round(height * scale)))
    overlay = cv2.resize(overlay, output_size)
    mask_panel = cv2.resize(mask_panel, output_size, interpolation=cv2.INTER_NEAREST)

    _write_png(out_dir / "spot_overlays" / f"{island}_{stem}_spots.png", overlay)
    _write_png(out_dir / "spot_masks" / f"{island}_{stem}_spot_mask.png", mask_panel)
    return summaries, spot_rows
