# -*- coding: utf-8 -*-
"""Reviewed nectar-guide spot extraction and visual QC.

The reviewed detector uses two evidence levels:

- ``strong``: clearly magenta pixels or strong local a*-b* contrast;
- ``weak_recovered``: faint but locally distinct magenta pixels that the original
  conservative mask missed.

Both are restricted to the reviewed corolla mask. Single-pixel noise is rejected
using the ruler-calibrated area scale. Brown/degraded tissue remains a separate
class and is never silently relabelled as a guide.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review
import measure_guides_review_traits as visitor_traits

MIN_SPOT_AREA_MM2 = 0.01
LARGE_SPOT_CLUSTER_MM2 = 3.0


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    return median, 1.4826 * mad


def spot_candidate_masks(
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    corolla_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strong, weak-recovered, and combined guide masks."""
    corolla = corolla_mask.astype(bool)
    difference = (a_channel - b_channel).astype(np.float32)
    values = difference[corolla]
    a_values = a_channel[corolla]
    if values.size == 0:
        empty = np.zeros_like(corolla, dtype=np.uint8)
        return empty, empty, empty

    p_median, p_sigma = _robust_scale(values)
    a_median, a_sigma = _robust_scale(a_values)

    working = difference.copy()
    working[~corolla] = p_median
    local_background_21 = cv2.morphologyEx(
        working,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    local_background_9 = cv2.morphologyEx(
        working,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    contrast_21 = difference - local_background_21
    contrast_9 = difference - local_background_9

    strong = (
        (
            (difference > 5.0)
            | ((contrast_21 > 1.45) & (difference > 1.0))
            | ((difference > p_median + 2.8 * p_sigma) & (difference > 0.75))
        )
        & (a_channel > max(0.7, a_median + 0.8 * a_sigma))
        & corolla
    )

    weak = (
        (
            ((contrast_9 > 0.50) & (difference > 0.25))
            | ((contrast_21 > 0.80) & (difference > 0.35))
            | ((difference > p_median + 1.55 * p_sigma) & (difference > 0.35))
        )
        & (a_channel > max(0.9, a_median + 1.15 * a_sigma))
        & corolla
        & ~strong
    )

    # Do not use an opening here: a 2x2 opening erased genuine tiny guide dots.
    # Ruler-calibrated connected-component area filtering is less destructive.
    combined = (strong | weak).astype(np.uint8)
    return strong.astype(np.uint8), weak.astype(np.uint8), combined


def reviewed_spot_segment(
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    corolla_mask: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper returning the high-recall combined mask."""
    _, _, combined = spot_candidate_masks(a_channel, b_channel, corolla_mask)
    return combined


def _accepted_spots(
    combined: np.ndarray,
    strong: np.ndarray,
    weak: np.ndarray,
    mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    raw = combined.astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    accepted = np.zeros_like(raw, dtype=np.uint8)
    components: list[dict[str, object]] = []
    mm2_per_px2 = mm_per_px * mm_per_px

    for index in range(1, stats.shape[0]):
        component_mask = labels == index
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        area_mm2 = area_px * mm2_per_px2
        if area_mm2 < MIN_SPOT_AREA_MM2:
            continue
        strong_px = int((component_mask & (strong > 0)).sum())
        weak_px = int((component_mask & (weak > 0)).sum())
        if strong_px > 0 and weak_px > 0:
            detection_class = "mixed"
        elif strong_px > 0:
            detection_class = "strong"
        else:
            detection_class = "weak_recovered"
        accepted[component_mask] = 1
        components.append(
            {
                "label_index": index,
                "cx": float(centroids[index][0]),
                "cy": float(centroids[index][1]),
                "area_px": area_px,
                "area_mm2": float(area_mm2),
                "equivalent_diameter_mm": 2.0 * math.sqrt(area_mm2 / math.pi),
                "strong_px": strong_px,
                "weak_px": weak_px,
                "detection_class": detection_class,
            }
        )

    components.sort(key=lambda row: (float(row["cy"]), float(row["cx"])))
    return accepted, labels, components


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def _draw_legend(image: np.ndarray, top: int, *, binary: bool = False) -> None:
    y0 = min(max(top + 12, 12), image.shape[0] - 58)
    cv2.rectangle(image, (12, y0), (900, y0 + 44), (255, 255, 255), -1)
    cv2.rectangle(image, (12, y0), (900, y0 + 44), (80, 80, 80), 1)
    if binary:
        text = "black=strong guides  blue=weak recovered  orange=brown/degraded"
    else:
        text = "cyan=strong guides  blue=weak recovered  orange=brown/degraded"
    cv2.putText(image, text, (22, y0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)


def write_spot_qc(
    image_path: str | Path,
    folder: str,
    out_dir: str | Path,
    *,
    auto_split: bool = True,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    folder_key = folder.lower()
    stem = image_path.stem
    sheet_key = (folder_key, stem.lower())
    island, _ = base.ISLANDS.get(folder_key, (folder_key, ""))

    image = base.load_bgr(str(image_path))
    height, width = image.shape[:2]
    top = v2.specimen_top(image)
    scale_info = visitor_traits.calibrate_ruler(image, top)
    mm_per_px = float(scale_info["mm_per_px"])

    previous_sheet = review._CURRENT_SHEET
    review._CURRENT_SHEET = sheet_key
    try:
        filled, a_channel, b_channel = v2.foreground_v2(image, top)
        components = v2.corollas(filled, auto_split=auto_split)
    finally:
        review._CURRENT_SHEET = previous_sheet

    difference = (a_channel - b_channel).astype(np.float32)
    brown = (a_channel > 6) & (difference < -15)
    overlay = image.copy()
    mask_panel = np.full_like(image, 255)
    summaries: dict[int, dict[str, object]] = {}
    spot_rows: list[dict[str, object]] = []

    for corolla_id, component in enumerate(components, start=1):
        corolla = component["mask"].astype(bool)
        strong, weak, combined = spot_candidate_masks(a_channel, b_channel, corolla)
        accepted, component_labels, spot_components = _accepted_spots(
            combined, strong, weak, mm_per_px
        )
        accepted_bool = (accepted > 0) & corolla
        accepted_strong = accepted_bool & (strong > 0)
        accepted_weak = accepted_bool & (weak > 0)

        corolla_area_px = int(corolla.sum())
        total_area_px = int(accepted_bool.sum())
        strong_area_px = int(accepted_strong.sum())
        weak_area_px = int(accepted_weak.sum())
        mm2_per_px2 = mm_per_px * mm_per_px
        corolla_area_mm2 = corolla_area_px * mm2_per_px2
        total_area_mm2 = total_area_px * mm2_per_px2
        coverage_pct = 100.0 * total_area_px / max(corolla_area_px, 1)
        strong_cov_pct = 100.0 * strong_area_px / max(corolla_area_px, 1)
        weak_cov_pct = 100.0 * weak_area_px / max(corolla_area_px, 1)
        density_cm2 = len(spot_components) / max(corolla_area_mm2 / 100.0, 1e-9)
        brown_overlap_px = int((accepted_bool & brown).sum())
        brown_overlap_pct = 100.0 * brown_overlap_px / max(total_area_px, 1)
        areas = np.array([float(row["area_mm2"]) for row in spot_components], dtype=float)
        n_large_clusters = int((areas >= LARGE_SPOT_CLUSTER_MM2).sum()) if areas.size else 0
        n_strong = sum(row["detection_class"] in ("strong", "mixed") for row in spot_components)
        n_weak = sum(row["detection_class"] == "weak_recovered" for row in spot_components)

        summary = {
            "island": island,
            "sheet": stem,
            "corolla_id": corolla_id,
            "n_spots": len(spot_components),
            "n_strong_spots": n_strong,
            "n_weak_recovered_spots": n_weak,
            "n_large_spot_clusters": n_large_clusters,
            "guide_area_mm2": round(total_area_mm2, 3),
            "guide_cov_pct": round(coverage_pct, 3),
            "guide_cov_strong_pct": round(strong_cov_pct, 3),
            "guide_cov_weak_recovered_pct": round(weak_cov_pct, 3),
            "spot_density_cm2": round(density_cm2, 3),
            "mean_spot_area_mm2": round(float(areas.mean()), 4) if areas.size else 0.0,
            "median_spot_area_mm2": round(float(np.median(areas)), 4) if areas.size else 0.0,
            "max_spot_area_mm2": round(float(areas.max()), 4) if areas.size else 0.0,
            "brown_overlap_pct_of_spots": round(brown_overlap_pct, 3),
            "guide_present": int(coverage_pct >= 0.5),
            "spot_qc_source": "raw_scan_two_stage_reviewed_ruler_scale",
            **scale_info,
        }
        summaries[corolla_id] = summary

        for spot_id, spot in enumerate(spot_components, start=1):
            label_index = int(spot["label_index"])
            spot_mask = component_labels == label_index
            values = difference[spot_mask]
            brown_pixels = int((spot_mask & brown).sum())
            area_mm2 = float(spot["area_mm2"])
            spot_rows.append(
                {
                    "island": island,
                    "sheet": stem,
                    "corolla_id": corolla_id,
                    "spot_id": spot_id,
                    "cx": round(float(spot["cx"]), 2),
                    "cy": round(float(spot["cy"]), 2),
                    "area_mm2": round(area_mm2, 4),
                    "equivalent_diameter_mm": round(float(spot["equivalent_diameter_mm"]), 4),
                    "detection_class": spot["detection_class"],
                    "strong_pixel_fraction": round(
                        float(spot["strong_px"]) / max(float(spot["area_px"]), 1.0), 3
                    ),
                    "mean_a_minus_b": round(float(values.mean()), 3) if values.size else "",
                    "median_a_minus_b": round(float(np.median(values)), 3) if values.size else "",
                    "brown_overlap_pct": round(
                        100.0 * brown_pixels / max(int(spot_mask.sum()), 1), 3
                    ),
                    "large_cluster_check": "check" if area_mm2 >= LARGE_SPOT_CLUSTER_MM2 else "",
                    "exclude_FILL": "",
                }
            )

        colour_layer = overlay.copy()
        colour_layer[accepted_strong] = (255, 255, 0)
        colour_layer[accepted_weak] = (255, 80, 0)
        colour_layer[brown & corolla & ~accepted_bool] = (0, 140, 255)
        overlay = cv2.addWeighted(overlay, 0.68, colour_layer, 0.32, 0)
        contours, _ = cv2.findContours(corolla.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

        mask_panel[corolla] = (230, 230, 230)
        mask_panel[accepted_strong] = (0, 0, 0)
        mask_panel[accepted_weak] = (255, 80, 0)
        mask_panel[brown & corolla & ~accepted_bool] = (0, 140, 255)
        cv2.drawContours(mask_panel, contours, -1, (0, 180, 0), 2)

        x = int(round(component["cx"]))
        y = int(round(component["cy"]))
        label = f"C{corolla_id} cov={coverage_pct:.1f}% n={len(spot_components)} +weak={n_weak}"
        cv2.putText(
            overlay, label, (max(2, x - 145), max(top + 22, y - 65)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.53, (180, 0, 180), 2,
        )
        cv2.putText(
            mask_panel, f"C{corolla_id}", (x - 18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 0, 180), 2,
        )

    cv2.line(overlay, (0, top), (width - 1, top), (180, 0, 180), 2)
    cv2.line(mask_panel, (0, top), (width - 1, top), (180, 0, 180), 2)
    _draw_legend(overlay, top, binary=False)
    _draw_legend(mask_panel, top, binary=True)
    scale = min(1.0, 1900.0 / max(height, width))
    output_size = (int(round(width * scale)), int(round(height * scale)))
    overlay = cv2.resize(overlay, output_size)
    mask_panel = cv2.resize(mask_panel, output_size, interpolation=cv2.INTER_NEAREST)

    _write_png(out_dir / "spot_overlays" / f"{island}_{stem}_spots.png", overlay)
    _write_png(out_dir / "spot_masks" / f"{island}_{stem}_spot_mask.png", mask_panel)
    return summaries, spot_rows
