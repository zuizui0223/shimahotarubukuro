# -*- coding: utf-8 -*-
"""Ruler calibration and pollinator-relevant planar floral traits.

The scans contain a real centimetre ruler, so metric traits are calibrated from
its long centimetre ticks rather than relying only on JPEG DPI metadata.

Traits derived from the flattened corolla are explicitly named ``flat_*`` or
``prov_*``. Corolla length, maximum span, and area are robust planar measurements.
Throat span, tube length, mouth diameter, and entrance area are biologically useful
proxies but remain provisional because flattening/folding changes three-dimensional
geometry. Definitive 3-D mouth diameter, tube depth, herkogamy, and stamen position
still require fresh flowers or calibrated lateral photographs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review

RULER_MIN_PX_PER_CM = 90.0
RULER_MAX_PX_PER_CM = 145.0
RULER_MAX_CV = 0.035


def calibrate_ruler(image: np.ndarray, specimen_top: int) -> dict[str, object]:
    """Estimate mm/px from the long 1-cm ticks of the ruler in the header.

    Falls back to the verified 300-DPI scale when ruler detection is not reliable.
    The returned diagnostics make the fallback explicit rather than silent.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    header_end = max(1, min(specimen_top, int(round(height * 0.48))))
    header = gray[:header_end]

    y_start = min(max(int(round(height * 0.08)), 0), header_end - 1)
    dark = header < 125
    row_score = dark.sum(axis=1).astype(float)
    row_score[:y_start] = 0
    baseline_y = int(np.argmax(row_score))

    band_top = max(0, baseline_y - max(90, int(round(height * 0.055))))
    band_bottom = max(band_top + 1, baseline_y - 4)
    band = gray[band_top:band_bottom]
    binary = (band < 135).astype(np.uint8) * 255
    vertical_kernel_height = max(19, int(round((band_bottom - band_top) * 0.18)))
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height)),
    )

    n, _, stats, centroids = cv2.connectedComponentsWithStats(vertical, 8)
    min_long_height = max(45, int(round((band_bottom - band_top) * 0.50)))
    centres: list[float] = []
    for index in range(1, n):
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        if component_height < min_long_height or component_width > 12:
            continue
        centres.append(float(centroids[index][0]))

    centres.sort()
    deduplicated: list[float] = []
    for x in centres:
        if not deduplicated or x - deduplicated[-1] > 5.0:
            deduplicated.append(x)
        else:
            deduplicated[-1] = (deduplicated[-1] + x) / 2.0

    diffs = np.diff(np.asarray(deduplicated, dtype=float)) if len(deduplicated) >= 2 else np.array([])
    plausible = diffs[(diffs >= RULER_MIN_PX_PER_CM * 0.70) & (diffs <= RULER_MAX_PX_PER_CM * 1.30)]
    if plausible.size:
        initial = float(np.median(plausible))
        regular = plausible[(plausible >= initial * 0.82) & (plausible <= initial * 1.18)]
    else:
        regular = np.array([])

    if regular.size:
        px_per_cm = float(np.median(regular))
        spacing_cv = float(np.std(regular) / max(np.mean(regular), 1e-9))
    else:
        px_per_cm = float("nan")
        spacing_cv = float("nan")

    ruler_ok = (
        len(deduplicated) >= 6
        and regular.size >= 5
        and RULER_MIN_PX_PER_CM <= px_per_cm <= RULER_MAX_PX_PER_CM
        and spacing_cv <= RULER_MAX_CV
    )
    if ruler_ok:
        mm_per_px = 10.0 / px_per_cm
        source = "ruler_1cm_ticks"
        qc = "ok"
    else:
        mm_per_px = base.MM_PX
        px_per_cm = 10.0 / mm_per_px
        source = "fallback_verified_300dpi"
        qc = "check"

    return {
        "scale_source": source,
        "scale_qc": qc,
        "mm_per_px": round(float(mm_per_px), 7),
        "px_per_cm": round(float(px_per_cm), 4),
        "ruler_spacing_cv": round(float(spacing_cv), 5) if math.isfinite(spacing_cv) else "",
        "ruler_tick_count": len(deduplicated),
        "ruler_regular_interval_count": int(regular.size),
        "ruler_baseline_y": baseline_y,
        "ruler_tick_x": "|".join(f"{x:.2f}" for x in deduplicated),
    }


def _rotate_vertical(mask: np.ndarray, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    axis = vt[0]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    matrix = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle - 90.0, 1.0)
    height, width = mask.shape
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)
    rotated_spots = cv2.warpAffine(spots.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)
    ry, rx = np.where(rotated_mask > 0)
    rotated_mask = rotated_mask[ry.min():ry.max() + 1, rx.min():rx.max() + 1]
    rotated_spots = rotated_spots[ry.min():ry.max() + 1, rx.min():rx.max() + 1]
    return rotated_mask, rotated_spots


def _row_runs(row: np.ndarray) -> int:
    values = row.astype(np.uint8)
    return int(np.sum((values[1:] > 0) & (values[:-1] == 0)) + (values[0] > 0))


def _end_complexity(mask: np.ndarray, start: int, stop: int) -> float:
    section = mask[start:stop]
    if section.size == 0:
        return 0.0
    widths = section.sum(axis=1).astype(float)
    occupied = widths > 0
    if not np.any(occupied):
        return 0.0
    widths = widths[occupied]
    run_counts = np.array([_row_runs(row) for row in section[occupied]], dtype=float)
    width_cv = float(np.std(widths) / max(np.mean(widths), 1e-9))
    return float(np.mean(run_counts) + 1.5 * width_cv)


def orient_tip_top_base_bottom(mask: np.ndarray, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Orient the lobe/tip end upward and the basal tube end downward."""
    rotated_mask, rotated_spots = _rotate_vertical(mask, spots)
    height = rotated_mask.shape[0]
    end = max(3, int(round(height * 0.30)))
    top_score = _end_complexity(rotated_mask, 0, end)
    bottom_score = _end_complexity(rotated_mask, height - end, height)
    total = max(top_score + bottom_score, 1e-9)
    confidence = abs(top_score - bottom_score) / total
    source = "shape_end_complexity"

    # Lobe end should be at the top. When shape is ambiguous, guide-rich tissue is
    # used as a secondary cue because nectar guides concentrate toward the tube base.
    flip = bottom_score > top_score
    if confidence < 0.08 and int(rotated_spots.sum()) >= 10:
        top_density = float(rotated_spots[:end].sum()) / max(float(rotated_mask[:end].sum()), 1.0)
        bottom_density = float(rotated_spots[-end:].sum()) / max(float(rotated_mask[-end:].sum()), 1.0)
        flip = top_density > bottom_density
        confidence = abs(top_density - bottom_density) / max(top_density + bottom_density, 1e-9)
        source = "guide_density_secondary"

    if flip:
        rotated_mask = rotated_mask[::-1]
        rotated_spots = rotated_spots[::-1]
    return rotated_mask, rotated_spots, float(confidence), source


def _sinus_row(mask: np.ndarray) -> tuple[float, int, str]:
    """Estimate the average lobe-sinus row from the upper boundary profile."""
    height, width = mask.shape
    column_top = np.full(width, -1, dtype=float)
    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size:
            column_top[x] = float(ys.min())
    profile = column_top[column_top >= 0]
    if profile.size < 12:
        return float(height * 0.40), 1, "fallback"
    kernel = max(3, (int(profile.size) // 25) | 1)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (kernel, 1), 0).ravel()
    span = float(profile.max() - profile.min())
    tips = [
        i for i in range(1, len(profile) - 1)
        if profile[i] <= profile[i - 1]
        and profile[i] < profile[i + 1]
        and profile.max() - profile[i] > 0.22 * max(span, 1e-6)
    ]
    sinuses = [
        i for i in range(1, len(profile) - 1)
        if profile[i] >= profile[i - 1]
        and profile[i] > profile[i + 1]
        and profile[i] - profile.min() > 0.22 * max(span, 1e-6)
    ]
    if sinuses:
        value = float(np.median([profile[i] for i in sinuses]))
        source = "boundary_sinuses"
    else:
        value = float(profile.min() + 0.42 * max(span, 1.0))
        source = "profile_fallback"
    value = min(max(value, height * 0.12), height * 0.70)
    return value, max(len(tips), 1), source


def _median_width(mask: np.ndarray, centre: float, half_window: int = 3) -> float:
    height = mask.shape[0]
    start = max(0, int(round(centre)) - half_window)
    stop = min(height, int(round(centre)) + half_window + 1)
    widths = mask[start:stop].sum(axis=1).astype(float)
    widths = widths[widths > 0]
    return float(np.median(widths)) if widths.size else 0.0


def measure_flat_traits(
    corolla: np.ndarray,
    spots: np.ndarray,
    mm_per_px: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, float]]:
    oriented, oriented_spots, orientation_confidence, orientation_source = orient_tip_top_base_bottom(
        corolla.astype(bool), spots.astype(bool)
    )
    height, width = oriented.shape
    row_widths = oriented.sum(axis=1).astype(float)
    length_px = float(np.count_nonzero(row_widths > 0))
    maximum_width_px = float(np.percentile(row_widths[row_widths > 0], 95))
    area_px = int(oriented.sum())

    sinus_y, n_lobes, sinus_source = _sinus_row(oriented)
    throat_span_px = _median_width(oriented, sinus_y, 3)
    tube_length_px = max(0.0, (height - 1.0) - sinus_y)
    middle_y = sinus_y + 0.50 * tube_length_px
    basal_y = sinus_y + 0.85 * tube_length_px
    mid_tube_width_px = _median_width(oriented, middle_y, 4)
    basal_tube_width_px = _median_width(oriented, basal_y, 4)
    lobe_length_px = sinus_y

    length_mm = length_px * mm_per_px
    maximum_width_mm = maximum_width_px * mm_per_px
    throat_span_mm = throat_span_px * mm_per_px
    tube_length_mm = tube_length_px * mm_per_px
    mid_tube_width_mm = mid_tube_width_px * mm_per_px
    basal_tube_width_mm = basal_tube_width_px * mm_per_px
    lobe_length_mm = lobe_length_px * mm_per_px
    area_mm2 = area_px * mm_per_px * mm_per_px
    width_length_ratio = maximum_width_mm / max(length_mm, 1e-9)

    folded_half = width_length_ratio < 0.55
    circumference_proxy_mm = throat_span_mm * (2.0 if folded_half else 1.0)
    mouth_diameter_mm = circumference_proxy_mm / math.pi
    entrance_area_mm2 = math.pi * (mouth_diameter_mm / 2.0) ** 2
    taper_ratio = throat_span_mm / max(basal_tube_width_mm, 1e-9)
    tube_slenderness = tube_length_mm / max(mid_tube_width_mm, 1e-9)

    tube_start = min(max(int(round(sinus_y)), 0), height - 1)
    tube_mask = oriented[tube_start:] > 0
    lobe_mask = oriented[:tube_start] > 0
    tube_spots = oriented_spots[tube_start:] > 0
    lobe_spots = oriented_spots[:tube_start] > 0
    guide_cov_tube_pct = 100.0 * float((tube_spots & tube_mask).sum()) / max(float(tube_mask.sum()), 1.0)
    guide_cov_lobes_pct = 100.0 * float((lobe_spots & lobe_mask).sum()) / max(float(lobe_mask.sum()), 1.0)

    trait_qc: list[str] = []
    if orientation_confidence < 0.08:
        trait_qc.append("orientation")
    if sinus_source != "boundary_sinuses":
        trait_qc.append("sinus")
    if folded_half:
        trait_qc.append("folded_mouth_proxy")

    traits = {
        "corolla_length_ruler_mm": round(length_mm, 3),
        "corolla_max_span_ruler_mm": round(maximum_width_mm, 3),
        "corolla_area_ruler_mm2": round(area_mm2, 3),
        "flat_lobe_length_mm": round(lobe_length_mm, 3),
        "flat_tube_length_mm": round(tube_length_mm, 3),
        "flat_throat_span_mm": round(throat_span_mm, 3),
        "flat_mid_tube_width_mm": round(mid_tube_width_mm, 3),
        "flat_basal_tube_width_mm": round(basal_tube_width_mm, 3),
        "prov_mouth_diameter_ruler_mm": round(mouth_diameter_mm, 3),
        "prov_entrance_area_mm2": round(entrance_area_mm2, 3),
        "mouth_to_tube_length_ratio": round(mouth_diameter_mm / max(tube_length_mm, 1e-9), 4),
        "flat_tube_taper_ratio": round(taper_ratio, 4),
        "flat_tube_slenderness": round(tube_slenderness, 4),
        "guide_cov_tube_pct": round(guide_cov_tube_pct, 3),
        "guide_cov_lobes_pct": round(guide_cov_lobes_pct, 3),
        "flat_n_lobes": int(n_lobes),
        "fold_state_auto": "folded_half" if folded_half else "opened_or_broad",
        "mouth_proxy_assumption": "2x_flat_span_over_pi" if folded_half else "flat_span_over_pi",
        "orientation_confidence": round(orientation_confidence, 4),
        "orientation_source": orientation_source,
        "sinus_source": sinus_source,
        "visitor_trait_qc": "|".join(trait_qc),
    }
    guides = {
        "sinus_y": float(sinus_y),
        "middle_y": float(middle_y),
        "basal_y": float(basal_y),
    }
    return traits, oriented, oriented_spots, guides


def _panel(
    corolla_id: int,
    mask: np.ndarray,
    spots: np.ndarray,
    guides: dict[str, float],
    traits: dict[str, object],
) -> np.ndarray:
    height, width = mask.shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[mask > 0] = (235, 235, 225)
    canvas[spots > 0] = (255, 255, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 180, 0), 2)
    colours = {"sinus_y": (180, 0, 180), "middle_y": (255, 0, 0), "basal_y": (0, 150, 0)}
    for key, colour in colours.items():
        y = min(max(int(round(guides[key])), 0), height - 1)
        cv2.line(canvas, (0, y), (width - 1, y), colour, 2)
    scale = min(1.0, 420.0 / max(height, width))
    panel = cv2.resize(canvas, (max(1, int(width * scale)), max(1, int(height * scale))))
    framed = np.full((500, 460, 3), 255, dtype=np.uint8)
    x0 = (460 - panel.shape[1]) // 2
    y0 = 65
    framed[y0:y0 + panel.shape[0], x0:x0 + panel.shape[1]] = panel
    cv2.putText(framed, f"C{corolla_id}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2)
    cv2.putText(
        framed,
        f"L={traits['corolla_length_ruler_mm']:.1f} tube={traits['flat_tube_length_mm']:.1f} mm",
        (68, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 1,
    )
    cv2.putText(
        framed,
        f"throat={traits['flat_throat_span_mm']:.1f} mouth~{traits['prov_mouth_diameter_ruler_mm']:.1f} mm",
        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1,
    )
    cv2.putText(
        framed,
        "magenta=sinus blue=mid-tube green=basal",
        (12, 488), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1,
    )
    return framed


def write_flower_trait_qc(
    image_path: str | Path,
    folder: str,
    out_dir: str | Path,
    spot_segmenter: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    auto_split: bool = True,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    folder_key = folder.lower()
    stem = image_path.stem
    island, _ = base.ISLANDS.get(folder_key, (folder_key, ""))
    image = base.load_bgr(str(image_path))
    top = v2.specimen_top(image)
    scale_info = calibrate_ruler(image, top)
    mm_per_px = float(scale_info["mm_per_px"])

    previous_sheet = review._CURRENT_SHEET
    review._CURRENT_SHEET = (folder_key, stem.lower())
    try:
        filled, a_channel, b_channel = v2.foreground_v2(image, top)
        components = v2.corollas(filled, auto_split=auto_split)
    finally:
        review._CURRENT_SHEET = previous_sheet

    summaries: dict[int, dict[str, object]] = {}
    panels: list[np.ndarray] = []
    for corolla_id, component in enumerate(components, start=1):
        corolla = component["mask"].astype(bool)
        spots = spot_segmenter(a_channel, b_channel, corolla).astype(bool)
        traits, oriented, oriented_spots, guides = measure_flat_traits(corolla, spots, mm_per_px)
        summary = {
            "island": island,
            "sheet": stem,
            "corolla_id": corolla_id,
            **scale_info,
            **traits,
        }
        summaries[corolla_id] = summary
        panels.append(_panel(corolla_id, oriented, oriented_spots, guides, traits))

    columns = 2
    rows = int(math.ceil(len(panels) / columns))
    sheet = np.full((rows * 500, columns * 460, 3), 245, dtype=np.uint8)
    for index, panel in enumerate(panels):
        y = (index // columns) * 500
        x = (index % columns) * 460
        sheet[y:y + 500, x:x + 460] = panel
    cv2.putText(
        sheet,
        f"scale={scale_info['mm_per_px']:.5f} mm/px ({scale_info['scale_source']})",
        (12, sheet.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
    )
    output = out_dir / "trait_overlays" / f"{island}_{stem}_visitor_traits.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", sheet)[1].tofile(str(output))
    return summaries, scale_info
