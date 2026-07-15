# -*- coding: utf-8 -*-
"""Extend refine13 with biologically useful floral measurements.

The reviewed shimask images remain evaluation-only. Runtime measurements use the
raw scan and the automatically extracted masks/components.

Adds:
- corolla maximum width and width/length ratio
- basal width, mid-tube width and throat/opening width
- tube length, lobe length and provisional lobe count
- explicit measurement status fields
- organ instance IDs so multiple evaluation points on one organ are not counted
  as multiple stamens/pistils
- conservative organ type candidates (never a forced biological identity)
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13

MM_PX = float(base.MM_PX)

_ORIGINAL_RECOMPUTE = refine._recompute_traits
_ORIGINAL_DETECT_ORGANS = refine13.detect_organs


def _median_positive(values: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)
    selected = selected[selected > 0]
    return float(np.median(selected)) if selected.size else 0.0


def corolla_shape_traits(mask: np.ndarray) -> dict:
    """Measure flattened-corolla traits after base-to-tip standardisation.

    The mask is oriented with the guide-rich/base end at the bottom. Tube length
    is the distance from the basal end to the mean lobe sinus estimated by the
    existing geometry routine. Widths are cross-sectional mask spans.
    """
    q = np.asarray(mask) > 0
    if int(q.sum()) < 20:
        return {
            "basal_width_mm": "", "basal_width_status": "not_measurable",
            "tube_width_mid_mm": "", "tube_width_status": "not_measurable",
            "throat_width_mm": "", "throat_width_status": "not_measurable",
            "tube_length_mm": "", "tube_length_status": "not_measurable",
            "lobe_length_mm": "", "lobe_length_status": "not_measurable",
            "lobe_count_auto": "", "lobe_count_status": "not_measurable",
            "width_length_ratio": "",
        }

    oriented, _ = base.orient_base_tip(q, np.zeros_like(q, dtype=bool))
    height = int(oriented.shape[0])
    widths = oriented.sum(axis=1).astype(float)
    geometry = base.geometry(oriented)

    tube_depth_px = float(np.clip(geometry.get("tube_depth", 0.0), 0.0, max(height - 1, 0)))
    sinus_y = float(height - 1) - tube_depth_px
    throat_row = int(np.clip(round(sinus_y), 0, height - 1))

    # The basal edge can be ragged, so use the 8--25% band from the bottom.
    b0 = int(round(height * 0.75))
    b1 = max(b0 + 1, int(round(height * 0.92)))
    basal_width_px = _median_positive(widths[b0:b1])

    # Mid-tube cross-section, restricted to the tubular region below the sinus.
    tube_start = int(np.clip(round(sinus_y), 0, height - 1))
    tube_mid = int(round((tube_start + height - 1) / 2.0))
    half_band = max(1, int(round(height * 0.025)))
    mid_width_px = _median_positive(widths[max(tube_start, tube_mid - half_band):min(height, tube_mid + half_band + 1)])

    throat_width_px = float(widths[throat_row]) if 0 <= throat_row < height else 0.0
    if throat_width_px <= 0:
        throat_width_px = float(geometry.get("throat_width", 0.0))

    total_length_mm = height * MM_PX
    tube_length_mm = tube_depth_px * MM_PX
    lobe_length_mm = max(0.0, total_length_mm - tube_length_mm)
    max_width_mm = float(widths.max()) * MM_PX if widths.size else 0.0

    # Flattening/folding makes aperture-like traits less reliable than area/length.
    reliable_profile = height >= 20 and throat_width_px > 0 and basal_width_px > 0
    profile_status = "automatic_provisional" if reliable_profile else "not_measurable"
    tube_status = "automatic_provisional" if reliable_profile and 0.18 * height <= tube_depth_px <= 0.92 * height else "review_required"

    return {
        "auto_max_width_mm": round(max_width_mm, 2),
        "max_width_status": "automatic_from_mask",
        "width_length_ratio": round(max_width_mm / total_length_mm, 3) if total_length_mm else "",
        "basal_width_mm": round(basal_width_px * MM_PX, 2) if basal_width_px else "",
        "basal_width_status": profile_status,
        "tube_width_mid_mm": round(mid_width_px * MM_PX, 2) if mid_width_px else "",
        "tube_width_status": profile_status,
        "throat_width_mm": round(throat_width_px * MM_PX, 2) if throat_width_px else "",
        "throat_width_status": profile_status,
        # Keep the previous opening-width name for downstream compatibility, but
        # explicitly define it as the flattened throat/aperture cross-section.
        "opening_width_mm": round(throat_width_px * MM_PX, 2) if throat_width_px else "",
        "opening_width_status": profile_status,
        "tube_length_mm": round(tube_length_mm, 2) if tube_length_mm else "",
        "tube_length_status": tube_status,
        "lobe_length_mm": round(lobe_length_mm, 2) if lobe_length_mm else "",
        "lobe_length_status": tube_status,
        "lobe_count_auto": int(geometry.get("n_lobes", 0)) if geometry.get("n_lobes", 0) else "",
        "lobe_count_status": "automatic_provisional",
    }


def _enhanced_recompute(image: np.ndarray, mask: np.ndarray, row: dict, channels: tuple[np.ndarray, ...]) -> dict:
    updated = _ORIGINAL_RECOMPUTE(image, mask, row, channels)
    updated.update(corolla_shape_traits(mask))
    return updated


def _classify_organ_candidate(width_mm: float, aspect: float) -> tuple[str, str]:
    """Conservative morphology-only label; no forced stamen/pistil identity."""
    if width_mm >= 3.0 and aspect < 5.0:
        return "pistil_or_ovary_candidate", "broad_base_or_low_aspect"
    if width_mm <= 2.3 and aspect >= 4.0:
        return "stamen_or_style_candidate", "thin_high_aspect"
    return "reproductive_organ_candidate", "ambiguous_geometry"


def _instance_aware_organs(union: np.ndarray, corollas: list[dict], top: int, channels) -> list[dict]:
    rows = _ORIGINAL_DETECT_ORGANS(union, corollas, top, channels)

    # refine13 emits several points along a long component to make boundary-point
    # evaluation fair. Those points must not be interpreted as separate organs.
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            int(row.get("nearest_corolla", 0) or 0),
            round(float(row.get("organ_len_mm", 0.0)), 2),
            round(float(row.get("organ_width_mm", 0.0)), 2),
            round(float(row.get("organ_aspect", 0.0)), 2),
        )
        grouped[key].append(row)

    output: list[dict] = []
    for instance_number, records in enumerate(
        sorted(grouped.values(), key=lambda group: (int(group[0].get("nearest_corolla", 0)), float(group[0].get("cy", 0)), float(group[0].get("cx", 0)))),
        1,
    ):
        records.sort(key=lambda item: (float(item.get("cy", 0)), float(item.get("cx", 0))))
        sample_count = len(records)
        for sample_index, row in enumerate(records, 1):
            width = float(row.get("organ_width_mm", 0.0))
            aspect = float(row.get("organ_aspect", 0.0))
            label, reason = _classify_organ_candidate(width, aspect)
            row = dict(row)
            row.update(
                organ_instance_id=instance_number,
                organ_sample_index=sample_index,
                organ_sample_count=sample_count,
                measurement_unit="organ_component_sample",
                organ_type_auto=label,
                organ_type_reason=reason,
                organ_identity_status="candidate_requires_validation",
            )
            output.append(row)
    return output


# Install the extensions before refine13 starts its batch pipeline.
refine._recompute_traits = _enhanced_recompute
refine13.detect_organs = _instance_aware_organs


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
