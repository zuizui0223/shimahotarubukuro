# -*- coding: utf-8 -*-
"""Guide-supported pruning for residual corolla-mask appendages.

The previous geometric passes remove most styles and paper folds.  Remaining
errors are boundary regions with almost no nectar-guide pigment: lateral loops,
rounded style bases, and low-chroma paper folds.  This pass compares components
lost by a 10-mm opening against the guide distribution and keeps real lobes when
they have guide support or a broad attachment.

Maximum width remains provisional and opening width remains deferred.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine2 as refine2
import measure_guides_v3_refine5 as refine5  # noqa: F401  (installs merged-candidate pass)

_BASE_MERGED_DETACH = refine.detach_thin_appendages
_GUIDE_OPENING_MM = 10.0


def _largest(mask: np.ndarray) -> np.ndarray:
    q = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(q, 8)
    if n <= 1:
        return q
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def should_remove_guide_free(
    *,
    area_mm2: float,
    aspect: float,
    spot_fraction: float,
    median_pigment_index: float,
    median_chroma: float,
    neck_mm: float,
    longitudinal_position: float,
    lateral_position: float,
) -> bool:
    """Conservative decision rule exposed for regression tests."""
    guide_free = spot_fraction <= 0.02 and median_pigment_index <= -1.0

    # Style loop or rounded base on a lateral side of the corolla.
    mid_lateral = (
        guide_free
        and 0.12 <= longitudinal_position <= 0.88
        and lateral_position >= 0.92
        and area_mm2 <= 25.0
        and aspect >= 1.45
    )

    # A terminal appendage is accepted only when the attachment is narrow.  This
    # separates C13-like style bases from genuine broad terminal lobes.
    terminal_narrow = (
        spot_fraction <= 0.02
        and median_pigment_index <= -4.0
        and (longitudinal_position < -0.02 or longitudinal_position > 1.02)
        and area_mm2 <= 15.0
        and aspect >= 1.45
        and neck_mm <= 3.2
    )

    # Paper folds are unusually neutral.  Restrict this rule to the lateral or
    # lower/end-side part of a candidate so a pale upper tube edge is retained.
    paper_like = (
        spot_fraction <= 0.005
        and median_pigment_index <= -2.0
        and median_chroma <= 6.5
        and area_mm2 <= 15.0
        and (lateral_position >= 0.70 or longitudinal_position >= 0.72)
    )
    return bool(mid_lateral or terminal_narrow or paper_like)


def _guide_free_components(mask: np.ndarray) -> list[np.ndarray]:
    channels = refine2._CURRENT_CHANNELS
    if channels is None:
        return []

    q = (np.asarray(mask) > 0).astype(np.uint8)
    if not q.any():
        return []
    ys, xs = np.where(q > 0)
    margin = max(20, int(round(_GUIDE_OPENING_MM / float(base.MM_PX))))
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(q.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(q.shape[0], int(ys.max()) + margin + 1)
    crop = q[y0:y1, x0:x1]

    diameter = max(9, int(round(_GUIDE_OPENING_MM / float(base.MM_PX))))
    if diameter % 2 == 0:
        diameter += 1
    opened = cv2.morphologyEx(
        crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    )
    core = _largest(opened)
    if not core.any():
        return []

    core_y, core_x = np.where(core > 0)
    points = np.column_stack((core_x, core_y)).astype(np.float64)
    centre = points.mean(axis=0)
    covariance = np.cov(points - centre, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    # Resolve the arbitrary eigenvector sign consistently with the upright scan.
    if axis[1] < 0:
        axis = -axis
    perpendicular = np.array([-axis[1], axis[0]])
    centred = points - centre
    longitudinal = centred @ axis
    lateral = centred @ perpendicular
    long_min = float(longitudinal.min())
    long_max = float(longitudinal.max())
    lateral_half_width = max(abs(float(lateral.min())), abs(float(lateral.max())), 1.0)

    _, a, b, _, _, _, chroma = channels
    ac = a[y0:y1, x0:x1]
    bc = b[y0:y1, x0:x1]
    chroma_crop = chroma[y0:y1, x0:x1]
    spots = base.spot_segment(ac, bc, crop.astype(bool))
    pigment_index = ac - bc

    difference = crop & (1 - core)
    touching = cv2.dilate(core, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(difference, 8)
    output: list[np.ndarray] = []
    for label in range(1, n):
        component = (labels == label).astype(np.uint8)
        area_mm2 = int(stats[label, cv2.CC_STAT_AREA]) * float(base.MM2_PX)
        if area_mm2 < 0.30:
            continue

        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        rect_width, rect_height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
        length_mm = max(float(rect_width), float(rect_height)) * float(base.MM_PX)
        width_mm = min(float(rect_width), float(rect_height)) * float(base.MM_PX)
        aspect = length_mm / max(width_mm, 1e-6)

        neck = component & touching
        neck_mm = 0.0
        neck_contours, _ = cv2.findContours(
            neck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if neck_contours:
            neck_width, neck_height = cv2.minAreaRect(
                max(neck_contours, key=cv2.contourArea)
            )[1]
            neck_mm = max(float(neck_width), float(neck_height)) * float(base.MM_PX)

        selected = component > 0
        spot_fraction = float(spots[selected].mean())
        median_pigment = float(np.median(pigment_index[selected]))
        median_chroma = float(np.median(chroma_crop[selected]))

        component_centre = np.asarray(centroids[label], dtype=np.float64)
        vector = component_centre - centre
        long_value = float(vector @ axis)
        side_value = float(vector @ perpendicular)
        longitudinal_position = (long_value - long_min) / max(long_max - long_min, 1e-9)
        lateral_position = abs(side_value) / lateral_half_width

        if not should_remove_guide_free(
            area_mm2=area_mm2,
            aspect=aspect,
            spot_fraction=spot_fraction,
            median_pigment_index=median_pigment,
            median_chroma=median_chroma,
            neck_mm=neck_mm,
            longitudinal_position=longitudinal_position,
            lateral_position=lateral_position,
        ):
            continue
        full = np.zeros_like(q)
        full[y0:y1, x0:x1] = component
        output.append(full)
    return output


def detach_with_guide_support(mask: np.ndarray):
    cleaned, removed = _BASE_MERGED_DETACH(mask)
    additional = _guide_free_components(cleaned)
    if not additional:
        return cleaned, removed

    remove_union = np.zeros_like(cleaned, np.uint8)
    for component in additional:
        remove_union |= (np.asarray(component) > 0).astype(np.uint8)
    # This late pass is deliberately small.  If the rules propose deleting more
    # than 8% of the current corolla, retain the earlier geometry-only result.
    if int(remove_union.sum()) > int(np.asarray(cleaned).astype(bool).sum()) * 0.08:
        return cleaned, removed

    candidate = _largest(
        (np.asarray(cleaned) > 0).astype(np.uint8) & (1 - remove_union)
    )
    merged_removed = refine5.merge_removed_masks([*removed, *additional])
    return candidate, merged_removed


refine.detach_thin_appendages = detach_with_guide_support


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
