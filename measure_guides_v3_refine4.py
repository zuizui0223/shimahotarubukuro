# -*- coding: utf-8 -*-
"""Propagate v3 appendage cleanup into residual lateral bulbs/ovaries.

After a thin style is removed, a rounded ovary or pale base can remain attached to
the corolla because it is too broad for the first opening rule.  This pass removes
only a small lateral component that is on the same side as an already removed
appendage, lies in the middle of the corolla rather than at a lobe-bearing end,
and is within 8 mm of that appendage.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine3 as refine3  # noqa: F401  (installs safe multiscale pass)

_BASE_MULTISCALE_DETACH = refine.detach_thin_appendages
_PROPAGATION_OPENING_MM = 6.0
_PROPAGATION_DISTANCE_MM = 8.0


def _largest(mask: np.ndarray) -> np.ndarray:
    q = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(q, 8)
    if n <= 1:
        return q
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def _component_geometry(component: np.ndarray) -> tuple[float, float, float]:
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0, 0.0, 0.0
    rect_width, rect_height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
    length = max(float(rect_width), float(rect_height)) * float(base.MM_PX)
    width = min(float(rect_width), float(rect_height)) * float(base.MM_PX)
    return length, width, length / max(width, 1e-6)


def should_propagate(
    *,
    longitudinal_position: float,
    lateral_position: float,
    area_mm2: float,
    width_mm: float,
    distance_mm: float,
    same_side: bool,
) -> bool:
    """Conservative rule for a residual side bulb, exposed for regression tests."""
    return bool(
        same_side
        and 0.14 <= longitudinal_position <= 0.86
        and lateral_position >= 0.84
        and 0.45 <= area_mm2 <= 20.0
        and width_mm <= 5.2
        and distance_mm <= _PROPAGATION_DISTANCE_MM
    )


def _propagated_components(cleaned: np.ndarray, removed: list[np.ndarray]) -> list[np.ndarray]:
    if not removed:
        return []
    q = (np.asarray(cleaned) > 0).astype(np.uint8)
    removed_union = np.zeros_like(q)
    for mask in removed:
        removed_union |= (np.asarray(mask) > 0).astype(np.uint8)
    if not q.any() or not removed_union.any():
        return []

    combined = q | removed_union
    ys, xs = np.where(combined > 0)
    margin = max(12, int(round(_PROPAGATION_DISTANCE_MM / float(base.MM_PX))))
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(q.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(q.shape[0], int(ys.max()) + margin + 1)
    crop = q[y0:y1, x0:x1]
    removed_crop = removed_union[y0:y1, x0:x1]

    diameter = max(9, int(round(_PROPAGATION_OPENING_MM / float(base.MM_PX))))
    if diameter % 2 == 0:
        diameter += 1
    core = cv2.morphologyEx(
        crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    )
    core = _largest(core)
    if not core.any():
        return []

    core_y, core_x = np.where(core > 0)
    points = np.column_stack((core_x, core_y)).astype(np.float64)
    centre = points.mean(axis=0)
    covariance = np.cov(points - centre, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    perpendicular = np.array([-axis[1], axis[0]])
    centred = points - centre
    longitudinal = centred @ axis
    lateral = centred @ perpendicular
    long_min, long_max = float(longitudinal.min()), float(longitudinal.max())
    lateral_half_width = max(abs(float(lateral.min())), abs(float(lateral.max())), 1.0)

    removed_y, removed_x = np.where(removed_crop > 0)
    removed_points = np.column_stack((removed_x, removed_y)).astype(np.float64)
    removed_area_labels = cv2.connectedComponentsWithStats(removed_crop, 8)
    _, removed_labels, removed_stats, removed_centroids = removed_area_labels
    if len(removed_stats) <= 1:
        return []
    largest_removed = 1 + int(np.argmax(removed_stats[1:, cv2.CC_STAT_AREA]))
    removed_centre = np.asarray(removed_centroids[largest_removed], dtype=np.float64)
    removed_side = float((removed_centre - centre) @ perpendicular)

    distance_to_removed = cv2.distanceTransform(
        (1 - removed_crop).astype(np.uint8), cv2.DIST_L2, 5
    )
    difference = crop & (1 - core)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(difference, 8)
    output: list[np.ndarray] = []
    for label in range(1, n):
        component = (labels == label).astype(np.uint8)
        area_mm2 = int(stats[label, cv2.CC_STAT_AREA]) * float(base.MM2_PX)
        if area_mm2 < 0.45 or area_mm2 > 20.0:
            continue
        _, width_mm, _ = _component_geometry(component)
        component_centre = np.asarray(centroids[label], dtype=np.float64)
        vector = component_centre - centre
        long_value = float(vector @ axis)
        side_value = float(vector @ perpendicular)
        long_position = (long_value - long_min) / max(long_max - long_min, 1e-9)
        lateral_position = abs(side_value) / lateral_half_width
        distance_mm = float(distance_to_removed[component > 0].min()) * float(base.MM_PX)
        same_side = side_value == 0.0 or removed_side == 0.0 or side_value * removed_side > 0.0
        if not should_propagate(
            longitudinal_position=long_position,
            lateral_position=lateral_position,
            area_mm2=area_mm2,
            width_mm=width_mm,
            distance_mm=distance_mm,
            same_side=same_side,
        ):
            continue
        full = np.zeros_like(q)
        full[y0:y1, x0:x1] = component
        output.append(full)
    return output


def detach_with_side_bulbs(mask: np.ndarray):
    cleaned, removed = _BASE_MULTISCALE_DETACH(mask)
    propagated = _propagated_components(cleaned, removed)
    if not propagated:
        return cleaned, removed

    remove_union = np.zeros_like(cleaned, np.uint8)
    for component in propagated:
        remove_union |= (np.asarray(component) > 0).astype(np.uint8)
    candidate = _largest((np.asarray(cleaned) > 0).astype(np.uint8) & (1 - remove_union))

    total_removed = sum(int(np.asarray(item).astype(bool).sum()) for item in removed)
    total_removed += int(remove_union.sum())
    if total_removed > int(np.asarray(mask).astype(bool).sum()) * 0.18:
        return cleaned, removed
    return candidate, [*removed, *propagated]


refine.detach_thin_appendages = detach_with_side_bulbs


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
