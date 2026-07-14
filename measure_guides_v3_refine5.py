# -*- coding: utf-8 -*-
"""Merge nearby removed appendage pieces into one organ candidate.

Residual side-bulb propagation can remove a thin style and its rounded base as
separate mask components.  They belong to one biological candidate and must not
inflate organ counts, so components within 8 mm are unioned for measurement while
the already-cleaned corolla mask is left unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine4 as refine4  # noqa: F401  (installs side-bulb pass)

_BASE_SIDE_BULB_DETACH = refine.detach_thin_appendages
_MERGE_DISTANCE_MM = 8.0


def _minimum_distance_mm(first: np.ndarray, second: np.ndarray) -> float:
    a = (np.asarray(first) > 0).astype(np.uint8)
    b = (np.asarray(second) > 0).astype(np.uint8)
    if not a.any() or not b.any():
        return float("inf")
    union = a | b
    ys, xs = np.where(union > 0)
    margin = 3
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(a.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(a.shape[0], int(ys.max()) + margin + 1)
    ac = a[y0:y1, x0:x1]
    bc = b[y0:y1, x0:x1]
    if np.any(ac & bc):
        return 0.0
    distance = cv2.distanceTransform((1 - bc).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[ac > 0].min()) * float(base.MM_PX)


def merge_removed_masks(
    removed: list[np.ndarray], max_distance_mm: float = _MERGE_DISTANCE_MM
) -> list[np.ndarray]:
    """Union removed components connected by a short biological-scale gap."""
    masks = [(np.asarray(mask) > 0).astype(np.uint8) for mask in removed if np.any(mask)]
    if len(masks) <= 1:
        return masks

    parent = list(range(len(masks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            if _minimum_distance_mm(masks[first], masks[second]) <= max_distance_mm:
                union(first, second)

    groups: dict[int, np.ndarray] = {}
    for index, mask in enumerate(masks):
        root = find(index)
        if root not in groups:
            groups[root] = np.zeros_like(mask)
        groups[root] |= mask
    return list(groups.values())


def detach_with_merged_candidates(mask: np.ndarray):
    cleaned, removed = _BASE_SIDE_BULB_DETACH(mask)
    return cleaned, merge_removed_masks(removed)


refine.detach_thin_appendages = detach_with_merged_candidates


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
