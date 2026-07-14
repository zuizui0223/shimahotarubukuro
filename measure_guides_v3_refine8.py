# -*- coding: utf-8 -*-
"""Split moderately sized touching corollas missed by the original v2 trigger.

The original splitter runs only above 55 mm length or 1350 mm2 area.  A clear
pair such as Niijima C8 is slightly smaller (about 53 mm, 1014 mm2) but strongly
concave/low-solidity.  This layer retains the original conservative validation
and adds a trigger only when a component is both large enough to contain two
flowers and substantially less convex than a normal single flattened corolla.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_v3_refine as refine
import measure_guides_v3_refine7 as refine7  # noqa: F401  (installs prior refinements)

_ORIGINAL_TRY_SPLIT = v2.try_split

_CONCAVE_AREA_MIN_MM2 = 850.0
_CONCAVE_LENGTH_MIN_MM = 44.0
_CONCAVE_SOLIDITY_MAX = 0.78
_CHILD_AREA_RATIO_MIN = 0.22


def should_try_concave_split(
    *, area_mm2: float, length_mm: float, solidity: float
) -> bool:
    """Return True only for large, elongated-enough, strongly concave components."""
    return bool(
        area_mm2 >= _CONCAVE_AREA_MIN_MM2
        and length_mm >= _CONCAVE_LENGTH_MIN_MM
        and solidity <= _CONCAVE_SOLIDITY_MAX
    )


def _validated_kmeans_split(mask: np.ndarray):
    q = (np.asarray(mask) > 0).astype(np.uint8)
    measured = v2.metrics(q)
    if measured is None:
        return [q], "split_rejected"

    ys, xs = np.where(q > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)
    if len(points) < 2:
        return [q], "split_rejected"

    cv2.setRNGSeed(20260710)
    _, labels, centres = cv2.kmeans(
        points,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5),
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    if np.linalg.norm(centres[0] - centres[1]) < max(80.0, 0.22 * measured["length_px"]):
        return [q], "split_centres_too_close"

    children: list[np.ndarray] = []
    for cluster in (0, 1):
        child = np.zeros_like(q)
        selected = labels.ravel() == cluster
        child[ys[selected], xs[selected]] = 1
        child = cv2.morphologyEx(child, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, child_labels, stats, _ = cv2.connectedComponentsWithStats(child, 8)
        if n > 1:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            child = (child_labels == keep).astype(np.uint8)
        children.append(child)

    child_metrics = [v2.metrics(child) for child in children]
    if any(item is None for item in child_metrics):
        return [q], "split_rejected"
    areas = [float(item["area_px"]) * float(base.MM2_PX) for item in child_metrics]
    lengths = [float(item["length_px"]) * float(base.MM_PX) for item in child_metrics]
    plausible = all(
        float(base.AREA_MM2_MIN) <= area <= 1500.0
        and 15.0 <= length <= 60.0
        and float(item["solidity"]) >= 0.40
        and float(item["aspect"]) <= 4.5
        for area, length, item in zip(areas, lengths, child_metrics)
    )
    balanced = min(areas) / max(areas) >= _CHILD_AREA_RATIO_MIN
    if plausible and balanced:
        return children, "auto_split"
    return [q], "split_rejected"


def try_split_with_concavity(mask: np.ndarray):
    """Run the original splitter, then the extra concavity trigger if needed."""
    pieces, status = _ORIGINAL_TRY_SPLIT(mask)
    if status != "not_triggered" or len(pieces) != 1:
        return pieces, status

    measured = v2.metrics(mask)
    if measured is None:
        return pieces, status
    area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
    length_mm = float(measured["length_px"]) * float(base.MM_PX)
    solidity = float(measured["solidity"])
    if not should_try_concave_split(
        area_mm2=area_mm2,
        length_mm=length_mm,
        solidity=solidity,
    ):
        return pieces, status
    return _validated_kmeans_split(mask)


v2.try_split = try_split_with_concavity


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
