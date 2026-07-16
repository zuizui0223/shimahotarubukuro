# -*- coding: utf-8 -*-
"""Direct measurements from human-confirmed shimask annotations.

Red closed outlines are filled and measured as corolla masks. Green traces are
measured directly as reproductive-organ centre lines. No floral-structure or
organ-identity inference is performed here.
"""
from __future__ import annotations

import cv2
import numpy as np

from export_shimask_ground_truth import skeletonize


def _skeleton_trace_length_px(skeleton: np.ndarray) -> float:
    q = np.asarray(skeleton) > 0
    if not q.any():
        return 0.0
    horizontal = int(np.sum(q[:, 1:] & q[:, :-1]))
    vertical = int(np.sum(q[1:, :] & q[:-1, :]))
    diag1 = int(np.sum(q[1:, 1:] & q[:-1, :-1]))
    diag2 = int(np.sum(q[1:, :-1] & q[:-1, 1:]))
    return float(horizontal + vertical + (diag1 + diag2) * np.sqrt(2.0))


def confirmed_corolla_masks(red: np.ndarray, min_area_px: int = 500) -> list[np.ndarray]:
    """Fill each human-drawn red closed outline without shape inference."""
    binary = (np.asarray(red) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    masks: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        if int(mask.sum()) >= min_area_px:
            masks.append(mask)

    def key(mask: np.ndarray) -> tuple[int, float]:
        moments = cv2.moments(mask)
        cx = moments["m10"] / moments["m00"] if moments["m00"] else 0.0
        cy = moments["m01"] / moments["m00"] if moments["m00"] else 0.0
        return (int(cy) // 170, cx)

    masks.sort(key=key)
    return masks


def simple_corolla_metrics(mask: np.ndarray, mm_per_px: float) -> dict[str, float | int | str]:
    """Measure only geometry directly observed from the filled red mask."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Empty corolla mask")
    contour = max(contours, key=cv2.contourArea)
    area_px = int(binary.sum())
    perimeter_px = float(cv2.arcLength(contour, True))
    (_cx, _cy), (side_a, side_b), _angle = cv2.minAreaRect(contour)
    length_px = float(max(side_a, side_b))
    width_px = float(min(side_a, side_b))
    hull = cv2.convexHull(contour)
    hull_area_px = float(cv2.contourArea(hull))
    circularity = 4.0 * np.pi * area_px / max(perimeter_px * perimeter_px, 1e-9)
    return {
        "corolla_area_px": area_px,
        "corolla_area_mm2": round(area_px * mm_per_px * mm_per_px, 3),
        "corolla_perimeter_px": round(perimeter_px, 3),
        "corolla_perimeter_mm": round(perimeter_px * mm_per_px, 3),
        "corolla_length_px": round(length_px, 3),
        "corolla_length_mm": round(length_px * mm_per_px, 3),
        "corolla_width_px": round(width_px, 3),
        "corolla_width_mm": round(width_px * mm_per_px, 3),
        "aspect_ratio": round(length_px / max(width_px, 1e-9), 4),
        "circularity": round(float(circularity), 4),
        "solidity": round(area_px / max(hull_area_px, 1e-9), 4),
        "measurement_status": "direct_from_filled_red_outline",
    }


def _confirmed_organs(green: np.ndarray, mm_per_px: float, sheet: str, island: str) -> list[dict]:
    """Measure each connected green trace directly; never merge nearby traces."""
    binary = (np.asarray(green) > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    rows: list[dict] = []
    for label in range(1, n):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 20:
            continue
        component = (labels == label).astype(np.uint8)
        skel = skeletonize(component)
        length_px = _skeleton_trace_length_px(skel)
        if length_px < 5:
            continue
        rows.append({
            "island": island,
            "sheet": sheet,
            "confirmed_organ_id": len(rows) + 1,
            "cx": round(float(centroids[label][0]), 2),
            "cy": round(float(centroids[label][1]), 2),
            "organ_length_px": round(length_px, 3),
            "organ_length_mm": round(length_px * mm_per_px, 3),
            "annotation_area_px": area_px,
            "organ_identity": "human_confirmed_reproductive_organ_untyped",
            "measurement_status": "direct_from_green_trace",
            "provenance": "shimask_human_review",
        })
    return rows
