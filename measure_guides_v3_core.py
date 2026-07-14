# -*- coding: utf-8 -*-
"""Geometry and confidence helpers for automatic-first floral extraction."""
from __future__ import annotations

import math

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2

COROLLA_HIGH = 0.80
COROLLA_MEDIUM = 0.60
ORGAN_ACCEPT = 0.72
ASSOCIATION_ACCEPT = 0.62


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence_label(score: float) -> str:
    if score >= COROLLA_HIGH:
        return "high"
    if score >= COROLLA_MEDIUM:
        return "medium"
    return "low"


def touches_border(mask: np.ndarray, margin: int = 5) -> bool:
    """Return True when a component reaches the image boundary."""
    q = np.asarray(mask, dtype=bool)
    if not q.any():
        return True
    margin = max(1, int(margin))
    return bool(
        q[:margin].any()
        or q[-margin:].any()
        or q[:, :margin].any()
        or q[:, -margin:].any()
    )


def _points_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return np.array([0.0, 0.0]), np.array([0.0, 1.0])
    centre = points.mean(axis=0)
    centred = points - centre
    covariance = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        axis = np.array([0.0, 1.0])
    else:
        axis = axis / norm
    return centre, axis


def pca_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit major-axis vector for a binary component."""
    ys, xs = np.where(np.asarray(mask) > 0)
    return _points_axis(np.column_stack((xs, ys)))


def contour_axis(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast major axis using only an already-computed contour."""
    return _points_axis(np.asarray(contour).reshape(-1, 2))


def projection_extents(mask: np.ndarray) -> tuple[float, float]:
    """Major-axis length and perpendicular span in pixels."""
    ys, xs = np.where(np.asarray(mask) > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)
    centre, axis = _points_axis(points)
    if len(points) < 2:
        return 0.0, 0.0
    perpendicular = np.array([-axis[1], axis[0]])
    centred = points - centre
    return float(np.ptp(centred @ axis)), float(np.ptp(centred @ perpendicular))


def contour_projection_extents(contour: np.ndarray) -> tuple[float, float]:
    """Provisional extents from contour points without scanning the whole mask."""
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    centre, axis = _points_axis(points)
    if len(points) < 2:
        return 0.0, 0.0
    perpendicular = np.array([-axis[1], axis[0]])
    centred = points - centre
    return float(np.ptp(centred @ axis)), float(np.ptp(centred @ perpendicular))


def corolla_confidence(
    component: dict,
    *,
    mm_per_px: float | None = None,
    mm2_per_px: float | None = None,
) -> dict:
    """Score whether a v2 corolla mask can be accepted without manual editing."""
    mm_per_px = float(base.MM_PX if mm_per_px is None else mm_per_px)
    mm2_per_px = float(base.MM2_PX if mm2_per_px is None else mm2_per_px)
    mask = np.asarray(component["mask"], dtype=np.uint8)
    measured = component.get("m") or v2.metrics(mask)
    if measured is None:
        return {
            "mask_confidence": 0.0,
            "mask_confidence_label": "low",
            "mask_qc_required": 1,
            "mask_qc_reasons": "empty_mask",
            "auto_max_width_mm": "",
            "max_width_status": "provisional_not_for_primary_analysis",
            "opening_width_mm": "",
            "opening_width_status": "deferred",
        }

    score = 1.0
    reasons: list[str] = []
    split_status = str(component.get("split_status", ""))
    area_mm2 = float(measured["area_px"]) * mm2_per_px
    min_width_mm = float(measured["width_px"]) * mm_per_px
    solidity = float(measured["solidity"])
    aspect = float(measured["aspect"])

    if split_status == "auto_split":
        score -= 0.10
        reasons.append("auto_split")
    elif split_status not in ("", "not_triggered"):
        score -= 0.38
        reasons.append(split_status or "split_uncertain")

    if v2.is_fragment(area_mm2, min_width_mm):
        score -= 0.55
        reasons.append("fragment_like")
    if solidity < 0.52:
        score -= 0.34
        reasons.append("very_low_solidity")
    elif solidity < 0.65:
        score -= 0.18
        reasons.append("low_solidity")
    if aspect > 4.0:
        score -= 0.35
        reasons.append("very_elongated")
    elif aspect > 2.8:
        score -= 0.16
        reasons.append("elongated")
    if touches_border(mask):
        score -= 0.30
        reasons.append("touches_image_border")

    area_min = float(base.AREA_MM2_MIN)
    area_max = float(base.AREA_MM2_MAX)
    if area_mm2 < area_min * 1.12 or area_mm2 > area_max * 0.88:
        score -= 0.12
        reasons.append("near_area_limit")

    contour = measured.get("contour")
    if contour is not None:
        _, auto_width_px = contour_projection_extents(contour)
    else:
        _, auto_width_px = projection_extents(mask)
    score = _clip01(score)
    label = _confidence_label(score)
    return {
        "mask_confidence": round(score, 3),
        "mask_confidence_label": label,
        "mask_qc_required": int(label != "high"),
        "mask_qc_reasons": "|".join(dict.fromkeys(reasons)),
        "auto_max_width_mm": round(auto_width_px * mm_per_px, 2),
        "max_width_status": "provisional_not_for_primary_analysis",
        "opening_width_mm": "",
        "opening_width_status": "deferred",
    }


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton using only core OpenCV operations."""
    image = (np.asarray(mask) > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        image = eroded
    return (skeleton > 0).astype(np.uint8)


def skeleton_length_px(skeleton: np.ndarray) -> float:
    """8-neighbour skeleton length without counting each edge twice."""
    q = np.asarray(skeleton, dtype=bool)
    if not q.any():
        return 0.0
    horizontal = np.count_nonzero(q[:, :-1] & q[:, 1:])
    vertical = np.count_nonzero(q[:-1, :] & q[1:, :])
    diagonal_a = np.count_nonzero(q[:-1, :-1] & q[1:, 1:])
    diagonal_b = np.count_nonzero(q[:-1, 1:] & q[1:, :-1])
    return float(horizontal + vertical + math.sqrt(2.0) * (diagonal_a + diagonal_b))


def _cluster_node_pixels(points: np.ndarray, shape: tuple[int, int], radius: int = 4) -> np.ndarray:
    """Collapse small endpoint/branch pixel clusters to one biologically useful node."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    node_mask = np.zeros(shape, np.uint8)
    for x, y in np.asarray(points, dtype=int):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            node_mask[y, x] = 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    grouped = cv2.dilate(node_mask, kernel)
    n, labels, _, centroids = cv2.connectedComponentsWithStats(grouped, 8)
    if n <= 1:
        return np.empty((0, 2), dtype=np.float64)
    output = []
    for label in range(1, n):
        region = labels == label
        original = np.argwhere(region & (node_mask > 0))
        if original.size:
            yx = original.mean(axis=0)
            output.append([yx[1], yx[0]])
        else:
            output.append(centroids[label].tolist())
    return np.asarray(output, dtype=np.float64)


def skeleton_nodes(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clustered endpoint and branch-point coordinates as x,y arrays."""
    q = (np.asarray(skeleton) > 0).astype(np.uint8)
    neighbours = cv2.filter2D(q, cv2.CV_16S, np.ones((3, 3), np.uint8)) - q
    ey, ex = np.where((q > 0) & (neighbours == 1))
    by, bx = np.where((q > 0) & (neighbours >= 3))
    endpoint_pixels = np.column_stack((ex, ey)).astype(np.float64)
    branch_pixels = np.column_stack((bx, by)).astype(np.float64)
    endpoints = _cluster_node_pixels(endpoint_pixels, q.shape, radius=4)
    branches = _cluster_node_pixels(branch_pixels, q.shape, radius=5)
    return endpoints, branches


def component_features(
    mask: np.ndarray,
    *,
    mm_per_px: float | None = None,
) -> dict:
    """Geometry used to classify a detached reproductive-organ candidate."""
    mm_per_px = float(base.MM_PX if mm_per_px is None else mm_per_px)
    q = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    rect_length = max(float(rw), float(rh))
    rect_width = min(float(rw), float(rh))
    if rw < rh:
        angle += 90.0
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    skeleton = skeletonize(q)
    endpoints, branches = skeleton_nodes(skeleton)
    distance = cv2.distanceTransform(q, cv2.DIST_L2, 5)
    widths_px = 2.0 * distance[skeleton > 0]
    median_width_px = float(np.median(widths_px)) if widths_px.size else rect_width
    max_width_px = float(np.max(widths_px)) if widths_px.size else rect_width
    endpoint_widths = []
    for x, y in endpoints:
        endpoint_widths.append(2.0 * float(distance[int(round(y)), int(round(x))]))
    tip_width_px = max(endpoint_widths, default=median_width_px)
    tip_bulge_ratio = tip_width_px / max(median_width_px, 1e-6)
    skel_length_px = skeleton_length_px(skeleton)
    straightness = rect_length / max(skel_length_px, 1e-6)

    return {
        "mask": q,
        "skeleton": skeleton,
        "endpoints": endpoints,
        "branches": branches,
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "length_mm": round(skel_length_px * mm_per_px, 2),
        "rect_length_mm": round(rect_length * mm_per_px, 2),
        "median_width_mm": round(median_width_px * mm_per_px, 3),
        "max_width_mm": round(max_width_px * mm_per_px, 3),
        "aspect": round(rect_length / max(rect_width, 1e-6), 2),
        "angle_deg": round(float(angle), 2),
        "n_endpoints": int(len(endpoints)),
        "n_branch_points": int(len(branches)),
        "tip_bulge_ratio": round(float(tip_bulge_ratio), 3),
        "straightness": round(float(straightness), 3),
        "area_px": int(q.sum()),
    }


def classify_organ(features: dict) -> tuple[str, float, list[str]]:
    """Conservative morphology-only classification.

    The labels deliberately say ``candidate`` because a scan alone cannot always
    distinguish one stamen from a pistil. Ambiguous objects remain unknown.
    """
    if not features:
        return "fragment_or_noise", 0.0, ["empty_component"]
    length = float(features["length_mm"])
    width = float(features["median_width_mm"])
    max_width = float(features["max_width_mm"])
    aspect = float(features["aspect"])
    tips = int(features["n_endpoints"])
    branches = int(features["n_branch_points"])
    bulge = float(features["tip_bulge_ratio"])
    straightness = float(features["straightness"])

    if length < 2.5 or aspect < 2.2 or width > 5.0 or max_width > 8.0:
        return "fragment_or_noise", 0.88, ["outside_reproductive_shape"]
    if tips == 0 or straightness < 0.25:
        return "fragment_or_noise", 0.78, ["poor_skeleton"]

    if (
        8.0 <= length <= 45.0
        and 0.18 <= width <= 2.8
        and aspect >= 4.0
        and tips <= 3
        and branches <= 2
        and bulge >= 1.35
    ):
        score = 0.72
        score += min((bulge - 1.35) * 0.16, 0.10)
        score += min((aspect - 4.0) * 0.025, 0.08)
        score += 0.04 if straightness >= 0.55 else 0.0
        return "pistil_candidate", _clip01(score), ["elongated", "swollen_tip"]

    if (
        3.0 <= length <= 35.0
        and 0.12 <= width <= 4.5
        and (aspect >= 3.0 or branches >= 1)
        and (branches >= 1 or tips >= 3 or bulge < 1.35)
    ):
        score = 0.65
        score += min(branches * 0.025, 0.10)
        score += min(max(tips - 2, 0) * 0.025, 0.08)
        score += 0.05 if aspect >= 4.0 else 0.0
        return "stamen_bundle_candidate", _clip01(score), ["slender_or_branched"]

    score = 0.48
    if 3.0 <= length <= 45.0 and aspect >= 2.8:
        score += 0.10
    return "reproductive_organ_unknown", _clip01(score), ["ambiguous_morphology"]


def _angle_difference(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa /= max(float(np.linalg.norm(aa)), 1e-9)
    bb /= max(float(np.linalg.norm(bb)), 1e-9)
    cosine = abs(float(np.dot(aa, bb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def associate_organ(
    organ: dict,
    corollas: list[dict],
    *,
    mm_per_px: float | None = None,
) -> dict:
    """Associate an organ using endpoint distance and corolla-axis agreement."""
    mm_per_px = float(base.MM_PX if mm_per_px is None else mm_per_px)
    if not corollas:
        return {
            "nearest_corolla": "",
            "association_confidence": 0.0,
            "association_distance_mm": "",
            "association_angle_deg": "",
            "association_qc_required": 1,
        }

    endpoints = np.asarray(organ.get("endpoints", []), dtype=float)
    if endpoints.size == 0:
        endpoints = np.array([[organ["cx"], organ["cy"]]], dtype=float)
    organ_axis = np.array(
        [math.cos(math.radians(float(organ["angle_deg"]))),
         math.sin(math.radians(float(organ["angle_deg"])))],
        dtype=float,
    )

    best: tuple[float, int, float, float] | None = None
    for cid, component in enumerate(corollas, 1):
        measured = component.get("m") or {}
        contour = measured.get("contour")
        if contour is None:
            mask = np.asarray(component["mask"], dtype=np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
        endpoint_distances = [
            abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
            for x, y in endpoints
        ]
        distance_mm = min(endpoint_distances) * mm_per_px
        _, corolla_axis = contour_axis(contour)
        angle = _angle_difference(organ_axis, corolla_axis)
        objective = distance_mm + 0.06 * angle
        if best is None or objective < best[0]:
            best = (objective, cid, distance_mm, angle)

    assert best is not None
    _, cid, distance_mm, angle = best
    distance_score = math.exp(-distance_mm / 12.0)
    angle_score = math.exp(-angle / 55.0)
    confidence = _clip01(0.72 * distance_score + 0.28 * angle_score)
    return {
        "nearest_corolla": cid,
        "association_confidence": round(confidence, 3),
        "association_distance_mm": round(distance_mm, 2),
        "association_angle_deg": round(angle, 1),
        "association_qc_required": int(confidence < ASSOCIATION_ACCEPT),
    }
