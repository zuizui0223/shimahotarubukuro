# -*- coding: utf-8 -*-
"""Convert human-reviewed shimask strokes into existing pipeline inputs.

Only the two input seams are implemented here:
- red outline -> v2-compatible corolla components
- green trace -> reviewed-organ rows

Nectar-guide extraction and floral-trait measurement remain in the established
review modules and are not reimplemented here.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2


def stroke_masks(annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return red and green hand-drawn stroke masks at annotation resolution."""
    if annotated.ndim != 3:
        raise ValueError("annotated must be a BGR colour image")
    b, g, r = cv2.split(annotated.astype(np.int16))
    red = ((r - np.maximum(g, b) >= 50) & (r >= 110)).astype(np.uint8)
    green = ((g - np.maximum(r, b) >= 45) & (g >= 90)).astype(np.uint8)
    return red, green


def _resize_nn(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def _odd_kernel_size(mm: float, *, minimum: int = 3) -> int:
    value = max(minimum, int(round(mm / float(base.MM_PX))))
    return value if value % 2 else value + 1


def _closed_red_regions(red: np.ndarray) -> list[np.ndarray]:
    """Fill regions enclosed by red strokes without shrinking their boundary.

    A small 0.5-mm closing only bridges pen gaps. Unlike the previous 3-mm
    close+erosion operation, this never contracts the confirmed outline inward.
    The filled region follows the outside edge of the visible red stroke, which
    stays within the human annotation width and is reproducible.
    """
    kernel_size = _odd_kernel_size(0.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    joined = cv2.morphologyEx((red > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    masks: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) * float(base.MM2_PX) < float(base.AREA_MM2_MIN):
            continue
        mask = np.zeros_like(joined, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        masks.append(mask)
    return masks


def red_corolla_components(raw: np.ndarray, annotated: np.ndarray) -> list[dict]:
    """Human red outlines -> components matching ``v2.corollas`` output."""
    red_small, _ = stroke_masks(annotated)
    red = _resize_nn(red_small, raw.shape[:2])
    components: list[dict] = []
    for source_id, mask in enumerate(_closed_red_regions(red), start=1):
        measured = v2.metrics(mask)
        if measured is None:
            continue
        area_mm2 = float(mask.sum()) * float(base.MM2_PX)
        length_mm = float(measured["length_px"]) * float(base.MM_PX)
        width_mm = float(measured["width_px"]) * float(base.MM_PX)
        if not float(base.AREA_MM2_MIN) <= area_mm2 <= float(base.AREA_MM2_MAX):
            continue
        if length_mm < 15.0 or width_mm < 10.0:
            continue
        moments = cv2.moments(mask)
        if moments["m00"] <= 0:
            continue
        components.append(
            {
                "mask": mask.astype(bool),
                "source_component_id": source_id,
                "split_piece": 1,
                "split_status": "shimask_red_outline",
                "cx": moments["m10"] / moments["m00"],
                "cy": moments["m01"] / moments["m00"],
                "m": measured,
            }
        )
    components.sort(key=lambda row: (int(float(row["cy"])) // 170, float(row["cx"])))
    return components


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8)
    skeleton = np.zeros_like(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        skeleton |= image & (1 - opened)
        image = cv2.erode(image, kernel)
    return skeleton


def _skeleton_length_px(skeleton: np.ndarray) -> float:
    """8-neighbour skeleton edge length, with every edge counted once."""
    values = skeleton > 0
    horizontal = int(np.sum(values[:, 1:] & values[:, :-1]))
    vertical = int(np.sum(values[1:, :] & values[:-1, :]))
    diagonal_a = int(np.sum(values[1:, 1:] & values[:-1, :-1]))
    diagonal_b = int(np.sum(values[1:, :-1] & values[:-1, 1:]))
    return float(horizontal + vertical + math.sqrt(2.0) * (diagonal_a + diagonal_b))


def _principal_endpoints(skeleton: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float]:
    ys, xs = np.where(skeleton > 0)
    if xs.size < 2:
        point = (int(xs[0]), int(ys[0])) if xs.size else (0, 0)
        return point, point, 0.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    centred = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    axis = vectors[0]
    projection = centred @ axis
    first = points[int(np.argmin(projection))]
    second = points[int(np.argmax(projection))]
    chord = float(np.linalg.norm(second - first))
    return (int(round(first[0])), int(round(first[1]))), (int(round(second[0])), int(round(second[1]))), chord


def green_organ_rows(raw: np.ndarray, annotated: np.ndarray) -> list[dict]:
    """Human green strokes -> rows matching the reviewed-organ contract."""
    _, green_small = stroke_masks(annotated)
    green = _resize_nn(green_small, raw.shape[:2])
    kernel_size = _odd_kernel_size(0.35)
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(green, 8)
    rows: list[dict] = []
    for index in range(1, count):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        span_px = max(int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
        if area_px * float(base.MM2_PX) < 0.7 or span_px * float(base.MM_PX) < 3.0:
            continue
        component = (labels == index).astype(np.uint8)
        skeleton = _skeletonize(component)
        length_px = _skeleton_length_px(skeleton)
        if length_px <= 0:
            continue
        (x1, y1), (x2, y2), chord_px = _principal_endpoints(skeleton)
        cx, cy = map(float, centroids[index])
        length_mm = length_px * float(base.MM_PX)
        chord_mm = chord_px * float(base.MM_PX)
        width_mm = area_px * float(base.MM2_PX) / max(length_mm, 1e-9)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        rows.append(
            {
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length_mm": round(length_mm, 2),
                "skeleton_length_mm": round(length_mm, 2),
                "endpoint_distance_mm": round(chord_mm, 2),
                "width_mm": round(width_mm, 2),
                "aspect": round(length_mm / max(width_mm, 1e-6), 2),
                "angle_deg": round(angle, 2),
                "score": round(1000.0 - len(rows), 3),
                "organ_type_auto": "reviewed_reproductive_organ",
                "organ_type_FILL": "",
                "exclude_FILL": "",
                "detection_source": "shimask_green_stroke",
                "nearest_corolla_hint": "",
                "association_confirmed": 1,
                "visibility_note": "human-reviewed green stroke; path length measured on skeleton",
            }
        )
    rows.sort(key=lambda row: (float(row["cy"]), float(row["cx"])))
    return rows


def write_annotation_overlay(raw: np.ndarray, annotated: np.ndarray, out_path: Path) -> None:
    """Write raw + red/green annotation strokes only; never draw guide results."""
    red_small, green_small = stroke_masks(annotated)
    red = _resize_nn(red_small, raw.shape[:2]) > 0
    green = _resize_nn(green_small, raw.shape[:2]) > 0
    overlay = raw.copy()
    overlay[cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 0, 255)
    overlay[cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError(f"Could not encode annotation overlay: {out_path}")
    encoded.tofile(str(out_path))
