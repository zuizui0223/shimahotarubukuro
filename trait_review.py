# -*- coding: utf-8 -*-
"""Image-grounded trait review calculations for one corolla."""
from __future__ import annotations

import math

import cv2
import numpy as np

import measure_guides as base
import measure_guides_review_spots as reviewed_spots


MEASUREMENT_GUIDES = {
    "max_span": {
        "label": "Maximum corolla span",
        "colour": "#1976d2",
        "fields": ("corolla_width_mm", "corolla_max_span_ruler_mm"),
    },
    "throat_span": {
        "label": "Throat / lobe boundary",
        "colour": "#c2185b",
        "fields": ("flat_throat_span_mm", "flat_lobe_length_mm", "flat_tube_length_mm"),
    },
    "mid_tube_span": {
        "label": "Mid-tube width",
        "colour": "#7b1fa2",
        "fields": ("flat_mid_tube_width_mm",),
    },
    "basal_tube_span": {
        "label": "Basal tube width",
        "colour": "#00897b",
        "fields": ("flat_basal_tube_width_mm",),
    },
}

REGION_TARGETS = {
    "guide": {"label": "Purple nectar guide", "colour_bgr": (230, 40, 190)},
    "oxidized": {"label": "Oxidized guide", "colour_bgr": (0, 165, 255)},
    "brown": {"label": "Brown / degraded tissue", "colour_bgr": (0, 105, 170)},
}

CORE_SHAPE_GUIDES = ("max_span", "throat_span", "basal_tube_span")

CORE_POLLINATION_TRAITS = (
    ("corolla_length_ruler_mm", "Corolla length", "mm", "Size"),
    ("corolla_area_ruler_mm2", "Corolla area", "mm2", "Size"),
    ("corolla_max_span_ruler_mm", "Maximum span", "mm", "Size"),
    ("flat_throat_span_mm", "Flattened throat span", "mm", "Access"),
    ("flat_throat_openness", "Relative throat openness", "ratio", "Access"),
    ("flat_tube_taper_ratio", "Tube taper", "ratio", "Access"),
    ("guide_area_mm2", "Guide area", "mm2", "Guide"),
    ("guide_cov_pct", "Guide coverage", "%", "Guide"),
    ("guide_present", "Guide detected", "0/1", "Guide"),
)

GUIDE_ANALYSIS_FIELDS = (
    "guide_area_mm2",
    "guide_cov_pct",
    "guide_present",
    "n_spots",
    "guide_area_incl_oxidized_mm2",
    "guide_cov_incl_oxidized_pct",
)


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def reviewed_guide_trait_values(values, presence_status: str):
    """Apply a manual presence decision to analysis-ready guide values."""
    output = dict(values)
    if presence_status == "present":
        output["guide_present"] = 1
    elif presence_status == "absent":
        for field in GUIDE_ANALYSIS_FIELDS:
            output[field] = 0
    elif presence_status == "uncertain":
        for field in GUIDE_ANALYSIS_FIELDS:
            output[field] = ""
    return output


def _axis(base_xy, tip_xy):
    base_point = np.asarray(base_xy, dtype=float)
    tip_point = np.asarray(tip_xy, dtype=float)
    vector = tip_point - base_point
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        vector = np.array([0.0, 1.0])
        length = 1.0
    unit = vector / length
    normal = np.array([-unit[1], unit[0]])
    return base_point, tip_point, unit, normal, length


def line_length(line) -> float:
    if line is None or len(line) != 2:
        return 0.0
    return float(np.linalg.norm(np.asarray(line[1], float) - np.asarray(line[0], float)))


def line_midpoint(line) -> np.ndarray:
    if line is None or len(line) != 2:
        return np.zeros(2, dtype=float)
    return (np.asarray(line[0], float) + np.asarray(line[1], float)) / 2.0


def organ_candidate_line(row, mm_per_px: float):
    """Return a detector candidate as a raw-image two-point centre line."""
    explicit = [_safe_float(row.get(key), float("nan")) for key in ("x1", "y1", "x2", "y2")]
    if all(math.isfinite(value) for value in explicit):
        return [[explicit[0], explicit[1]], [explicit[2], explicit[3]]]

    cx = _safe_float(row.get("cx"), float("nan"))
    cy = _safe_float(row.get("cy"), float("nan"))
    length_mm = _safe_float(row.get("length_mm"), 0.0)
    angle_deg = _safe_float(row.get("angle_deg"), float("nan"))
    if not all(math.isfinite(value) for value in (cx, cy, length_mm, angle_deg)):
        return None
    if length_mm <= 0:
        return None
    theta = math.radians(angle_deg)
    half = 0.5 * length_mm / max(float(mm_per_px), 1e-9)
    direction = np.array([math.cos(theta), math.sin(theta)])
    centre = np.array([cx, cy], dtype=float)
    return [(centre - direction * half).tolist(), (centre + direction * half).tolist()]


def nearest_detached_organ_candidate(rows, mask, mm_per_px: float, used_ids=()):
    """Return the closest unassigned detached-organ detector candidate."""
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    mask_centre = np.array([float(xs.mean()), float(ys.mean())])
    used = {str(value) for value in used_ids}
    best = None
    for row in rows or []:
        candidate_id = str(row.get("organ_id", ""))
        if candidate_id in used:
            continue
        line = organ_candidate_line(row, mm_per_px)
        if line is None:
            continue
        distance = float(np.linalg.norm(line_midpoint(line) - mask_centre))
        candidate = {
            "candidate_id": candidate_id,
            "line": line,
            "width_mm": min(max(_safe_float(row.get("width_mm"), 1.5), 0.5), 5.0),
            "distance_px": distance,
        }
        if best is None or distance < best[0]:
            best = (distance, candidate)
    return best[1] if best else None


def best_organ_candidate(rows, mask, mm_per_px: float):
    """Return the detector line that overlaps this corolla mask most strongly."""
    if mask is None or not np.any(mask):
        return None

    height, width = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    mask_centre = np.array([float(xs.mean()), float(ys.mean())])
    mask_diagonal = max(float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())), 1.0)
    best = None

    for row in rows or []:
        line = organ_candidate_line(row, mm_per_px)
        if line is None:
            continue
        first, second = (np.asarray(point, dtype=float) for point in line)
        centre = (first + second) / 2.0
        cx, cy = centre
        samples = np.linspace(first, second, 81)
        sample_x = np.rint(samples[:, 0]).astype(int)
        sample_y = np.rint(samples[:, 1]).astype(int)
        in_bounds = (
            (sample_x >= 0) & (sample_x < width)
            & (sample_y >= 0) & (sample_y < height)
        )
        overlap = 0.0
        if in_bounds.any():
            overlap = float((mask[sample_y[in_bounds], sample_x[in_bounds]] > 0).mean())
        centre_inside = (
            0 <= int(round(cx)) < width
            and 0 <= int(round(cy)) < height
            and mask[int(round(cy)), int(round(cx))] > 0
        )
        distance = float(np.linalg.norm(centre - mask_centre)) / mask_diagonal
        score = overlap + (0.35 if centre_inside else 0.0) - 0.08 * distance
        if best is None or score > best[0]:
            best = (score, {
                "candidate_id": row.get("organ_id", ""),
                "line": [first.tolist(), second.tolist()],
                "width_mm": min(max(_safe_float(row.get("width_mm"), 1.5), 0.5), 5.0),
                "overlap": overlap,
            })

    if best is not None and best[1]["overlap"] >= 0.12:
        return best[1]
    return thin_appendage_candidate(mask, mm_per_px)


def thin_appendage_candidate(mask, mm_per_px: float):
    """Seed an organ line from a narrow appendage attached to a broad mask."""
    binary = (mask > 0).astype(np.uint8)
    opening_radius = max(int(round(1.25 / max(float(mm_per_px), 1e-9))), 8)
    kernel_size = 2 * opening_radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    broad_core = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    residual = binary & (broad_core == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual, connectivity=8
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    best = None

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        ys, xs = np.where(labels == label)
        points = np.stack([xs, ys], axis=1).astype(float)
        centre = points.mean(axis=0)
        covariance = np.cov(points - centre, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = (points - centre) @ direction
        first_q, second_q = np.percentile(projection, [1.0, 99.0])
        length_px = float(second_q - first_q)
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(
            points.astype(np.float32)
        )
        minor = min(float(rect_width), float(rect_height))
        major = max(float(rect_width), float(rect_height))
        aspect = major / max(minor, 1e-9)
        length_mm = length_px * float(mm_per_px)
        if aspect < 3.0 or length_mm < 8.0:
            continue
        score = length_mm * aspect * math.sqrt(area)
        if best is None or score > best[0]:
            local_radius = float(np.percentile(distance[labels == label], 90))
            width_mm = min(max(2.0 * local_radius * float(mm_per_px), 0.5), 5.0)
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = [
                [[float(x), float(y)] for [[x, y]] in contour]
                for contour in contours
                if len(contour) >= 3
            ]
            best = (score, {
                "candidate_id": "mask-thin-appendage",
                "line": [
                    (centre + direction * first_q).tolist(),
                    (centre + direction * second_q).tolist(),
                ],
                "width_mm": width_mm,
                "overlap": 1.0,
                "polygons": polygons,
            })

    return best[1] if best is not None else None


def transform_seed_polygons(
    polygons, source_line, target_line, width_scale=1.0, image_shape=None
):
    """Transform a curved seed region with its draggable centre-line handles."""
    if not polygons or line_length(source_line) <= 1e-9 or line_length(target_line) <= 1e-9:
        return []
    source_start = np.asarray(source_line[0], dtype=float)
    source_vector = np.asarray(source_line[1], dtype=float) - source_start
    source_length = float(np.linalg.norm(source_vector))
    source_unit = source_vector / source_length
    source_normal = np.array([-source_unit[1], source_unit[0]])
    target_start = np.asarray(target_line[0], dtype=float)
    target_vector = np.asarray(target_line[1], dtype=float) - target_start
    target_length = float(np.linalg.norm(target_vector))
    target_unit = target_vector / target_length
    target_normal = np.array([-target_unit[1], target_unit[0]])
    height = width = None
    if image_shape is not None:
        height, width = image_shape[:2]

    transformed = []
    for polygon in polygons:
        output = []
        for point in polygon:
            relative = np.asarray(point, dtype=float) - source_start
            along = float(relative @ source_unit) / source_length
            across = float(relative @ source_normal) * float(width_scale)
            mapped = (
                target_start
                + target_unit * (along * target_length)
                + target_normal * across
            )
            if width is not None and height is not None:
                mapped[0] = np.clip(mapped[0], 0, width - 1)
                mapped[1] = np.clip(mapped[1], 0, height - 1)
            output.append(mapped.tolist())
        if len(output) >= 3:
            transformed.append(output)
    return transformed


def _mask_axis_coordinates(mask, base_xy, tip_xy):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    points = np.stack([xs, ys], axis=1).astype(float)
    base_point, _, unit, normal, length = _axis(base_xy, tip_xy)
    relative = points - base_point
    return points, relative @ unit, relative @ normal, base_point, unit, normal, length


def _cross_section(mask, base_xy, tip_xy, axis_position: float) -> list[list[float]]:
    coordinates = _mask_axis_coordinates(mask, base_xy, tip_xy)
    if coordinates is None:
        return [list(map(float, base_xy)), list(map(float, tip_xy))]
    _, longitudinal, transverse, base_point, unit, normal, length = coordinates
    axis_position = min(max(float(axis_position), float(longitudinal.min())), float(longitudinal.max()))
    selected = None
    for window in (1.5, 3.0, 6.0, 12.0, 24.0):
        candidate = np.abs(longitudinal - axis_position) <= window
        if int(candidate.sum()) >= 4:
            selected = candidate
            break
    if selected is None:
        selected = np.ones_like(longitudinal, dtype=bool)
    q0 = float(np.percentile(transverse[selected], 1.0))
    q1 = float(np.percentile(transverse[selected], 99.0))
    first = base_point + unit * axis_position + normal * q0
    second = base_point + unit * axis_position + normal * q1
    return [first.tolist(), second.tolist()]


def _maximum_span(mask, base_xy, tip_xy) -> list[list[float]]:
    coordinates = _mask_axis_coordinates(mask, base_xy, tip_xy)
    if coordinates is None:
        return [list(map(float, base_xy)), list(map(float, tip_xy))]
    _, longitudinal, transverse, _, _, _, length = coordinates
    valid = (longitudinal >= -0.10 * length) & (longitudinal <= 1.10 * length)
    if not bool(valid.any()):
        valid = np.ones_like(longitudinal, dtype=bool)
    bins = np.round(longitudinal[valid]).astype(int)
    q_values = transverse[valid]
    best_position = float(np.median(longitudinal[valid]))
    best_width = -1.0
    for value in np.unique(bins):
        values = q_values[bins == value]
        if len(values) < 4:
            continue
        width = float(np.percentile(values, 99) - np.percentile(values, 1))
        if width > best_width:
            best_width = width
            best_position = float(value)
    return _cross_section(mask, base_xy, tip_xy, best_position)


def automatic_measurement_lines(mask, base_xy, tip_xy, original_row=None):
    original_row = original_row or {}
    _, _, _, _, axis_length = _axis(base_xy, tip_xy)
    original_length_mm = _safe_float(
        original_row.get("corolla_length_ruler_mm", original_row.get("corolla_len_mm")),
        axis_length * base.MM_PX,
    )
    lobe_length_mm = _safe_float(original_row.get("flat_lobe_length_mm"), 0.22 * original_length_mm)
    lobe_fraction = min(max(lobe_length_mm / max(original_length_mm, 1e-9), 0.08), 0.55)
    throat_position = axis_length * (1.0 - lobe_fraction)
    return {
        "max_span": _maximum_span(mask, base_xy, tip_xy),
        "throat_span": _cross_section(mask, base_xy, tip_xy, throat_position),
        "mid_tube_span": _cross_section(mask, base_xy, tip_xy, 0.50 * throat_position),
        "basal_tube_span": _cross_section(mask, base_xy, tip_xy, 0.15 * throat_position),
    }


def ensure_measurement_lines(cs, mask, original_row=None):
    lines = cs.setdefault("measurement_lines", {})
    automatic = automatic_measurement_lines(mask, cs["axis_base"], cs["axis_tip"], original_row)
    for key, value in automatic.items():
        if key not in lines or len(lines[key]) != 2:
            lines[key] = value
    cs.setdefault("measurement_lines_changed", [])
    return lines


def _shape_descriptors(mask):
    area = float((mask > 0).sum())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    rw, rh = cv2.minAreaRect(contour)[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-9)
    return solidity, aspect


def shape_trait_values(mask, cs, mm_per_px: float):
    lines = cs.get("measurement_lines", {})
    base_point, tip_point, unit, _, axis_px = _axis(cs["axis_base"], cs["axis_tip"])
    max_span_px = line_length(lines.get("max_span"))
    throat_px = line_length(lines.get("throat_span"))
    middle_px = line_length(lines.get("mid_tube_span"))
    basal_px = line_length(lines.get("basal_tube_span"))
    throat_midpoint = line_midpoint(lines.get("throat_span"))
    tube_px = min(max(float((throat_midpoint - base_point) @ unit), 0.0), axis_px)
    lobe_px = max(axis_px - tube_px, 0.0)

    length_mm = axis_px * mm_per_px
    width_mm = max_span_px * mm_per_px
    throat_mm = throat_px * mm_per_px
    middle_mm = middle_px * mm_per_px
    basal_mm = basal_px * mm_per_px
    tube_mm = tube_px * mm_per_px
    lobe_mm = lobe_px * mm_per_px
    area_mm2 = float((mask > 0).sum()) * mm_per_px * mm_per_px
    fold_factor = 2.0 if cs.get("fold_state") == "folded_half" else 1.0
    mouth_mm = throat_mm * fold_factor / math.pi
    entrance_mm2 = math.pi * (mouth_mm / 2.0) ** 2
    solidity, aspect = _shape_descriptors(mask)

    values = {
        "corolla_len_mm": round(length_mm, 3),
        "corolla_length_ruler_mm": round(length_mm, 3),
        "corolla_width_mm": round(width_mm, 3),
        "corolla_max_span_ruler_mm": round(width_mm, 3),
        "corolla_area_mm2": round(area_mm2, 3),
        "corolla_area_ruler_mm2": round(area_mm2, 3),
        "wl_ratio": round(width_mm / max(length_mm, 1e-9), 4),
        "flat_lobe_length_mm": round(lobe_mm, 3),
        "flat_tube_length_mm": round(tube_mm, 3),
        "prov_tube_depth_mm": round(tube_mm, 3),
        "flat_throat_span_mm": round(throat_mm, 3),
        "flat_mid_tube_width_mm": round(middle_mm, 3),
        "flat_basal_tube_width_mm": round(basal_mm, 3),
        "prov_mouth_diam_mm": round(mouth_mm, 3),
        "prov_mouth_diameter_ruler_mm": round(mouth_mm, 3),
        "prov_entrance_area_mm2": round(entrance_mm2, 3),
        "mouth_to_tube_length_ratio": round(mouth_mm / max(tube_mm, 1e-9), 4),
        "flat_tube_taper_ratio": round(throat_mm / max(basal_mm, 1e-9), 4),
        "flat_tube_slenderness": round(tube_mm / max(middle_mm, 1e-9), 4),
        "flat_incision_rel": round(lobe_mm / max(length_mm, 1e-9), 4),
        "flat_throat_openness": round(throat_mm / max(width_mm, 1e-9), 4),
        "solidity": round(solidity, 4),
        "aspect": round(aspect, 4),
        "fold_state_auto": "folded_half" if cs.get("fold_state") == "folded_half" else "opened_or_broad",
        "mouth_proxy_assumption": (
            "2x_flat_span_over_pi" if cs.get("fold_state") == "folded_half" else "flat_span_over_pi"
        ),
    }
    if cs.get("flat_n_lobes") is not None:
        values["flat_n_lobes"] = int(cs["flat_n_lobes"])
        values["prov_n_lobes"] = int(cs["flat_n_lobes"])
    return values


def apply_polygon_edits(mask, edits):
    output = mask.astype(np.uint8).copy()
    for polygon in edits.get("add", []):
        cv2.fillPoly(output, [np.asarray(polygon, np.int32)], 1)
    for polygon in edits.get("subtract", []):
        cv2.fillPoly(output, [np.asarray(polygon, np.int32)], 0)
    return output


def _accepted_binary(mask, mm_per_px: float):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    accepted = np.zeros_like(mask, dtype=np.uint8)
    areas = []
    minimum_px = reviewed_spots.MIN_SPOT_AREA_MM2 / max(mm_per_px * mm_per_px, 1e-9)
    for index in range(1, count):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        if area_px < minimum_px:
            continue
        accepted[labels == index] = 1
        areas.append(area_px * mm_per_px * mm_per_px)
    return accepted, np.asarray(areas, dtype=float)


def colour_review_masks(raw, analysis_mask, box, cs, mm_per_px: float):
    x0, y0, x1, y1 = box
    image = raw[y0:y1, x0:x1]
    corolla = analysis_mask[y0:y1, x0:x1].astype(bool)
    lightness, a_channel, b_channel = base.channels(image)
    strong, weak, combined = reviewed_spots.spot_candidate_masks(a_channel, b_channel, corolla)
    guide, _, _ = reviewed_spots._accepted_spots(combined, strong, weak, mm_per_px)
    oxidized = reviewed_spots.oxidized_guide_mask(a_channel, lightness, corolla, strong, combined)
    oxidized, _, _ = reviewed_spots._accepted_oxidized(oxidized, mm_per_px)
    brown = ((a_channel > 6) & ((a_channel - b_channel) < -15) & corolla).astype(np.uint8)

    region_edits = cs.get("region_edits", {})
    output = {}
    for key, automatic in (("guide", guide), ("oxidized", oxidized), ("brown", brown)):
        local_edits = {"add": [], "subtract": []}
        for operation in ("add", "subtract"):
            for polygon in region_edits.get(key, {}).get(operation, []):
                local_edits[operation].append(
                    [[float(x) - x0, float(y) - y0] for x, y in polygon]
                )
        output[key] = apply_polygon_edits(automatic, local_edits) & corolla.astype(np.uint8)
    output["strong"] = strong.astype(np.uint8) & output["guide"]
    output["weak"] = weak.astype(np.uint8) & output["guide"]
    return output


def colour_trait_values(raw, analysis_mask, box, cs, mm_per_px: float):
    masks = colour_review_masks(raw, analysis_mask, box, cs, mm_per_px)
    x0, y0, x1, y1 = box
    corolla = analysis_mask[y0:y1, x0:x1].astype(bool)
    guide, areas = _accepted_binary(masks["guide"] & corolla, mm_per_px)
    oxidized, oxidized_areas = _accepted_binary(masks["oxidized"] & corolla & ~(guide > 0), mm_per_px)
    brown = (masks["brown"] > 0) & corolla
    guide_bool = guide > 0
    oxidized_bool = oxidized > 0
    corolla_px = int(corolla.sum())
    guide_px = int(guide_bool.sum())
    oxidized_px = int(oxidized_bool.sum())
    area_mm2 = corolla_px * mm_per_px * mm_per_px
    guide_area_mm2 = guide_px * mm_per_px * mm_per_px
    incl_area_mm2 = (guide_px + oxidized_px) * mm_per_px * mm_per_px

    ys, xs = np.where(guide_bool)
    base_point, _, unit, _, axis_px = _axis(cs["axis_base"], cs["axis_tip"])
    if len(xs):
        global_points = np.stack([xs + x0, ys + y0], axis=1).astype(float)
        positions = (global_points - base_point) @ unit
        guide_extent = float(np.clip(positions.max() / max(axis_px, 1e-9), 0.0, 1.0))
        guide_centroid = float(np.clip(positions.mean() / max(axis_px, 1e-9), 0.0, 1.0))
    else:
        guide_extent = 0.0
        guide_centroid = 0.0

    throat_midpoint = line_midpoint(cs.get("measurement_lines", {}).get("throat_span"))
    throat_position = float((throat_midpoint - base_point) @ unit)
    cy, cx = np.where(corolla)
    corolla_positions = (np.stack([cx + x0, cy + y0], axis=1).astype(float) - base_point) @ unit
    gy, gx = np.where(guide_bool)
    guide_positions = (np.stack([gx + x0, gy + y0], axis=1).astype(float) - base_point) @ unit if len(gx) else np.array([])
    tube_area = int((corolla_positions <= throat_position).sum())
    lobe_area = int((corolla_positions > throat_position).sum())
    tube_guide = int((guide_positions <= throat_position).sum()) if guide_positions.size else 0
    lobe_guide = int((guide_positions > throat_position).sum()) if guide_positions.size else 0
    brown_overlap = int((brown & guide_bool).sum())

    strong_px = int(((masks["strong"] > 0) & guide_bool).sum())
    weak_px = int(((masks["weak"] > 0) & guide_bool).sum())
    values = {
        "guide_area_mm2": round(guide_area_mm2, 3),
        "guide_cov_pct": round(100.0 * guide_px / max(corolla_px, 1), 3),
        "n_spots": int(len(areas)),
        "spot_density_cm2": round(len(areas) / max(area_mm2 / 100.0, 1e-9), 3),
        "guide_extent_rel": round(guide_extent, 4),
        "guide_present": int(100.0 * guide_px / max(corolla_px, 1) >= 0.5),
        "brown_frac": round(float(brown.sum()) / max(corolla_px, 1), 4),
        "degraded_flag": int(float(brown.sum()) / max(corolla_px, 1) > 0.10),
        "n_strong_spots": int(cv2.connectedComponents(
            (masks["strong"] > 0).astype(np.uint8), connectivity=8
        )[0] - 1),
        "n_weak_recovered_spots": int(cv2.connectedComponents(
            (masks["weak"] > 0).astype(np.uint8), connectivity=8
        )[0] - 1),
        "n_large_spot_clusters": int((areas >= reviewed_spots.LARGE_SPOT_CLUSTER_MM2).sum()),
        "guide_cov_strong_pct": round(100.0 * strong_px / max(corolla_px, 1), 3),
        "guide_cov_weak_recovered_pct": round(100.0 * weak_px / max(corolla_px, 1), 3),
        "mean_spot_area_mm2": round(float(areas.mean()), 4) if areas.size else 0.0,
        "median_spot_area_mm2": round(float(np.median(areas)), 4) if areas.size else 0.0,
        "max_spot_area_mm2": round(float(areas.max()), 4) if areas.size else 0.0,
        "brown_overlap_pct_of_spots": round(100.0 * brown_overlap / max(guide_px, 1), 3),
        "n_oxidized_recovered_spots": int(len(oxidized_areas)),
        "guide_area_incl_oxidized_mm2": round(incl_area_mm2, 3),
        "guide_cov_incl_oxidized_pct": round(100.0 * (guide_px + oxidized_px) / max(corolla_px, 1), 3),
        "guide_cov_tube_pct": round(100.0 * tube_guide / max(tube_area, 1), 3),
        "guide_cov_lobes_pct": round(100.0 * lobe_guide / max(lobe_area, 1), 3),
        "guide_centroid_rel": round(guide_centroid, 4),
        "spot_qc_source": "app_reviewed_regions",
    }
    return values, masks


def render_region_overlay(raw, analysis_mask, box, masks, target):
    x0, y0, x1, y1 = box
    crop = raw[y0:y1, x0:x1].copy()
    corolla = analysis_mask[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(corolla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (0, 150, 0), 2)
    region = masks[target] > 0
    colour = np.full_like(crop, REGION_TARGETS[target]["colour_bgr"])
    crop[region] = cv2.addWeighted(crop, 0.35, colour, 0.65, 0)[region]
    return crop
