# -*- coding: utf-8 -*-
"""Convert human-reviewed shimask strokes into existing pipeline inputs.

Only the two input seams are implemented here:
- red outline -> v2-compatible corolla components
- green trace -> reviewed-organ rows

The annotation strokes are recovered from the pixel difference between the raw
scan and the reviewed shimask image. Natural purple nectar guides occur in both
images and therefore are not eligible annotation pixels.

Nectar-guide extraction and floral-trait measurement remain in the established
review modules and are not reimplemented here.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2

# A hand-drawn stroke is defined by being ADDED, not by an absolute colour, so we
# detect the DIRECTIONAL colour-dominance INCREASE of the reviewed image over the
# raw scan (red-ward for the corolla outline, green-ward for the organs). The
# natural purple nectar guide is present in both images and cancels; a purple
# corolla can therefore never read as red. Two robustness pieces make the
# difference reliable on all 20 sheets:
#   * ECC registration aligns the raw to the preview, so a resize/offset -- the
#     case that made niijiama2-4 look "misaligned" and where a resize-only
#     difference collapses to zero corollas -- does not corrupt the difference; and
#   * hysteresis (pure OpenCV, connected-component seeded) keeps the faint part of
#     a stroke -- e.g. where it crosses the guide-dense mouth -- connected into one
#     continuous loop instead of a fragmented line.
_RED_LOW, _RED_HIGH = 14.0, 40.0
_GREEN_LOW, _GREEN_HIGH = 16.0, 42.0


def _raw_at_annotation_resolution(raw: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Resize the raw scan to the reviewed image dimensions for direct differencing."""
    if raw.ndim != 3 or annotated.ndim != 3:
        raise ValueError("raw and annotated must be BGR colour images")
    height, width = annotated.shape[:2]
    if raw.shape[:2] == (height, width):
        return raw.copy()
    interpolation = cv2.INTER_AREA if raw.shape[0] >= height and raw.shape[1] >= width else cv2.INTER_LINEAR
    return cv2.resize(raw, (width, height), interpolation=interpolation)


def _register_raw_to_preview(raw: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Resize the raw scan to the preview, then ECC-refine the alignment."""
    ref = _raw_at_annotation_resolution(raw, annotated)
    g_ann = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        cv2.findTransformECC(g_ann, g_ref, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        ref = cv2.warpAffine(
            ref, warp, (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        pass  # fall back to the plain resize alignment
    return ref


def _remove_tiny_components(mask: np.ndarray, minimum_pixels: int = 3) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels:
            cleaned[labels == index] = 1
    return cleaned


def _hysteresis(gain: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak-threshold components that contain at least one strong-threshold seed."""
    weak = (gain >= low).astype(np.uint8)
    strong = gain >= high
    _, labels = cv2.connectedComponents(weak, 8)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(weak)
    return np.isin(labels, keep).astype(np.uint8)


def stroke_masks(raw: np.ndarray, annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hand-drawn red and green masks at annotation resolution, from the
    directional colour-dominance INCREASE of the reviewed image over the raw scan.

    Natural purple nectar guides are present in both images, so their dominance
    does not increase and they cannot enter the red mask even though they are
    reddish. No fixed colour range is used.
    """
    if raw.ndim != 3 or annotated.ndim != 3:
        raise ValueError("raw and annotated must be BGR colour images")
    ref = _register_raw_to_preview(raw, annotated).astype(np.float32)
    review = annotated.astype(np.float32)
    ab, ag, ar = cv2.split(review)
    rb, rg, rr = cv2.split(ref)
    red_gain = (ar - np.maximum(ag, ab)) - (rr - np.maximum(rg, rb))
    green_gain = (ag - np.maximum(ar, ab)) - (rg - np.maximum(rr, rb))
    red = _hysteresis(red_gain, _RED_LOW, _RED_HIGH)
    green = _hysteresis(green_gain, _GREEN_LOW, _GREEN_HIGH)
    return _remove_tiny_components(red), _remove_tiny_components(green)


def stroke_colour_rows(raw: np.ndarray, annotated: np.ndarray) -> list[dict[str, object]]:
    """Summarise the measured RGB values of the recovered annotation pixels."""
    red, green = stroke_masks(raw, annotated)
    rows: list[dict[str, object]] = []
    rgb = annotated[:, :, ::-1]
    for name, mask in (("red_corolla_outline", red), ("green_reproductive_organ", green)):
        values = rgb[mask > 0]
        if values.size == 0:
            rows.append({"stroke": name, "n_pixels": 0})
            continue
        row: dict[str, object] = {"stroke": name, "n_pixels": int(values.shape[0])}
        for channel_index, channel_name in enumerate(("R", "G", "B")):
            channel = values[:, channel_index].astype(np.uint8)
            counts = np.bincount(channel, minlength=256)
            row[f"{channel_name}_mode"] = int(np.argmax(counts))
            row[f"{channel_name}_median"] = round(float(np.median(channel)), 1)
            row[f"{channel_name}_p05"] = round(float(np.percentile(channel, 5)), 1)
            row[f"{channel_name}_p95"] = round(float(np.percentile(channel, 95)), 1)
        rows.append(row)
    return rows


def write_stroke_colour_stats(raw: np.ndarray, annotated: np.ndarray, out_path: Path) -> None:
    rows = stroke_colour_rows(raw, annotated)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resize_nn(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def _odd_kernel_size(mm: float, *, minimum: int = 3) -> int:
    value = max(minimum, int(round(mm / float(base.MM_PX))))
    return value if value % 2 else value + 1


def _closed_red_regions(red: np.ndarray) -> list[np.ndarray]:
    """Fill regions enclosed by red strokes without shrinking their boundary.

    A hand-drawn outline can have gaps (e.g. over the guide-dense mouth). Close by
    a few millimetres to bridge them, then fill the interior of each CLOSED loop by
    flood-filling the background and inverting -- so text, the ruler and any open
    stroke, which enclose nothing, contribute no region. Morphological closing
    restores the loop's outer edge, so the filled boundary stays on the drawn line
    rather than being eroded inward.
    """
    kernel_size = _odd_kernel_size(3.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    sealed = cv2.morphologyEx((red > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    flood = sealed.copy()
    cv2.floodFill(flood, np.zeros((sealed.shape[0] + 2, sealed.shape[1] + 2), np.uint8), (0, 0), 1)
    filled = (sealed | (1 - flood)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    masks: list[np.ndarray] = []
    for index in range(1, count):
        if float(stats[index, cv2.CC_STAT_AREA]) * float(base.MM2_PX) < float(base.AREA_MM2_MIN):
            continue
        masks.append((labels == index).astype(np.uint8))
    return masks


def red_corolla_components(raw: np.ndarray, annotated: np.ndarray) -> list[dict]:
    """Human red outlines -> components matching ``v2.corollas`` output."""
    red_small, _ = stroke_masks(raw, annotated)
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
                "split_status": "shimask_red_outline_from_raw_difference",
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
    """Human green strokes -> rows matching the reviewed-organ contract.

    Organs are elongated strokes laid out BESIDE the corollas on bare paper. The
    raw-difference can also leave faint green speckle inside a corolla or on the
    ruler ticks, so we drop the ruler band, drop components sitting inside a
    corolla outline, and require a genuinely elongated bar.
    """
    red_small, green_small = stroke_masks(raw, annotated)
    green = _resize_nn(green_small, raw.shape[:2])
    green[: v2.specimen_top(raw)] = 0  # ruler band residue
    kernel_size = _odd_kernel_size(0.35)
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )

    corolla_union = np.zeros(raw.shape[:2], np.uint8)
    for corolla in _closed_red_regions(_resize_nn(red_small, raw.shape[:2])):
        corolla_union |= corolla

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(green, 8)
    rows: list[dict] = []
    for index in range(1, count):
        cx0, cy0 = int(round(centroids[index][0])), int(round(centroids[index][1]))
        if corolla_union[min(max(cy0, 0), raw.shape[0] - 1), min(max(cx0, 0), raw.shape[1] - 1)] > 0:
            continue  # speckle inside a corolla, not a laid-out organ
        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        rect_len = max(rect_w, rect_h)
        rect_wid = max(min(rect_w, rect_h), 1.0)
        # An organ stroke is an elongated bar; reject small round speckle noise.
        if rect_len * float(base.MM_PX) < 6.0 or (rect_len / rect_wid) < 2.2:
            continue
        skeleton = _skeletonize(component)
        length_px = _skeleton_length_px(skeleton)
        if length_px <= 0:
            continue
        (x1, y1), (x2, y2), chord_px = _principal_endpoints(skeleton)
        cx, cy = map(float, centroids[index])
        length_mm = length_px * float(base.MM_PX)
        chord_mm = chord_px * float(base.MM_PX)
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        width_mm = area_px * float(base.MM2_PX) / max(length_mm, 1e-9)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        rows.append(
            {
                "cx": round(cx, 2), "cy": round(cy, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "length_mm": round(length_mm, 2),
                "skeleton_length_mm": round(length_mm, 2),
                "endpoint_distance_mm": round(chord_mm, 2),
                "width_mm": round(width_mm, 2),
                "aspect": round(length_mm / max(width_mm, 1e-6), 2),
                "angle_deg": round(angle, 2),
                "score": round(1000.0 - len(rows), 3),
                "organ_type_auto": "reviewed_reproductive_organ",
                "organ_type_FILL": "", "exclude_FILL": "",
                "detection_source": "shimask_green_stroke_from_raw_difference",
                "nearest_corolla_hint": "", "association_confirmed": 1,
                "visibility_note": "human-reviewed green stroke recovered by raw-image difference; path length measured on skeleton",
            }
        )
    rows.sort(key=lambda row: (float(row["cy"]), float(row["cx"])))
    return rows


def write_annotation_overlay(raw: np.ndarray, annotated: np.ndarray, out_path: Path) -> None:
    """Write raw + recovered red/green annotation strokes only; never draw guides."""
    red_small, green_small = stroke_masks(raw, annotated)
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
