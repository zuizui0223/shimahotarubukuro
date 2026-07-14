# -*- coding: utf-8 -*-
"""Refine v3 corolla masks and detect laid-out style/pistil candidates.

This second-stage prototype starts from v3's 18 corolla detections, removes
long thin appendages from masks, distinguishes coloured plant tissue from paper
creases, and limits external organ candidates to plausible objects near a
corolla.  Stamen length/type is not inferred from scans where filaments and
anthers are not preserved as separable intact objects.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_v3 as v3
from measure_guides_v3_core import associate_organ, component_features, corolla_confidence


OPENING_DIAMETER_MM = 2.6
APPENDAGE_AREA_MM2 = 1.7
APPENDAGE_LENGTH_MM = 6.7
APPENDAGE_ASPECT = 4.0
EXTERNAL_MAX_DISTANCE_MM = 12.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    q = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(q, 8)
    if n <= 1:
        return q
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def detach_thin_appendages(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cut long, narrow branches from a broad corolla body.

    A large elliptical opening identifies the broad body.  Only connected
    pieces lost by that opening that are long, narrow and sufficiently large
    are removed; ordinary lobe margins are retained.
    """
    q = (np.asarray(mask) > 0).astype(np.uint8)
    diameter = max(9, int(round(OPENING_DIAMETER_MM / float(base.MM_PX))))
    if diameter % 2 == 0:
        diameter += 1
    opened = cv2.morphologyEx(
        q,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    )
    difference = q & (1 - opened)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(difference, 8)
    removed: list[np.ndarray] = []
    remove_union = np.zeros_like(q)
    area_min_px = APPENDAGE_AREA_MM2 / float(base.MM2_PX)
    length_min_px = APPENDAGE_LENGTH_MM / float(base.MM_PX)
    for label in range(1, n):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        upper = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < area_min_px or max(width, height) < length_min_px:
            continue
        crop = (labels[upper : upper + height, left : left + width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect_width, rect_height = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
        length = max(float(rect_width), float(rect_height))
        narrow = min(float(rect_width), float(rect_height))
        aspect = length / max(narrow, 1e-6)
        if length < length_min_px or aspect < APPENDAGE_ASPECT:
            continue
        full = np.zeros_like(q)
        full[upper : upper + height, left : left + width][crop > 0] = 1
        removed.append(full)
        remove_union |= full
    cleaned = _largest_component(q & (1 - remove_union))
    return cleaned, removed


def _lab_channels(image: np.ndarray) -> tuple[np.ndarray, ...]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    local_a = a - cv2.GaussianBlur(a, (0, 0), 31)
    local_b = b - cv2.GaussianBlur(b, (0, 0), 31)
    local_dark = cv2.GaussianBlur(light, (0, 0), 31) - light
    chroma = np.sqrt(a * a + b * b)
    return light, a, b, local_a, local_b, local_dark, chroma


def _candidate_colour(image_channels: tuple[np.ndarray, ...], mask: np.ndarray) -> dict:
    light, a, b, local_a, local_b, local_dark, chroma = image_channels
    selected = np.asarray(mask) > 0
    if not selected.any():
        return {
            "mean_light": 255.0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_local_a": 0.0,
            "mean_local_b": 0.0,
            "mean_local_dark": 0.0,
            "median_chroma": 0.0,
        }
    return {
        "mean_light": round(float(light[selected].mean()), 2),
        "mean_a": round(float(a[selected].mean()), 2),
        "mean_b": round(float(b[selected].mean()), 2),
        "mean_local_a": round(float(local_a[selected].mean()), 2),
        "mean_local_b": round(float(local_b[selected].mean()), 2),
        "mean_local_dark": round(float(local_dark[selected].mean()), 2),
        "median_chroma": round(float(np.median(chroma[selected])), 2),
    }


def classify_style_candidate(features: dict) -> tuple[str, float, str]:
    """Separate coloured style/pistil tissue from paper creases and debris."""
    chroma = float(features["median_chroma"])
    mean_b = float(features["mean_b"])
    length = float(features["rect_length_mm"])
    aspect = float(features["aspect"])
    width = float(features["median_width_mm"])
    if chroma < 5.7 or mean_b < 5.0:
        return "fragment_or_paper", 0.90, "low_plant_chroma"
    if length < 4.0 or length > 38.0 or aspect < 2.2 or width > 7.0:
        return "reproductive_organ_unknown", 0.48, "implausible_style_geometry"
    score = 0.52
    score += min(max((chroma - 5.7) / 12.0, 0.0), 1.0) * 0.18
    score += min(aspect / 8.0, 1.0) * 0.13
    score += math.exp(-abs(length - 18.0) / 14.0) * 0.10
    score += min(max((mean_b - 5.0) / 12.0, 0.0), 1.0) * 0.07
    return "style_or_pistil_candidate", round(min(score, 0.96), 3), "elongated_coloured_plant_tissue"


def _global_features(mask: np.ndarray, channels: tuple[np.ndarray, ...], source: str) -> dict:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return {}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1]
    features = component_features(crop)
    if not features:
        return {}
    features["cx"] = round(float(features["cx"]) + x0, 2)
    features["cy"] = round(float(features["cy"]) + y0, 2)
    features["endpoints"] = features["endpoints"] + np.array([x0, y0])
    features["branches"] = features["branches"] + np.array([x0, y0])
    features.update(_candidate_colour(channels, mask), candidate_source=source)
    label, confidence, reason = classify_style_candidate(features)
    features.update(
        organ_type_auto=label,
        organ_confidence=confidence,
        organ_qc_reasons=reason,
        stamen_status="not_recoverable_as_separate_intact_structure",
    )
    return features


def external_candidates(
    image: np.ndarray,
    corolla_union: np.ndarray,
    corollas: list[dict],
    top: int,
    channels: tuple[np.ndarray, ...],
) -> list[dict]:
    """Detect only elongated, coloured objects close to a cleaned corolla."""
    light, a, b, local_a, local_b, local_dark, _ = channels
    candidate = (
        ((local_a > 1.0) & (local_b > 2.0) & (local_dark > 2.0) & (b > 4.0))
        | ((local_a > 4.0) & (local_dark > 3.0) & (a > 2.0))
        | ((local_b > 6.0) & (local_dark > 5.0) & (b > 7.0) & (a > -3.0))
        | ((local_dark > 3.0) & (b > 3.0) & (light < 245.0) & (a > -4.0))
    ).astype(np.uint8)
    candidate[:top] = 0
    candidate[cv2.dilate(corolla_union, np.ones((3, 3), np.uint8)) > 0] = 0
    candidate[:, : max(40, int(image.shape[1] * 0.02))] = 0
    candidate[:, -max(50, int(image.shape[1] * 0.03)) :] = 0
    candidate[-max(40, int(image.shape[0] * 0.02)) :] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    merged = np.zeros_like(candidate)
    length = max(17, int(round(3.4 / float(base.MM_PX))))
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, length)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3)),
        np.eye(length, dtype=np.uint8),
        np.fliplr(np.eye(length, dtype=np.uint8)),
    ):
        merged |= cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    output: list[dict] = []
    area_min = 0.7 / float(base.MM2_PX)
    for label in range(1, n):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        upper = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < area_min or max(width, height) < 4.0 / float(base.MM_PX):
            continue
        crop = (labels[upper : upper + height, left : left + width] == label).astype(np.uint8)
        full = np.zeros_like(merged)
        full[upper : upper + height, left : left + width][crop > 0] = 1
        features = _global_features(full, channels, "external")
        if not features:
            continue
        if features["organ_type_auto"] == "fragment_or_paper":
            continue
        if float(features["rect_length_mm"]) < 4.0 or float(features["rect_length_mm"]) > 38.0:
            continue
        if float(features["aspect"]) < 2.2 or float(features["median_width_mm"]) > 7.0:
            continue
        association = associate_organ(features, corollas)
        if float(association["association_distance_mm"]) > EXTERNAL_MAX_DISTANCE_MM:
            continue
        features.update(association)
        output.append(features)
    return output


def _deduplicate(candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            row.get("candidate_source") == "attached_mask_appendage",
            float(row.get("organ_confidence", 0.0)),
            float(row.get("association_confidence", 0.0)),
        ),
        reverse=True,
    )
    kept: list[dict] = []
    duplicate_distance_px = 4.0 / float(base.MM_PX)
    for row in ordered:
        if any(
            math.hypot(float(row["cx"]) - float(other["cx"]), float(row["cy"]) - float(other["cy"]))
            < duplicate_distance_px
            for other in kept
        ):
            continue
        kept.append(row)
    return kept


def _recompute_traits(image: np.ndarray, mask: np.ndarray, row: dict, channels: tuple[np.ndarray, ...]) -> dict:
    _, a, b, _, _, _, _ = channels
    spots = base.spot_segment(a, b, mask.astype(bool))
    area_px = int(mask.sum())
    spot_px = int(spots.sum())
    area_mm2 = area_px * float(base.MM2_PX)
    coverage = spot_px / max(area_px, 1)
    rotated, rotated_spots = base.orient_base_tip(mask.astype(bool), spots.astype(bool))
    geometry = base.geometry(rotated)
    guide_extent = ""
    if rotated_spots.sum() > 10:
        guide_extent = (rotated.shape[0] - 1 - np.where(rotated_spots > 0)[0].min()) / max(rotated.shape[0] - 1, 1)
    n_spots, _, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), 8)
    spot_count = sum(
        stats[index, cv2.CC_STAT_AREA] * float(base.MM2_PX) >= 0.02
        for index in range(1, n_spots)
    )
    brown = (a > 6) & ((a - b) < -15)
    measured = v2.metrics(mask)
    component = {"mask": mask.astype(bool), "split_status": row.get("split_status", ""), "m": measured}
    confidence = corolla_confidence(component)
    moments = cv2.moments(mask)
    updated = dict(row)
    updated.update(
        cx=round(moments["m10"] / moments["m00"]) if moments["m00"] else row.get("cx", ""),
        cy=round(moments["m01"] / moments["m00"]) if moments["m00"] else row.get("cy", ""),
        corolla_len_mm=round(float(geometry["length"]) * float(base.MM_PX), 2),
        corolla_area_mm2=round(area_mm2, 1),
        guide_area_mm2=round(spot_px * float(base.MM2_PX), 2),
        guide_cov_pct=round(coverage * 100.0, 2),
        n_spots=spot_count,
        spot_density_cm2=round(spot_count / max(area_mm2 / 100.0, 1e-9), 2),
        guide_extent_rel=round(float(guide_extent), 3) if guide_extent != "" else "",
        guide_present=int(coverage * 100.0 >= 0.5),
        brown_frac=round(int((brown & mask.astype(bool)).sum()) / max(area_px, 1), 3),
        **confidence,
    )
    return updated


def process_sheet(path: str, folder: str, out_dir: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run v3 detection, then refine masks and organ candidates."""
    initial_traits, _, initial_qc = v3.process_sheet(path, folder, out_dir, {}, True)
    image = base.load_bgr(path)
    top = v2.specimen_top(image)
    channels = _lab_channels(image)
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    stem = Path(path).stem
    mask_dir = Path(out_dir) / "masks" / island / stem

    cleaned_components: list[dict] = []
    cleaned_traits: list[dict] = []
    appendage_rows: list[dict] = []
    organ_candidates: list[dict] = []
    union = np.zeros(image.shape[:2], np.uint8)
    for row in initial_traits:
        cid = int(row["corolla_id"])
        mask = (cv2.imdecode(np.fromfile(mask_dir / f"C{cid}.png", np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        cleaned, removed = detach_thin_appendages(mask)
        union[cleaned > 0] = 1
        measured = v2.metrics(cleaned)
        moments = cv2.moments(cleaned)
        component = {
            "mask": cleaned.astype(bool),
            "split_status": row.get("split_status", ""),
            "m": measured,
            "cx": moments["m10"] / moments["m00"] if moments["m00"] else row.get("cx", 0),
            "cy": moments["m01"] / moments["m00"] if moments["m00"] else row.get("cy", 0),
        }
        cleaned_components.append(component)
        refined = _recompute_traits(image, cleaned, row, channels)
        if removed:
            refined["mask_confidence"] = min(float(refined["mask_confidence"]), 0.74)
            refined["mask_confidence_label"] = "medium"
            refined["mask_qc_required"] = 1
            prior = str(refined.get("mask_qc_reasons", "") or "")
            refined["mask_qc_reasons"] = "|".join(filter(None, [prior, "thin_appendage_removed"]))
        cleaned_traits.append(refined)
        cv2.imencode(".png", cleaned * 255)[1].tofile(str(mask_dir / f"C{cid}.png"))
        for removed_mask in removed:
            features = _global_features(removed_mask, channels, "attached_mask_appendage")
            if not features:
                continue
            features.update(
                nearest_corolla=cid,
                association_confidence=0.98,
                association_distance_mm=0.0,
                association_angle_deg=0.0,
                association_qc_required=0,
            )
            appendage_rows.append(
                {
                    "island": island,
                    "sheet": stem,
                    "corolla_id": cid,
                    "removed_area_mm2": round(int(removed_mask.sum()) * float(base.MM2_PX), 2),
                    "classification": features["organ_type_auto"],
                    "confidence": features["organ_confidence"],
                    "reason": features["organ_qc_reasons"],
                }
            )
            if features["organ_type_auto"] != "fragment_or_paper":
                organ_candidates.append(features)

    organ_candidates.extend(external_candidates(image, union, cleaned_components, top, channels))
    organ_candidates = _deduplicate(organ_candidates)
    # The preparation normally places one style/pistil beside a corolla. Keep at
    # most two plausible candidates per corolla for initial QC, rather than
    # flooding the user with paper fibres.
    grouped: dict[int, list[dict]] = {}
    for candidate in organ_candidates:
        grouped.setdefault(int(candidate["nearest_corolla"]), []).append(candidate)
    selected: list[dict] = []
    for records in grouped.values():
        records.sort(
            key=lambda row: float(row["organ_confidence"]) * float(row["association_confidence"]),
            reverse=True,
        )
        selected.extend(records[:2])

    organ_rows: list[dict] = []
    qc_rows = [row for row in initial_qc if row.get("record_type") == "corolla"]
    for oid, candidate in enumerate(sorted(selected, key=lambda row: (row["cy"], row["cx"])), 1):
        serial = {
            key: value
            for key, value in candidate.items()
            if key not in {"mask", "skeleton", "endpoints", "branches"}
        }
        qc_required = int(
            candidate["candidate_source"] == "external"
            or float(candidate["organ_confidence"]) < 0.75
        )
        serial.update(
            island=island,
            sheet=stem,
            organ_id=oid,
            organ_qc_required=qc_required,
        )
        organ_rows.append(serial)
        if qc_required:
            qc_rows.append(
                {
                    "record_type": "organ",
                    "island": island,
                    "sheet": stem,
                    "record_id": f"R{oid}",
                    "confidence": candidate["organ_confidence"],
                    "reason": candidate["organ_qc_reasons"],
                    "suggested_action": "accept_style_or_pistil|unknown|noise",
                }
            )

    overlay = image.copy()
    for row, component in zip(cleaned_traits, cleaned_components):
        mask = np.asarray(component["mask"], np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = (0, 190, 0) if not int(row["mask_qc_required"]) else (0, 165, 255)
        cv2.drawContours(overlay, contours, -1, colour, 3)
        cv2.putText(
            overlay,
            f"C{int(row['corolla_id'])} {float(row['mask_confidence']):.2f}",
            (int(component["cx"]) - 30, int(component["cy"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
        )
    for row in organ_rows:
        x, y = int(round(float(row["cx"]))), int(round(float(row["cy"])))
        cv2.circle(overlay, (x, y), 9, (255, 80, 0), -1)
        cv2.putText(
            overlay,
            f"R{int(row['organ_id'])} C{int(row['nearest_corolla'])} {float(row['organ_confidence']):.2f}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 0),
            2,
        )
    overlays = Path(out_dir) / "overlays"
    scale = min(1.0, 1900.0 / max(image.shape[:2]))
    preview = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", preview)[1].tofile(str(overlays / f"{island}_{stem}.png"))
    return cleaned_traits, organ_rows, qc_rows, appendage_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="results_v3")
    arguments = parser.parse_args()
    out = Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traits: list[dict] = []
    organs: list[dict] = []
    qc: list[dict] = []
    cleanup: list[dict] = []
    for folder in sorted(os.listdir(arguments.data_root)):
        directory = Path(arguments.data_root) / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            sheet_traits, sheet_organs, sheet_qc, sheet_cleanup = process_sheet(
                str(path), folder.lower(), str(out)
            )
            traits.extend(sheet_traits)
            organs.extend(sheet_organs)
            qc.extend(sheet_qc)
            cleanup.extend(sheet_cleanup)
    if not traits:
        raise SystemExit("No corollas detected")
    write_csv(out / "traits_v3.csv", traits)
    write_csv(out / "organs_v3.csv", organs)
    write_csv(out / "qc_required.csv", qc)
    write_csv(out / "mask_cleanup.csv", cleanup)
    print(
        f"corollas={len(traits)} organs={len(organs)} "
        f"qc_required={len(qc)} cleaned_appendages={len(cleanup)} -> {out}"
    )


if __name__ == "__main__":
    main()
