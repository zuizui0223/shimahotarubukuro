# -*- coding: utf-8 -*-
"""Automatic-first v3 floral trait pipeline.

High-confidence corolla masks pass automatically. Only uncertain masks and
ambiguous reproductive-organ candidates are routed to ``qc_required.csv``.
Maximum width is provisional and opening width is deliberately deferred.
All outputs are ordinary files; no Streamlit state is used.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
from measure_guides_v3_core import (
    ASSOCIATION_ACCEPT,
    ORGAN_ACCEPT,
    associate_organ,
    classify_organ,
    component_features,
    corolla_confidence,
    pca_axis,
    projection_extents,
    skeleton_length_px,
    skeleton_nodes,
    skeletonize,
    touches_border,
)


def organ_candidates(
    image: np.ndarray,
    corolla_union: np.ndarray,
    top: int,
) -> list[dict]:
    """Extract and characterize detached organ candidates."""
    lc, a, b = base.channels(image)
    chroma = np.sqrt(a * a + b * b)
    candidate = (
        (lc < 248)
        & ((chroma > 2.2) | (b > 2) | (a > 2.5))
        & ~((lc < 115) & (chroma < 10))
    ).astype(np.uint8)
    candidate[:top] = 0
    exclusion = cv2.dilate(
        np.asarray(corolla_union, dtype=np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    candidate[exclusion > 0] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    merged = np.zeros_like(candidate)
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)),
        np.eye(11, dtype=np.uint8),
        np.fliplr(np.eye(11, dtype=np.uint8)),
    )
    for kernel in kernels:
        merged |= cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    output: list[dict] = []
    for component_id in range(1, n):
        area_mm2 = float(stats[component_id, cv2.CC_STAT_AREA]) * float(base.MM2_PX)
        if not 0.25 <= area_mm2 <= 120.0:
            continue
        mask = (labels == component_id).astype(np.uint8)
        features = component_features(mask)
        if not features:
            continue
        organ_type, confidence, reasons = classify_organ(features)
        features.update(
            organ_type_auto=organ_type,
            organ_confidence=round(confidence, 3),
            organ_qc_reasons="|".join(reasons),
        )
        output.append(features)
    output.sort(key=lambda row: (row["cy"], row["cx"]))
    return output


def _serializable(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"mask", "skeleton", "endpoints", "branches", "m"}
    }


def write_csv(path: Path, rows: Iterable[dict], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_sheet(
    path: str,
    folder: str,
    out_dir: str,
    loc_map: dict | None = None,
    auto_split: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run automatic-first extraction for one scan."""
    island, order = base.ISLANDS.get(folder, (folder, ""))
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    image = base.load_bgr(path)
    height, width = image.shape[:2]
    loc_map = loc_map or {}
    site_numbers, _, _ = base.site_numbers(filename)
    island_sites = sorted({number for isl, number in loc_map if isl == folder}) if loc_map else []
    if len(island_sites) == 1:
        site_numbers = island_sites
    if len(site_numbers) == 1:
        site = site_numbers[0]
        latitude, longitude = loc_map.get((folder, site), ("", ""))
        site_candidates = ""
    else:
        site = ""
        latitude = longitude = ""
        site_candidates = "|".join(map(str, site_numbers))

    top = v2.specimen_top(image)
    filled, a_channel, b_channel = v2.foreground_v2(image, top)
    corollas = v2.corollas(filled, auto_split)
    union = np.zeros((height, width), np.uint8)
    for component in corollas:
        union[np.asarray(component["mask"], dtype=bool)] = 1

    masks_dir = Path(out_dir) / "masks" / island / stem
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    trait_rows: list[dict] = []
    qc_rows: list[dict] = []

    for cid, component in enumerate(corollas, 1):
        mask = np.asarray(component["mask"], dtype=np.uint8)
        confidence = corolla_confidence(component)
        spots = base.spot_segment(a_channel, b_channel, mask.astype(bool))
        area_px = int(mask.sum())
        spot_px = int(spots.sum())
        area_mm2 = area_px * float(base.MM2_PX)
        coverage = spot_px / max(area_px, 1)
        n_spots, _, spot_stats, _ = cv2.connectedComponentsWithStats(
            spots.astype(np.uint8), 8
        )
        spot_count = sum(
            spot_stats[index, cv2.CC_STAT_AREA] * float(base.MM2_PX) >= 0.02
            for index in range(1, n_spots)
        )
        brown = (a_channel > 6) & ((a_channel - b_channel) < -15)
        brown_fraction = int((brown & mask.astype(bool)).sum()) / max(area_px, 1)
        rotated_corolla, rotated_spots = base.orient_base_tip(mask.astype(bool), spots.astype(bool))
        geometry = base.geometry(rotated_corolla)
        guide_extent = ""
        if rotated_spots.sum() > 10:
            guide_extent = (
                rotated_corolla.shape[0] - 1 - np.where(rotated_spots > 0)[0].min()
            ) / max(rotated_corolla.shape[0] - 1, 1)

        length_mm = round(float(geometry["length"]) * float(base.MM_PX), 2)
        row = {
            "island": island,
            "region_order": order,
            "sheet": stem,
            "site_no": site,
            "site_candidates": site_candidates,
            "site_lat": latitude,
            "site_lon": longitude,
            "corolla_id": cid,
            "cx": round(float(component["cx"])),
            "cy": round(float(component["cy"])),
            "source_component_id": component["source_component_id"],
            "split_piece": component["split_piece"],
            "split_status": component["split_status"],
            "corolla_len_mm": length_mm,
            "corolla_area_mm2": round(area_mm2, 1),
            "guide_area_mm2": round(spot_px * float(base.MM2_PX), 2),
            "guide_cov_pct": round(coverage * 100.0, 2),
            "n_spots": spot_count,
            "spot_density_cm2": round(spot_count / max(area_mm2 / 100.0, 1e-9), 2),
            "guide_extent_rel": round(float(guide_extent), 3) if guide_extent != "" else "",
            "guide_present": int(coverage * 100.0 >= 0.5),
            "brown_frac": round(brown_fraction, 3),
            "degraded_flag": int(brown_fraction > 0.10),
            **confidence,
        }
        trait_rows.append(row)
        if row["mask_qc_required"]:
            qc_rows.append(
                {
                    "record_type": "corolla",
                    "island": island,
                    "sheet": stem,
                    "record_id": f"C{cid}",
                    "confidence": row["mask_confidence"],
                    "reason": row["mask_qc_reasons"],
                    "suggested_action": "accept|exclude|split",
                }
            )

        cv2.imencode(".png", mask * 255)[1].tofile(str(masks_dir / f"C{cid}.png"))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = {
            "high": (0, 190, 0),
            "medium": (0, 165, 255),
            "low": (0, 0, 230),
        }[row["mask_confidence_label"]]
        cv2.drawContours(overlay, contours, -1, colour, 3)
        cv2.putText(
            overlay,
            f"C{cid} {row['mask_confidence']:.2f}",
            (int(component["cx"]) - 30, int(component["cy"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
        )

    organ_rows: list[dict] = []
    for oid, organ in enumerate(organ_candidates(image, union, top), 1):
        association = associate_organ(organ, corollas)
        qc_required = int(
            organ["organ_type_auto"] == "reproductive_organ_unknown"
            or (
                organ["organ_type_auto"] != "fragment_or_noise"
                and (
                    float(organ["organ_confidence"]) < ORGAN_ACCEPT
                    or association["association_qc_required"]
                )
            )
        )
        row = {
            "island": island,
            "sheet": stem,
            "organ_id": oid,
            **_serializable(organ),
            **association,
            "organ_qc_required": qc_required,
        }
        organ_rows.append(row)
        if qc_required:
            qc_rows.append(
                {
                    "record_type": "organ",
                    "island": island,
                    "sheet": stem,
                    "record_id": f"R{oid}",
                    "confidence": row["organ_confidence"],
                    "reason": row["organ_qc_reasons"],
                    "suggested_action": "pistil|stamen_bundle|unknown|noise",
                }
            )
        x, y = int(round(float(organ["cx"]))), int(round(float(organ["cy"])))
        cv2.circle(overlay, (x, y), 9, (255, 120, 0), -1)
        cv2.putText(
            overlay,
            f"R{oid} {organ['organ_type_auto']} {organ['organ_confidence']:.2f}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 120, 0),
            2,
        )

    overlays = Path(out_dir) / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    scale = min(1.0, 1900.0 / max(height, width))
    preview = cv2.resize(
        overlay,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    cv2.imencode(".png", preview)[1].tofile(str(overlays / f"{island}_{stem}.png"))
    return trait_rows, organ_rows, qc_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="results_v3")
    parser.add_argument("--locations", default="")
    parser.add_argument("--no-auto-split", action="store_true")
    arguments = parser.parse_args()

    out = Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    locations = base.load_locations(arguments.locations)
    traits: list[dict] = []
    organs: list[dict] = []
    qc: list[dict] = []
    for folder in sorted(os.listdir(arguments.data_root)):
        directory = Path(arguments.data_root) / folder
        if not directory.is_dir():
            continue
        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            sheet_traits, sheet_organs, sheet_qc = process_sheet(
                str(image_path),
                folder.lower(),
                str(out),
                locations,
                not arguments.no_auto_split,
            )
            traits.extend(sheet_traits)
            organs.extend(sheet_organs)
            qc.extend(sheet_qc)

    if not traits:
        raise SystemExit("No corollas detected")
    write_csv(out / "traits_v3.csv", traits)
    write_csv(out / "organs_v3.csv", organs)
    write_csv(
        out / "qc_required.csv",
        qc,
        [
            "record_type",
            "island",
            "sheet",
            "record_id",
            "confidence",
            "reason",
            "suggested_action",
        ],
    )
    print(
        f"corollas={len(traits)} organs={len(organs)} qc_required={len(qc)} -> {out}"
    )


if __name__ == "__main__":
    main()
