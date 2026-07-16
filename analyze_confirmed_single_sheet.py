#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the established reviewed pipeline on one sheet with confirmed annotations.

Only the two previously unreliable inputs are replaced:
- red shimask outlines -> confirmed corolla masks
- green shimask traces -> confirmed reproductive-organ measurements

The established nectar-guide detector, ruler calibration, spot summaries, and
flat floral-trait measurements are reused without changing their algorithms.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review_spots as reviewed_spots
import measure_guides_review_traits as reviewed_traits
from evaluate_shimask_labels import load_bgr
from export_shimask_ground_truth import close_and_fill_boundaries
from measure_confirmed_shimask_traits import _confirmed_organs
from shimask_annotation_diff import annotation_masks


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _confirmed_components(red: np.ndarray, raw_shape: tuple[int, ...]) -> list[dict]:
    height, width = raw_shape[:2]
    resized = cv2.resize(red.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    masks = close_and_fill_boundaries(resized)
    components: list[dict] = []
    for source_id, mask in enumerate(masks, start=1):
        binary = (mask > 0).astype(np.uint8)
        metrics = v2.metrics(binary)
        if metrics is None or int(binary.sum()) < 500:
            continue
        moments = cv2.moments(binary)
        if moments["m00"] <= 0:
            continue
        components.append(
            {
                "mask": binary.astype(bool),
                "source_component_id": source_id,
                "split_piece": 1,
                "split_status": "human_confirmed_red_outline",
                "cx": moments["m10"] / moments["m00"],
                "cy": moments["m01"] / moments["m00"],
                "m": metrics,
            }
        )
    components.sort(key=lambda row: (int(float(row["cy"])) // 170, float(row["cx"])))
    if not components:
        raise RuntimeError("No confirmed corolla masks were recovered from red outlines")
    return components


def _annotation_overlay(raw: np.ndarray, red: np.ndarray, green: np.ndarray) -> np.ndarray:
    height, width = raw.shape[:2]
    red_full = cv2.resize(red.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    green_full = cv2.resize(green.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    overlay = raw.copy()
    # Preserve the annotation semantics exactly: red=confirmed corolla outline,
    # green=confirmed reproductive-organ trace. Nectar guides remain raw pixels.
    overlay[red_full] = (0, 0, 255)
    overlay[green_full] = (0, 255, 0)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shimask", type=Path, required=True)
    parser.add_argument("--folder", default="oshima")
    parser.add_argument("--out-dir", type=Path, default=Path("results_confirmed_single"))
    args = parser.parse_args()

    raw = load_bgr(args.raw)
    annotated = load_bgr(args.shimask)
    red, green = annotation_masks(annotated, raw)
    components = _confirmed_components(red, raw.shape)

    union = np.zeros(raw.shape[:2], np.uint8)
    for component in components:
        union[component["mask"]] = 255
    _, a_channel, b_channel = base.channels(raw)

    original_foreground = v2.foreground_v2
    original_corollas = v2.corollas

    def confirmed_foreground(_image: np.ndarray, _top: int):
        return union.copy(), a_channel.copy(), b_channel.copy()

    def confirmed_corollas(_filled: np.ndarray, auto_split: bool = True):
        del auto_split
        return components

    v2.foreground_v2 = confirmed_foreground
    v2.corollas = confirmed_corollas
    try:
        spot_summaries, spot_rows = reviewed_spots.write_spot_qc(
            args.raw, args.folder, args.out_dir, auto_split=False
        )
        trait_summaries, scale_info = reviewed_traits.write_flower_trait_qc(
            args.raw,
            args.folder,
            args.out_dir,
            spot_segmenter=reviewed_spots.reviewed_spot_segment,
            auto_split=False,
        )
    finally:
        v2.foreground_v2 = original_foreground
        v2.corollas = original_corollas

    if len(spot_summaries) != len(components) or len(trait_summaries) != len(components):
        raise RuntimeError(
            f"Confirmed-mask output mismatch: masks={len(components)} "
            f"spots={len(spot_summaries)} traits={len(trait_summaries)}"
        )

    combined_traits: list[dict] = []
    for corolla_id in range(1, len(components) + 1):
        combined_traits.append(
            {
                "island": base.ISLANDS.get(args.folder.lower(), (args.folder, ""))[0],
                "sheet": args.raw.stem,
                "corolla_id": corolla_id,
                "mask_source": "human_confirmed_red_outline",
                **spot_summaries[corolla_id],
                **trait_summaries[corolla_id],
            }
        )

    mm_per_px = float(scale_info["mm_per_px"])
    green_full = cv2.resize(
        green.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    organ_rows = _confirmed_organs(
        green_full,
        mm_per_px,
        args.raw.stem,
        base.ISLANDS.get(args.folder.lower(), (args.folder, ""))[0],
    )

    _write_csv(args.out_dir / "confirmed_reviewed_traits.csv", combined_traits)
    _write_csv(args.out_dir / "confirmed_reviewed_spots.csv", spot_rows)
    _write_csv(args.out_dir / "confirmed_reproductive_organs.csv", organ_rows)
    _write_csv(args.out_dir / "scale_calibration.csv", [scale_info])

    annotation = _annotation_overlay(raw, red, green)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", annotation, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError("Could not encode annotation overlay")
    encoded.tofile(str(args.out_dir / "annotation_overlay_red_mask_green_organs.jpg"))

    (args.out_dir / "PROVENANCE.txt").write_text(
        "corolla_mask=human_confirmed_red_outline\n"
        "reproductive_organs=human_confirmed_green_traces\n"
        "nectar_guide_detector=measure_guides_review_spots.py unchanged\n"
        "trait_measurement=measure_guides_review_traits.py unchanged\n",
        encoding="utf-8",
    )
    print(
        f"confirmed_corollas={len(components)} confirmed_organs={len(organ_rows)} "
        f"spots={len(spot_rows)} scale={mm_per_px} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
