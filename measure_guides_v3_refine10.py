# -*- coding: utf-8 -*-
"""Recover faint laid-out reproductive organs using geometry and layout priors.

The reviewed shimask sheets show that many true organs are pale grey or weakly
brown.  Earlier v3 stages rejected them immediately as paper because chroma was
low.  This layer keeps the colour-first detector, then adds a conservative
per-corolla search for long, narrow, locally darker structures near each flower.
The reviewed images are used only for evaluation; runtime input remains raw scans.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine9 as refine9  # noqa: F401  (installs prior stages)
from measure_guides_v3_core import associate_organ

_ORIGINAL_EXTERNAL = refine.external_candidates

_LAYOUT_SEARCH_MM = 20.0
_LAYOUT_MIN_LENGTH_MM = 6.0
_LAYOUT_MAX_LENGTH_MM = 35.0
_LAYOUT_MAX_WIDTH_MM = 4.5
_LAYOUT_MIN_ASPECT = 3.0
_LAYOUT_MAX_DISTANCE_MM = 17.0


def _line_candidates(image: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Return faint linear plant/debris pixels without requiring high chroma."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 21)
    local_dark = background - gray
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)

    # Faint styles are often only 1.5--3 grey levels darker than their local
    # paper background.  A weak colour alternative preserves pale brown tissue.
    pixels = (
        ((local_dark >= 1.35) & (gray <= 249.0))
        | ((local_dark >= 0.8) & (chroma >= 4.0) & (b >= 0.0))
    ).astype(np.uint8)
    pixels[excluded > 0] = 0
    pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    merged = np.zeros_like(pixels)
    length = max(11, int(round(2.8 / float(base.MM_PX))))
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, length)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (length, 2)),
        np.eye(length, dtype=np.uint8),
        np.fliplr(np.eye(length, dtype=np.uint8)),
    )
    for kernel in kernels:
        merged |= cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, kernel)
    return merged


def _candidate_rows(mask: np.ndarray, channels: tuple[np.ndarray, ...]) -> list[dict]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    rows: list[dict] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if max(w, h) < _LAYOUT_MIN_LENGTH_MM / float(base.MM_PX):
            continue
        crop = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
        full = np.zeros_like(mask)
        full[y : y + h, x : x + w][crop > 0] = 1
        features = refine._global_features(full, channels, "layout_low_chroma")
        if not features:
            continue
        length = float(features["rect_length_mm"])
        width = float(features["median_width_mm"])
        aspect = float(features["aspect"])
        local_dark = float(features["mean_local_dark"])
        chroma = float(features["median_chroma"])
        if not (_LAYOUT_MIN_LENGTH_MM <= length <= _LAYOUT_MAX_LENGTH_MM):
            continue
        if width > _LAYOUT_MAX_WIDTH_MM or aspect < _LAYOUT_MIN_ASPECT:
            continue
        if local_dark < 0.65 and chroma < 3.0:
            continue

        # Override the earlier colour-only paper rejection.  Geometry and local
        # contrast remain explicit, and these candidates always require QC.
        score = 0.50
        score += min((aspect - _LAYOUT_MIN_ASPECT) / 7.0, 1.0) * 0.15
        score += min(max(local_dark, 0.0) / 5.0, 1.0) * 0.13
        score += math.exp(-abs(length - 17.0) / 12.0) * 0.12
        score += min(chroma / 12.0, 1.0) * 0.06
        features.update(
            organ_type_auto="style_or_pistil_candidate",
            organ_confidence=round(min(score, 0.86), 3),
            organ_qc_reasons="faint_linear_near_corolla",
        )
        rows.append(features)
    return rows


def layout_candidates(
    image: np.ndarray,
    corolla_union: np.ndarray,
    corollas: list[dict],
    top: int,
    channels: tuple[np.ndarray, ...],
) -> list[dict]:
    excluded = cv2.dilate(
        (corolla_union > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    excluded[:top] = 1
    linear = _line_candidates(image, excluded)
    rows = _candidate_rows(linear, channels)
    output: list[dict] = []
    search_px = _LAYOUT_SEARCH_MM / float(base.MM_PX)
    for row in rows:
        association = associate_organ(row, corollas)
        distance = float(association["association_distance_mm"])
        if distance > _LAYOUT_MAX_DISTANCE_MM:
            continue
        nearest = int(association["nearest_corolla"])
        flower = corollas[nearest - 1] if 0 < nearest <= len(corollas) else None
        if flower is None:
            continue
        dx = float(row["cx"]) - float(flower.get("cx", 0.0))
        dy = abs(float(row["cy"]) - float(flower.get("cy", 0.0)))
        # Preparation places most detached organs beside the corresponding
        # corolla.  Also permit a short vertical offset for crowded sheets.
        side_prior = dx >= -0.15 * search_px and dx <= search_px and dy <= search_px
        if not side_prior:
            continue
        row.update(association)
        row["association_qc_required"] = 1
        output.append(row)
    return output


def external_candidates(
    image: np.ndarray,
    corolla_union: np.ndarray,
    corollas: list[dict],
    top: int,
    channels: tuple[np.ndarray, ...],
) -> list[dict]:
    colour_rows = _ORIGINAL_EXTERNAL(image, corolla_union, corollas, top, channels)
    faint_rows = layout_candidates(image, corolla_union, corollas, top, channels)
    combined = refine._deduplicate([*colour_rows, *faint_rows])

    # Keep at most one faint layout-prior candidate per flower.  Existing
    # colour-supported candidates retain priority in the downstream selector.
    selected: list[dict] = []
    faint_seen: set[int] = set()
    for row in sorted(
        combined,
        key=lambda item: (
            item.get("candidate_source") != "layout_low_chroma",
            float(item.get("organ_confidence", 0.0)),
        ),
        reverse=True,
    ):
        cid = int(row.get("nearest_corolla", 0) or 0)
        if row.get("candidate_source") == "layout_low_chroma":
            if cid in faint_seen:
                continue
            faint_seen.add(cid)
        selected.append(row)
    return selected


refine.external_candidates = external_candidates


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
