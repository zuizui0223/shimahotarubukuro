# -*- coding: utf-8 -*-
"""Guarded local expansion for corollas that remain under-segmented after refine19.

The accepted refine19 mask is kept unless a narrow exterior band is both
colour-continuous with the corolla rim and bounded by a stronger transition to
paper farther out. shimask is never read here; it remains evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine19 as refine19  # installs accepted GrabCut process

MM_PX = float(base.MM_PX)
_MAX_GROW_MM = 1.2
_INNER_RIM_MM = 0.45
_OUTER_PAPER_MM = 1.8
_MAX_AREA_RATIO = 1.12
_MIN_GAIN_FRACTION = 0.002
_MIN_EVIDENCE = 2.0

_ORIGINAL_PROCESS_SHEET = refine13.process_sheet


def _px(mm: float) -> int:
    return max(1, int(round(mm / MM_PX)))


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def _mean_colour(lab: np.ndarray, region: np.ndarray) -> np.ndarray | None:
    values = lab[region > 0]
    if len(values) < 20:
        return None
    return np.median(values, axis=0)


def _colour_distance(lab: np.ndarray, centre: np.ndarray) -> np.ndarray:
    delta = lab - centre.reshape(1, 1, 3)
    # L is less discriminating than chroma for pale flattened flowers.
    delta[..., 0] *= 0.55
    return np.sqrt(np.sum(delta * delta, axis=2))


def guarded_expand(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Expand only through tissue-like pixels in a narrow connected exterior band."""
    q = (np.asarray(mask) > 0).astype(np.uint8)
    area0 = int(q.sum())
    if area0 < 50:
        return q, 0.0, "too_small"

    grow_px = _px(_MAX_GROW_MM)
    rim_px = _px(_INNER_RIM_MM)
    paper_px = _px(_OUTER_PAPER_MM)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
    k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_px + 1, 2 * rim_px + 1))
    k_paper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * paper_px + 1, 2 * paper_px + 1))

    inner_rim = q - cv2.erode(q, k_rim)
    grow_band = cv2.dilate(q, k_grow) - q
    outer_band = cv2.dilate(q, k_paper) - cv2.dilate(q, k_grow)

    lab = _lab(image)
    tissue = _mean_colour(lab, inner_rim)
    paper = _mean_colour(lab, outer_band)
    if tissue is None or paper is None:
        return q, 0.0, "insufficient_colour_samples"

    tissue_to_paper = float(np.linalg.norm((tissue - paper) * np.array([0.55, 1.0, 1.0], np.float32)))
    if tissue_to_paper < 5.0:
        return q, tissue_to_paper, "weak_tissue_paper_separation"

    d_tissue = _colour_distance(lab, tissue)
    d_paper = _colour_distance(lab, paper)
    # A pixel must be appreciably closer to the tissue model than to paper.
    eligible = ((grow_band > 0) & (d_tissue + 1.5 < d_paper)).astype(np.uint8)

    # Keep only eligible regions connected to the accepted mask.
    joined = ((q > 0) | (eligible > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    if n <= 1:
        return q, 0.0, "no_connected_growth"
    overlap = []
    for label in range(1, n):
        overlap.append(int(np.logical_and(labels == label, q > 0).sum()))
    keep_label = 1 + int(np.argmax(overlap))
    candidate = (labels == keep_label).astype(np.uint8)

    area1 = int(candidate.sum())
    gain = area1 - area0
    ratio = area1 / max(area0, 1)
    if gain < max(10, int(round(area0 * _MIN_GAIN_FRACTION))):
        return q, 0.0, "negligible_growth"
    if ratio > _MAX_AREA_RATIO:
        return q, 0.0, "area_guard"

    # Evidence: newly added pixels should favour tissue, while the new exterior
    # rim should favour paper. This rejects folds and interior-colour leakage.
    added = (candidate > q).astype(np.uint8)
    new_outer = cv2.dilate(candidate, k_rim) - candidate
    added_margin = float(np.median((d_paper - d_tissue)[added > 0])) if added.any() else -999.0
    outer_margin = float(np.median((d_tissue - d_paper)[new_outer > 0])) if new_outer.any() else -999.0
    evidence = min(added_margin, outer_margin)
    if evidence < _MIN_EVIDENCE:
        return q, evidence, "weak_growth_evidence"

    return candidate, evidence, "accepted"


def process_sheet(path: str, folder: str, out_dir: str):
    traits, organs, qc_rows, cleanup = _ORIGINAL_PROCESS_SHEET(path, folder, out_dir)
    image = base.load_bgr(path)
    channels = refine._lab_channels(image)
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    stem = Path(path).stem
    mask_dir = Path(out_dir) / "masks" / island / stem

    updated: list[dict] = []
    for row in traits:
        cid = int(row["corolla_id"])
        mask_path = mask_dir / f"C{cid}.png"
        mask = (cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        grown, evidence, status = guarded_expand(image, mask)
        row["local_growth_status"] = status
        row["local_growth_evidence"] = round(float(evidence), 4)
        if status == "accepted" and int(grown.sum()) != int(mask.sum()):
            cv2.imencode(".png", grown * 255)[1].tofile(str(mask_path))
            row = refine._recompute_traits(image, grown, row, channels)
            prior = str(row.get("mask_qc_reasons", "") or "")
            row["mask_qc_reasons"] = "|".join(filter(None, [prior, "guarded_local_growth"]))
        updated.append(row)
    return updated, organs, qc_rows, cleanup


refine13.process_sheet = process_sheet


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
