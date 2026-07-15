# -*- coding: utf-8 -*-
"""Select a corolla boundary from several GrabCut scales using image evidence only.

refine19 improved the mean boundary score with one fixed GrabCut seed.  Real scans
vary in halo width and tissue contrast, so refine20 generates several conservative
candidates and chooses the one whose boundary has the strongest tissue/paper edge.
shimask is never read by this module; it remains evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine19 as refine19

MM_PX = float(base.MM_PX)
_SEEDS_MM = (0.55, 0.8, 1.05, 1.3, 1.6)
_PAD_MM = 3.8
_AREA_LO, _AREA_HI = 0.68, 1.18
_ORIGINAL_PROCESS_SHEET = refine19.process_sheet


def _px(mm: float) -> int:
    return max(1, int(round(mm / MM_PX)))


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    q = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(q, 8)
    if n > 1:
        q = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ff = q.copy()
    cv2.floodFill(ff, np.zeros((q.shape[0] + 2, q.shape[1] + 2), np.uint8), (0, 0), 1)
    return (q | (1 - ff)).astype(np.uint8)


def _candidate(image: np.ndarray, mask: np.ndarray, seed_mm: float) -> np.ndarray:
    q = (mask > 0).astype(np.uint8)
    ys, xs = np.where(q > 0)
    if len(xs) < 50:
        return q
    pad = _px(_PAD_MM)
    x0, x1 = max(0, int(xs.min()) - pad), min(q.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(q.shape[0], int(ys.max()) + pad + 1)
    sub = np.ascontiguousarray(image[y0:y1, x0:x1])
    m = q[y0:y1, x0:x1]
    seed = _px(seed_mm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed + 1, 2 * seed + 1))
    sure_fg = cv2.erode(m, kernel)
    outside = cv2.dilate(m, kernel)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[outside > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(sub, gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return q
    res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    out = np.zeros_like(q)
    out[y0:y1, x0:x1] = res
    out = _largest_filled(out)
    ratio = int(out.sum()) / max(int(q.sum()), 1)
    if int(out.sum()) < 50 or not (_AREA_LO <= ratio <= _AREA_HI):
        return q
    return out


def boundary_evidence(image: np.ndarray, mask: np.ndarray, reference: np.ndarray) -> float:
    """Higher is better: strong local colour edge, coherent shape, modest area shift."""
    q = (mask > 0).astype(np.uint8)
    if int(q.sum()) < 50:
        return -1e9
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(q, cv2.MORPH_GRADIENT, k3) > 0
    if not np.any(boundary):
        return -1e9
    # Colour-gradient support exactly at the proposed contour.
    gradients = []
    for channel in cv2.split(lab):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(gx, gy))
    grad = np.sqrt(sum(g * g for g in gradients))
    edge_strength = float(np.median(grad[boundary]))

    # Direct colour separation across a narrow inner/outer ring.
    ring_px = _px(0.35)
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    inner = (q > 0) & (cv2.erode(q, kr) == 0)
    outer = (cv2.dilate(q, kr) > 0) & (q == 0)
    if np.any(inner) and np.any(outer):
        colour_gap = float(np.linalg.norm(np.median(lab[inner], axis=0) - np.median(lab[outer], axis=0)))
    else:
        colour_gap = 0.0

    ratio = int(q.sum()) / max(int((reference > 0).sum()), 1)
    area_penalty = abs(np.log(max(ratio, 1e-6)))
    contours, _ = cv2.findContours(q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) if contours else 0.0
    compactness_penalty = perimeter * perimeter / max(4.0 * np.pi * float(q.sum()), 1.0)
    return edge_strength + 2.0 * colour_gap - 18.0 * area_penalty - 0.8 * compactness_penalty


def multiscale_refine(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, str]:
    q = (mask > 0).astype(np.uint8)
    candidates: list[tuple[np.ndarray, str]] = [(q, "input")]
    for seed_mm in _SEEDS_MM:
        candidates.append((_candidate(image, q, seed_mm), f"seed_{seed_mm:.2f}mm"))
    scored = [(boundary_evidence(image, candidate, q), candidate, label)
              for candidate, label in candidates]
    score, best, label = max(scored, key=lambda item: item[0])
    return best, float(score), label


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
        selected, score, label = multiscale_refine(image, mask)
        row["boundary_model"] = label
        row["boundary_evidence_score"] = round(score, 4)
        if not np.array_equal(selected, mask):
            cv2.imencode(".png", selected * 255)[1].tofile(str(mask_path))
            row = refine._recompute_traits(image, selected, row, channels)
            prior = str(row.get("mask_qc_reasons", "") or "")
            row["mask_qc_reasons"] = "|".join(filter(None, [prior, "multiscale_edge_select"]))
        updated.append(row)
    return updated, organs, qc_rows, cleanup


refine13.process_sheet = process_sheet


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
