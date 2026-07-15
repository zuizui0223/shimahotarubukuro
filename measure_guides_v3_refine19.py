# -*- coding: utf-8 -*-
"""Replace the fixed corolla-edge erosion with a GrabCut colour-model boundary snap.

Against the corrected annotation-only shimask ground truth, a full-20-sheet
comparison of edge treatments gave (boundary recall / precision / F1):
    fixed 0.7 mm erosion   0.761 / 0.848 / 0.802
    GrabCut colour snap    0.779 / 0.877 / 0.824
GrabCut wins on BOTH recall and precision because its GMM colour model plus
smoothness prior locks the boundary onto the true tissue/paper edge, whereas a
uniform erosion cannot follow a variable-width paper halo. Pixel-level tissue
thresholding and per-corolla adaptive erosion were also tried and both lost, so
the coherent GrabCut boundary — not naive snapping — is what helps.

Keeps all refine14/refine18 traits. A conservative area-change guard falls back to
the original mask if GrabCut degenerates. shimask stays evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine18 as refine18  # noqa: F401  traits + refine17 chain

MM_PX = float(base.MM_PX)
_PAD_MM = 3.4
_SEED_MM = 1.2
_ITERS = 3
_AREA_LO, _AREA_HI = 0.70, 1.15   # accept GrabCut only within this area ratio

# GrabCut supplies the edge treatment, so disable the fixed erosion.
refine13.COROLLA_EDGE_ERODE_MM = 0.0

_ORIGINAL_PROCESS_SHEET = refine13.process_sheet


def _px(mm: float) -> int:
    return max(1, int(round(mm / MM_PX)))


def grabcut_refine(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Snap a cleaned corolla mask onto the true colour edge with GrabCut."""
    q = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(q > 0)
    if len(xs) < 50:
        return q
    pad = _px(_PAD_MM)
    x0, x1 = max(0, int(xs.min()) - pad), min(q.shape[1], int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(q.shape[0], int(ys.max()) + pad)
    sub = image[y0:y1, x0:x1]
    m = q[y0:y1, x0:x1]

    seed = _px(_SEED_MM)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seed + 1, 2 * seed + 1))
    sure_fg = cv2.erode(m, kernel)
    outside = cv2.dilate(m, kernel)

    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[outside > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(np.ascontiguousarray(sub), gc, None,
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    _ITERS, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return q
    res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    out = np.zeros_like(q)
    out[y0:y1, x0:x1] = res

    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    if n > 1:
        out = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # Fill holes so interior guide spots/folds are not punched out.
    ff = out.copy()
    cv2.floodFill(ff, np.zeros((out.shape[0] + 2, out.shape[1] + 2), np.uint8), (0, 0), 1)
    out = (out | (1 - ff)).astype(np.uint8)

    ratio = int(out.sum()) / max(int(q.sum()), 1)
    if int(out.sum()) < 50 or not (_AREA_LO <= ratio <= _AREA_HI):
        return q
    return out


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
        snapped = grabcut_refine(image, mask)
        if int(snapped.sum()) != int(mask.sum()):
            cv2.imencode(".png", snapped * 255)[1].tofile(str(mask_path))
            row = refine._recompute_traits(image, snapped, row, channels)
            prior = str(row.get("mask_qc_reasons", "") or "")
            row["mask_qc_reasons"] = "|".join(filter(None, [prior, "grabcut_edge_snap"]))
        updated.append(row)
    return updated, organs, qc_rows, cleanup


refine13.process_sheet = process_sheet


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
