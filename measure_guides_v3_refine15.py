# -*- coding: utf-8 -*-
"""Refine14 plus adaptive recovery of true corolla tissue at mask edges.

The fixed 0.6-mm erosion in refine13 gives high precision but leaves the predicted
boundary systematically inside the reviewed boundary on several sheets.  This
stage does not blindly dilate masks.  It searches only a narrow ring around each
automatic mask and restores pixels that are connected to the mask and supported
by raw-image plant signal (local darkness and/or chroma).  ``shimask`` remains
strictly evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine14 as refine14  # installs trait + organ extensions

MM_PX = float(base.MM_PX)
SEARCH_MM = 0.85
MAX_AREA_GAIN = 0.10

_ORIGINAL_PROCESS_SHEET = refine13.process_sheet


def _px(mm: float) -> int:
    return max(1, int(round(mm / MM_PX)))


def recover_supported_edge(image: np.ndarray, mask: np.ndarray, channels: tuple[np.ndarray, ...]) -> np.ndarray:
    """Recover connected tissue pixels in a narrow exterior ring.

    The criterion is intentionally conservative: pale tissue may be supported by
    local darkness, while strongly coloured tissue may be supported by chroma.
    Neutral bright paper and broad shadows are rejected.  Expansion proceeds
    geodesically from the current mask so isolated handwriting/debris cannot enter.
    """
    q = (np.asarray(mask) > 0).astype(np.uint8)
    if int(q.sum()) < 20:
        return q

    light, _a, b, _local_a, local_b, local_dark, chroma = channels
    radius = _px(SEARCH_MM)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    search = cv2.dilate(q, kernel)

    # Tissue support from the raw scan.  The yellow/brown b* term keeps very pale
    # reproductive/flower tissue while local-dark and chroma reject white paper.
    support = (
        (light < 251.5)
        & (
            ((local_dark > 0.75) & (chroma > 1.7))
            | (chroma > 3.4)
            | ((local_b > 1.2) & (b > 2.3) & (local_dark > 0.35))
        )
    ).astype(np.uint8)
    support &= search

    # Remove isolated scanner noise but keep one-pixel continuity along margins.
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    grown = q.copy()
    step_kernel = np.ones((3, 3), np.uint8)
    for _ in range(radius):
        frontier = cv2.dilate(grown, step_kernel) & support & search
        updated = grown | frontier
        if np.array_equal(updated, grown):
            break
        grown = updated

    # Keep only the component containing the original centroid.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, 8)
    if n > 1:
        ys, xs = np.where(q > 0)
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))
        keep = int(labels[min(max(cy, 0), q.shape[0] - 1), min(max(cx, 0), q.shape[1] - 1)])
        if keep > 0:
            grown = (labels == keep).astype(np.uint8)

    original_area = int(q.sum())
    gain = (int(grown.sum()) - original_area) / max(original_area, 1)
    if gain < 0 or gain > MAX_AREA_GAIN:
        return q
    return grown


def process_sheet(path: str, folder: str, out_dir: str):
    traits, organs, qc_rows, cleanup = _ORIGINAL_PROCESS_SHEET(path, folder, out_dir)
    image = base.load_bgr(path)
    channels = refine._lab_channels(image)
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    stem = Path(path).stem
    mask_dir = Path(out_dir) / "masks" / island / stem

    updated_traits: list[dict] = []
    changed_ids: set[int] = set()
    for row in traits:
        cid = int(row["corolla_id"])
        mask_path = mask_dir / f"C{cid}.png"
        mask = (cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        recovered = recover_supported_edge(image, mask, channels)
        if int(recovered.sum()) > int(mask.sum()):
            changed_ids.add(cid)
            cv2.imencode(".png", recovered * 255)[1].tofile(str(mask_path))
            row = refine._recompute_traits(image, recovered, row, channels)
            prior = str(row.get("mask_qc_reasons", "") or "")
            row["mask_qc_reasons"] = "|".join(filter(None, [prior, "image_supported_edge_recovery"]))
            row["edge_recovery_area_pct"] = round(
                100.0 * (int(recovered.sum()) - int(mask.sum())) / max(int(mask.sum()), 1), 3
            )
        else:
            row = dict(row)
            row["edge_recovery_area_pct"] = 0.0
        updated_traits.append(row)

    # Replace stale corolla QC rows for changed masks with explicit traceability.
    for row in qc_rows:
        if row.get("record_type") == "corolla":
            try:
                cid = int(str(row.get("record_id", "")).lstrip("C"))
            except ValueError:
                continue
            if cid in changed_ids:
                reason = str(row.get("reason", "") or "")
                row["reason"] = "|".join(filter(None, [reason, "image_supported_edge_recovery"]))

    return updated_traits, organs, qc_rows, cleanup


refine13.process_sheet = process_sheet


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
