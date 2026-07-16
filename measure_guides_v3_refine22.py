# -*- coding: utf-8 -*-
"""Apply guarded local growth only to genuinely uncertain refine19 corollas.

refine21 showed that broad application over-expanded almost every corolla. This
version keeps refine19 unchanged for high-confidence masks and only considers
rows already routed to QC or below high confidence. shimask remains evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine19 as refine19  # installs accepted GrabCut process

# Capture accepted refine19 before importing refine21, whose module import rewires process_sheet.
_ACCEPTED_PROCESS_SHEET = refine13.process_sheet
import measure_guides_v3_refine21 as refine21  # noqa: E402

# Restore our own wrapper target after refine21 import side effects.
refine13.process_sheet = _ACCEPTED_PROCESS_SHEET

_CONFIDENCE_THRESHOLD = 0.90
_MIN_STRICT_EVIDENCE = 4.0


def should_attempt_growth(row: dict) -> bool:
    """Only uncertain masks are eligible for experimental local expansion."""
    try:
        confidence = float(row.get("mask_confidence", 1.0) or 1.0)
    except (TypeError, ValueError):
        confidence = 1.0
    qc_required = str(row.get("mask_qc_required", "0")).strip().lower() in {
        "1", "true", "yes",
    }
    return qc_required or confidence < _CONFIDENCE_THRESHOLD


def process_sheet(path: str, folder: str, out_dir: str):
    traits, organs, qc_rows, cleanup = _ACCEPTED_PROCESS_SHEET(path, folder, out_dir)
    image = base.load_bgr(path)
    channels = refine._lab_channels(image)
    island = base.ISLANDS.get(folder, (folder, ""))[0]
    stem = Path(path).stem
    mask_dir = Path(out_dir) / "masks" / island / stem

    updated: list[dict] = []
    for row in traits:
        row["local_growth_attempted"] = 0
        row["local_growth_status"] = "not_eligible_high_confidence"
        row["local_growth_evidence"] = ""
        if not should_attempt_growth(row):
            updated.append(row)
            continue

        row["local_growth_attempted"] = 1
        cid = int(row["corolla_id"])
        mask_path = mask_dir / f"C{cid}.png"
        mask = (cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        grown, evidence, status = refine21.guarded_expand(image, mask)
        row["local_growth_evidence"] = round(float(evidence), 4)

        # refine21's evidence threshold was too permissive in real scans. Require
        # a larger tissue-vs-paper margin before modifying an uncertain mask.
        if status == "accepted" and evidence < _MIN_STRICT_EVIDENCE:
            status = "rejected_strict_evidence"
            grown = mask
        row["local_growth_status"] = status

        if status == "accepted" and int(grown.sum()) != int(mask.sum()):
            cv2.imencode(".png", grown * 255)[1].tofile(str(mask_path))
            row = refine._recompute_traits(image, grown, row, channels)
            prior = str(row.get("mask_qc_reasons", "") or "")
            row["mask_qc_reasons"] = "|".join(filter(None, [prior, "guarded_low_confidence_growth"]))
        updated.append(row)
    return updated, organs, qc_rows, cleanup


refine13.process_sheet = process_sheet


def main() -> None:
    refine13.main()


if __name__ == "__main__":
    main()
