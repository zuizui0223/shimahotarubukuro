#!/usr/bin/env python3
"""Run the existing reviewed QC pipeline with human-confirmed shimask inputs.

Only the input seams are swapped:
- raw-vs-shimask red difference -> confirmed corolla masks
- raw-vs-shimask green difference -> confirmed reproductive-organ rows

The established nectar-guide and floral-trait modules remain unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review_organs as reviewed_organs
import qc_single_sheet
import shimask_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="raw scan")
    parser.add_argument("--shimask", type=Path, required=True, help="human-annotated shimask preview")
    parser.add_argument("--folder", required=True, help="island folder key, e.g. oshima")
    parser.add_argument("--out-dir", type=Path, default=Path("results_shimask_input"))
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.shimask.exists():
        raise SystemExit(f"shimask not found: {args.shimask}")

    raw = base.load_bgr(str(args.image))
    annotated = base.load_bgr(str(args.shimask))

    red_components = shimask_input.red_corolla_components(raw, annotated)
    green_organs = shimask_input.green_organ_rows(raw, annotated)
    if not red_components:
        raise SystemExit("No red corolla outlines found in raw-vs-shimask difference")

    v2.corollas = lambda filled, auto_split=True, _c=red_components: [dict(c) for c in _c]
    reviewed_organs.organs_reviewed = (
        lambda img, corolla_mask, top, _r=green_organs: [dict(r) for r in _r]
    )

    island, _ = base.ISLANDS.get(args.folder.lower(), (args.folder, ""))
    shimask_input.write_annotation_overlay(
        raw,
        annotated,
        args.out_dir / "annotation_overlays" / f"{island}_{args.image.stem}_annotation.png",
    )
    shimask_input.write_stroke_colour_stats(
        raw,
        annotated,
        args.out_dir / "annotation_colour_stats.csv",
    )

    sys.argv = [
        "qc_single_sheet.py",
        "--image", str(args.image),
        "--folder", args.folder,
        "--out-dir", str(args.out_dir),
    ]
    qc_single_sheet.main()
    print(
        f"shimask-input QC done: corollas={len(red_components)} "
        f"organs={len(green_organs)} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
