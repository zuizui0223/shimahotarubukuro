#!/usr/bin/env python3
"""Run the reviewed detector on exactly one scan sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import measure_guides_v2 as v2
import measure_guides_review as review
import measure_guides_review_organs as reviewed_organs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--folder", required=True, help="island folder key, e.g. shikinejima")
    parser.add_argument("--out-dir", type=Path, default=Path("results_single"))
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    review.install_reviewed_overrides()
    v2.organs = reviewed_organs.organs_reviewed
    rows, organs = v2.process_sheet(
        str(args.image),
        args.folder.lower(),
        str(args.out_dir),
        loc_map={},
        auto_split=True,
    )
    if not rows:
        raise SystemExit("No corollas detected")

    v2.write_csv(args.out_dir / "traits.csv", rows)
    organ_fields = [
        "island", "sheet", "organ_id", "nearest_corolla",
        "cx", "cy", "x1", "y1", "x2", "y2",
        "length_mm", "width_mm", "aspect", "angle_deg",
        "organ_type_auto", "organ_type_FILL", "exclude_FILL",
        "detection_source", "nearest_corolla_hint",
        "association_confirmed", "visibility_note",
    ]
    v2.write_csv(args.out_dir / "organs.csv", organs, organ_fields)
    print(f"corollas={len(rows)} organs={len(organs)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
