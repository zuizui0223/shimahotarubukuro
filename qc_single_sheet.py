#!/usr/bin/env python3
"""Run the reviewed detector on exactly one scan sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review
import measure_guides_review_organs as reviewed_organs
import measure_guides_review_spots as reviewed_spots
import measure_guides_review_traits as visitor_traits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--folder", required=True, help="island folder key, e.g. shikinejima")
    parser.add_argument("--out-dir", type=Path, default=Path("results_single"))
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folder_key = args.folder.lower()
    island, _ = base.ISLANDS.get(folder_key, (folder_key, ""))

    review.install_reviewed_overrides()
    v2.organs = reviewed_organs.organs_reviewed
    rows, organs = v2.process_sheet(
        str(args.image),
        folder_key,
        str(args.out_dir),
        loc_map={},
        auto_split=True,
    )
    if not rows:
        raise SystemExit("No corollas detected")

    spot_summaries, spot_rows = reviewed_spots.write_spot_qc(
        args.image,
        folder_key,
        args.out_dir,
        auto_split=True,
    )
    trait_summaries, scale_info = visitor_traits.write_flower_trait_qc(
        args.image,
        folder_key,
        args.out_dir,
        spot_segmenter=reviewed_spots.reviewed_spot_segment,
        auto_split=True,
    )
    if len(spot_summaries) != len(rows) or len(trait_summaries) != len(rows):
        raise SystemExit(
            "Output/corolla mismatch: "
            f"spots={len(spot_summaries)} visitor_traits={len(trait_summaries)} "
            f"corollas={len(rows)}"
        )

    spot_fields_to_merge = (
        "n_spots",
        "n_strong_spots",
        "n_weak_recovered_spots",
        "n_large_spot_clusters",
        "guide_area_mm2",
        "guide_cov_pct",
        "guide_cov_strong_pct",
        "guide_cov_weak_recovered_pct",
        "spot_density_cm2",
        "mean_spot_area_mm2",
        "median_spot_area_mm2",
        "max_spot_area_mm2",
        "brown_overlap_pct_of_spots",
        "guide_present",
        "n_oxidized_recovered_spots",
        "guide_area_incl_oxidized_mm2",
        "guide_cov_incl_oxidized_pct",
        "spot_qc_source",
    )
    trait_fields_to_merge = tuple(
        key for key in next(iter(trait_summaries.values())).keys()
        if key not in {"island", "sheet", "corolla_id"}
    )

    for row in rows:
        corolla_id = int(row["corolla_id"])
        spot_summary = spot_summaries[corolla_id]
        trait_summary = trait_summaries[corolla_id]
        for field in spot_fields_to_merge:
            row[field] = spot_summary[field]
        for field in trait_fields_to_merge:
            row[field] = trait_summary[field]

    v2.write_csv(args.out_dir / "traits.csv", rows)
    v2.write_csv(
        args.out_dir / "spot_summary.csv",
        [spot_summaries[index] for index in sorted(spot_summaries)],
    )
    v2.write_csv(
        args.out_dir / "visitor_traits.csv",
        [trait_summaries[index] for index in sorted(trait_summaries)],
    )
    v2.write_csv(args.out_dir / "scale_calibration.csv", [scale_info])

    spot_fields = [
        "island", "sheet", "corolla_id", "spot_id", "cx", "cy",
        "area_mm2", "equivalent_diameter_mm", "detection_class",
        "strong_pixel_fraction", "mean_a_minus_b", "median_a_minus_b",
        "brown_overlap_pct", "large_cluster_check", "exclude_FILL",
    ]
    v2.write_csv(args.out_dir / "spots.csv", spot_rows, spot_fields)

    organ_fields = [
        "island", "sheet", "organ_id", "nearest_corolla",
        "cx", "cy", "x1", "y1", "x2", "y2",
        "length_mm", "width_mm", "aspect", "angle_deg",
        "organ_type_auto", "organ_type_FILL", "exclude_FILL",
        "detection_source", "nearest_corolla_hint",
        "association_confirmed", "visibility_note",
    ]
    v2.write_csv(args.out_dir / "organs.csv", organs, organ_fields)

    expected = [
        args.out_dir / "spot_overlays" / f"{island}_{args.image.stem}_spots.png",
        args.out_dir / "spot_masks" / f"{island}_{args.image.stem}_spot_mask.png",
        args.out_dir / "trait_overlays" / f"{island}_{args.image.stem}_visitor_traits.png",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit("Reviewed QC images were not created: " + ", ".join(missing))

    weak_recovered = sum(int(summary["n_weak_recovered_spots"]) for summary in spot_summaries.values())
    print(
        f"corollas={len(rows)} organs={len(organs)} spots={len(spot_rows)} "
        f"weak_recovered={weak_recovered} scale={scale_info['mm_per_px']}mm/px "
        f"({scale_info['scale_source']}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
