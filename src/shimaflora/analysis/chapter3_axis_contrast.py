"""Summarize the prespecified Chapter 3 phenotype-axis contrast.

This analysis compares the two axis groups named in ``docs/THESIS_EMPIRICAL_BRIDGE.md``
before the 2026-08-27 mechanistic-bridge revision. Trait Pst values are correlated
summaries from the same plants, so the output is descriptive and does not treat traits
as independent biological replicates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median

ABSOLUTE_INVESTMENT = (
    "corolla_length_mm",
    "corolla_width_fulleq_mm",
    "corolla_area_fulleq_mm2",
    "throat_width_mm",
    "mouth_width_mm",
    "organ_length_mm",
    "guide_coverage_pct",
)

PROPORTIONAL_SPATIAL = (
    "corolla_aspect_L_W",
    "tube_flare_W_throat",
    "organ_corolla_ratio",
    "guide_basal_frac",
    "guide_midline_ratio",
)


def _cliffs_delta(left: list[float], right: list[float]) -> float:
    greater = sum(a > b for a in left for b in right)
    less = sum(a < b for a in left for b in right)
    return (greater - less) / (len(left) * len(right))


def summarize(stats_csv: str | Path) -> dict:
    rows = list(csv.DictReader(Path(stats_csv).open(encoding="utf-8-sig")))
    by_key = {row["key"]: row for row in rows}
    required = set(ABSOLUTE_INVESTMENT) | set(PROPORTIONAL_SPATIAL)
    missing = sorted(required - set(by_key))
    if missing:
        raise ValueError(f"missing prespecified trait keys: {missing}")

    def group(keys: tuple[str, ...]) -> dict:
        values = [float(by_key[key]["pst"]) for key in keys]
        significant = [
            key for key in keys if float(by_key[key]["site_p_adj"]) < 0.05
        ]
        return {
            "n_traits": len(keys),
            "trait_keys": list(keys),
            "pst_values": values,
            "mean_pst": mean(values),
            "median_pst": median(values),
            "min_pst": min(values),
            "max_pst": max(values),
            "site_corrected_significant_count": len(significant),
            "site_corrected_significant_traits": significant,
        }

    absolute = group(ABSOLUTE_INVESTMENT)
    proportional = group(PROPORTIONAL_SPATIAL)
    return {
        "schema_version": "1.0",
        "analysis": "chapter3_prespecified_axis_contrast",
        "input": str(stats_csv),
        "absolute_investment": absolute,
        "proportional_spatial": proportional,
        "contrast": {
            "mean_pst_difference": absolute["mean_pst"] - proportional["mean_pst"],
            "median_pst_difference": absolute["median_pst"] - proportional["median_pst"],
            "mean_pst_ratio": absolute["mean_pst"] / proportional["mean_pst"],
            "complete_separation": absolute["min_pst"] > proportional["max_pst"],
            "cliffs_delta": _cliffs_delta(
                absolute["pst_values"], proportional["pst_values"]
            ),
        },
        "excluded_trait": {
            "key": "lobe_incision_mm",
            "reason": "not assigned to either prespecified group in the prior bridge",
        },
        "interpretation_boundary": (
            "Trait axes are correlated measurements from the same plants. This contrast "
            "describes phenotype architecture and is not an independent-replicate test of "
            "a pollinator mechanism or natural selection."
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats",
        default="results_shimask_all/island_analysis_stats.csv",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    result = summarize(args.stats)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
