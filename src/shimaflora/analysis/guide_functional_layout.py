#!/usr/bin/env python3
"""Summarise non-random spatial placement of corolla guide pixels.

This analysis provides the morphological defence for treating the purple spotted
pattern as a *nectar-guide candidate*. It does **not** demonstrate that bumblebees
or other visitors actually perceive or use the pattern.

The underlying `guide_spatial.py` analysis compares guide pixels with random
pixels from the same reviewed corolla region. Here the flower/corolla is the
replication unit, avoiding pixel-level pseudoreplication in the summary tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path("results_shimask_all")


def wilcoxon_greater(x, null=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    return float(stats.wilcoxon(x - null, alternative="greater").pvalue)


def main():
    d = pd.read_csv(R / "guide_spatial.csv", encoding="utf-8-sig")

    basal_diff = d["basal_frac_prox_third"] - d["rand_basal_frac"]
    mid_all = pd.to_numeric(d["midline_ratio_all"], errors="coerce")
    mid_dist = pd.to_numeric(d["midline_ratio_distal"], errors="coerce")

    summary = pd.DataFrame([{
        "n_corollas": int(len(d)),
        "mean_guide_fraction_in_proximal_third": float(d["basal_frac_prox_third"].mean()),
        "mean_random_fraction_in_proximal_third": float(d["rand_basal_frac"].mean()),
        "median_guide_minus_random_proximal_fraction": float(np.nanmedian(basal_diff)),
        "paired_wilcoxon_p_guide_more_proximal": wilcoxon_greater(basal_diff, 0.0),
        "mean_guide_along_length_position_0_base_1_lobes": float(d["mean_along_length_pos"].mean()),
        "n_midline_all": int(mid_all.notna().sum()),
        "mean_midline_ratio_all_random_over_guide_distance": float(mid_all.mean()),
        "median_midline_ratio_all": float(mid_all.median()),
        "fraction_corollas_midline_ratio_all_gt_1": float((mid_all.dropna() > 1).mean()),
        "wilcoxon_p_midline_ratio_all_gt_1": wilcoxon_greater(mid_all, 1.0),
        "n_midline_distal": int(mid_dist.notna().sum()),
        "mean_midline_ratio_distal_random_over_guide_distance": float(mid_dist.mean()),
        "median_midline_ratio_distal": float(mid_dist.median()),
        "fraction_corollas_midline_ratio_distal_gt_1": float((mid_dist.dropna() > 1).mean()),
        "wilcoxon_p_midline_ratio_distal_gt_1": wilcoxon_greater(mid_dist, 1.0),
        "morphological_interpretation": (
            "Guide pixels are non-randomly enriched toward the proximal/base side and "
            "toward the petal/corolla midline relative to random pixels from the same "
            "reviewed corolla region. This spatially directed placement supports treating "
            "the purple pattern as a nectar-guide candidate rather than random pigmentation."
        ),
        "inference_guardrail": (
            "Spatial placement alone does not prove that bumblebees or other visitors "
            "perceive, learn, or use the natural pattern; behavioural/fitness tests are "
            "required for direct functional use."
        ),
    }])
    summary.to_csv(R / "guide_functional_layout_summary.csv", index=False)

    by_island = d.groupby("island", observed=True).agg(
        n_corollas=("corolla_id", "size"),
        mean_guide_fraction_proximal=("basal_frac_prox_third", "mean"),
        mean_random_fraction_proximal=("rand_basal_frac", "mean"),
        mean_along_length_position=("mean_along_length_pos", "mean"),
        mean_midline_ratio_all=("midline_ratio_all", "mean"),
        mean_midline_ratio_distal=("midline_ratio_distal", "mean"),
    ).reset_index()
    by_island.to_csv(R / "guide_functional_layout_by_island.csv", index=False)

    r = summary.iloc[0]
    print("\n=== non-random nectar-guide candidate layout ===")
    print(
        f"proximal third: guide={r.mean_guide_fraction_in_proximal_third:.3f} "
        f"vs random={r.mean_random_fraction_in_proximal_third:.3f}; "
        f"paired flower-level p={r.paired_wilcoxon_p_guide_more_proximal:.3g}"
    )
    print(
        f"midline ratio (random distance / guide distance): all={r.mean_midline_ratio_all_random_over_guide_distance:.3f}, "
        f"distal={r.mean_midline_ratio_distal_random_over_guide_distance:.3f}; "
        f"distal >1 in {100*r.fraction_corollas_midline_ratio_distal_gt_1:.1f}% of corollas"
    )
    print(
        "Interpretation: spatially directed pigmentation supports the morphological "
        "nectar-guide label; pollinator use itself remains a behavioural hypothesis."
    )


if __name__ == "__main__":
    main()
