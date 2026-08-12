#!/usr/bin/env python3
"""Summarise differential lability and selective disassembly of floral traits.

This script does not add new causal tests. It integrates already validated
univariate and body-size-adjusted outputs to expose which parts of the floral
phenotype are labile versus conserved. In particular, nectar-guide amount is
kept distinct from the spatial architecture of the guide.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results_shimask_all")

MODULE_MAP = {
    "corolla_length_mm": "body_size",
    "corolla_width_fulleq_mm": "body_size",
    "corolla_area_fulleq_mm2": "body_size",
    "guide_coverage_pct": "attraction_signal_amount",
    "guide_basal_frac": "attraction_signal_architecture",
    "guide_midline_ratio": "attraction_signal_architecture",
    "mouth_width_mm": "pollination_function",
    "throat_width_mm": "pollination_function",
    "tube_flare_W_throat": "pollination_function",
    "organ_length_mm": "reproductive_geometry",
    "organ_corolla_ratio": "reproductive_geometry",
    "corolla_aspect_L_W": "shape",
    "lobe_incision_mm": "shape",
}


def main():
    uni = pd.read_csv(R / "island_analysis_stats.csv", encoding="utf-8-sig")
    uni = uni[uni["key"].isin(MODULE_MAP)].copy()
    uni = uni.rename(columns={"trait": "trait_label"})
    uni["module_component"] = uni["key"].map(MODULE_MAP)
    uni = uni.rename(columns={
        "key": "trait", "pst": "raw_pst", "pst_lo": "raw_pst_lo", "pst_hi": "raw_pst_hi",
        "site_p": "raw_site_p", "site_p_adj": "raw_site_p_bh",
    })

    allom_path = R / "multivariate_allometry_tests.csv"
    if allom_path.exists():
        a = pd.read_csv(allom_path)
        cols = ["trait", "site_ancova_partial_r2", "site_ancova_permutation_p",
                "site_ancova_permutation_p_bh", "site_interaction_partial_r2",
                "site_interaction_permutation_p", "site_interaction_permutation_p_bh",
                "common_support_site_residual_permutation_p",
                "common_support_site_residual_permutation_p_bh"]
        a = a[[c for c in cols if c in a.columns]]
        uni = uni.merge(a, on="trait", how="left")

    modes_path = R / "syndrome_module_response_modes.csv"
    if modes_path.exists():
        modes = pd.read_csv(modes_path)[["trait", "response_mode"]]
        uni = uni.merge(modes, on="trait", how="left")

    # Descriptive only: this is not an evolutionary-rate estimate.
    uni["raw_pst_rank_desc"] = uni["raw_pst"].rank(ascending=False, method="min")
    uni["scope_note"] = np.where(
        uni.module_component.eq("reproductive_geometry"),
        "reproductive geometry proxy; not reproductive assurance",
        "cross-sectional phenotypic divergence",
    )
    uni.sort_values(["module_component", "raw_pst_rank_desc", "trait"]).to_csv(
        R / "syndrome_trait_lability.csv", index=False
    )

    guide = uni[uni["trait"].isin([
        "guide_coverage_pct", "guide_basal_frac", "guide_midline_ratio"
    ])].copy()
    coverage = float(guide.loc[guide.trait == "guide_coverage_pct", "raw_pst"].iloc[0])
    guide["guide_component"] = guide["trait"].map({
        "guide_coverage_pct": "signal_amount",
        "guide_basal_frac": "spatial_architecture_basal",
        "guide_midline_ratio": "spatial_architecture_midline",
    })
    guide["pst_relative_to_coverage"] = guide["raw_pst"] / coverage
    guide["interpretation"] = guide["trait"].map({
        "guide_coverage_pct": "signal amount is comparatively labile among islands",
        "guide_basal_frac": "basal placement is more conserved than signal amount",
        "guide_midline_ratio": "midline placement is more conserved than signal amount",
    })
    guide[["guide_component", "trait", "raw_pst", "raw_pst_lo", "raw_pst_hi",
           "raw_site_p", "raw_site_p_bh", "pst_relative_to_coverage",
           "interpretation"]].to_csv(R / "syndrome_attraction_disassembly.csv", index=False)

    print("\n=== trait lability / selective-disassembly summary ===")
    for _, r in guide.iterrows():
        print(
            f"{r.guide_component:30s} P_ST={r.raw_pst:.3f} "
            f"[{r.raw_pst_lo:.3f},{r.raw_pst_hi:.3f}] site-BH={r.raw_site_p_bh:.3g}"
        )
    print(
        "Interpretation: guide amount is more labile than guide spatial architecture; "
        "this is selective signal attenuation, not proof of temporal order."
    )


if __name__ == "__main__":
    main()
