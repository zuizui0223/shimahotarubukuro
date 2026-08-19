#!/usr/bin/env python3
"""Contrast receiver-dependent visual signalling with morphomechanical traits.

This synthesis keeps three evidential levels separate for the purple spotted pattern:
1) non-random within-flower placement supports calling it a nectar-guide candidate;
2) guide amount varies among islands beyond simple flower-size allometry; and
3) direct use by natural pollinators remains untested.

The morphomechanical side asks whether dimensions controlling physical access or
contact change through island-specific allometric relationships.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results_shimask_all")


def row_for_trait(tests, trait, system, prediction):
    r = tests.loc[tests.trait == trait].iloc[0]
    if r.site_interaction_permutation_p_bh < 0.05:
        response = "island_specific_allometric_reconfiguration"
    elif r.site_ancova_permutation_p_bh < 0.05:
        response = "island_level_shift_beyond_common_allometry"
    else:
        response = "simple_scaling_or_unresolved"
    return {
        "system": system,
        "trait": trait,
        "prediction": prediction,
        "observed_response": response,
        "site_ancova_partial_r2": r.site_ancova_partial_r2,
        "site_ancova_permutation_p": r.site_ancova_permutation_p,
        "site_ancova_permutation_p_bh": r.site_ancova_permutation_p_bh,
        "site_interaction_partial_r2": r.site_interaction_partial_r2,
        "site_interaction_permutation_p": r.site_interaction_permutation_p,
        "site_interaction_permutation_p_bh": r.site_interaction_permutation_p_bh,
        "common_support_permutation_p": r.common_support_site_residual_permutation_p,
        "common_support_permutation_p_bh": r.common_support_site_residual_permutation_p_bh,
    }


def main():
    tests = pd.read_csv(R / "multivariate_allometry_tests.csv")
    corr = pd.read_csv(R / "multivariate_size_guide_correlations.csv")
    guide = pd.read_csv(R / "syndrome_attraction_disassembly.csv")
    layout = pd.read_csv(R / "guide_functional_layout_summary.csv").iloc[0]

    pd.DataFrame([
        row_for_trait(tests, "guide_coverage_pct", "visual_signal_amount",
                      "guide amount may shift among populations beyond simple floral miniaturisation"),
        row_for_trait(tests, "mouth_width_mm", "morphomechanical_interface",
                      "physical access/contact may be rematched through island-specific allometry"),
        row_for_trait(tests, "throat_width_mm", "morphomechanical_interface",
                      "internal geometry may either rematch or retain common size scaling"),
        row_for_trait(tests, "organ_length_mm", "pollen_transfer_geometry",
                      "contact geometry may show island shifts beyond overall body size"),
    ]).to_csv(R / "syndrome_signal_mechanical_trait_evidence.csv", index=False)

    g = guide.set_index("guide_component")
    amount = float(g.loc["signal_amount", "raw_pst"])
    basal = float(g.loc["spatial_placement_basal", "raw_pst"])
    mid = float(g.loc["spatial_placement_midline", "raw_pst"])

    within = corr[corr.comparison == "within-island centred plants"].iloc[0]
    between = corr[corr.comparison == "between-island means"].iloc[0]
    guide_test = tests[tests.trait == "guide_coverage_pct"].iloc[0]
    mouth_test = tests[tests.trait == "mouth_width_mm"].iloc[0]
    throat_test = tests[tests.trait == "throat_width_mm"].iloc[0]
    organ_test = tests[tests.trait == "organ_length_mm"].iloc[0]

    summary = pd.DataFrame([
        {
            "hypothesis": "nectar_guide_candidate_spatial_layout",
            "current_support": "direct_morphological_support",
            "key_result": "guide pixels are non-randomly concentrated toward the proximal/base side and petal/corolla midline relative to random pixels within the same corolla",
            "metric_1": "mean_guide_fraction_proximal_third",
            "value_1": float(layout.mean_guide_fraction_in_proximal_third),
            "metric_2": "mean_random_fraction_proximal_third",
            "value_2": float(layout.mean_random_fraction_in_proximal_third),
            "metric_3": "mean_midline_ratio_distal_random_over_guide",
            "value_3": float(layout.mean_midline_ratio_distal_random_over_guide_distance),
            "metric_4": "fraction_corollas_distal_midline_ratio_gt_1",
            "value_4": float(layout.fraction_corollas_midline_ratio_distal_gt_1),
            "guardrail": "supports the morphological nectar-guide label but does not demonstrate that bumblebees or other visitors perceive or use the natural pattern",
        },
        {
            "hypothesis": "receiver_dependent_visual_signal_amount",
            "current_support": "phenotypic_support_only",
            "key_result": "guide amount diverges beyond body size; between-island divergence in amount is larger than in basal/midline placement metrics",
            "metric_1": "guide_site_ancova_BH_p",
            "value_1": float(guide_test.site_ancova_permutation_p_bh),
            "metric_2": "guide_amount_PST",
            "value_2": amount,
            "metric_3": "basal_placement_PST",
            "value_3": basal,
            "metric_4": "midline_placement_PST",
            "value_4": mid,
            "guardrail": "does not directly test receiver perception/cognition, selective cause, or guide gene regulation",
        },
        {
            "hypothesis": "signal_not_generic_individual_allometry",
            "current_support": "supported_at_current_sampling_scale",
            "key_result": "body size and guide covary among islands but not within islands",
            "metric_1": "within_island_plant_r",
            "value_1": float(within.pearson_r),
            "metric_2": "within_island_plant_p",
            "value_2": float(within.pearson_p),
            "metric_3": "between_island_mean_r",
            "value_3": float(between.pearson_r),
            "metric_4": "between_island_mean_spearman_rho",
            "value_4": float(between.spearman_rho),
            "guardrail": "five island means and uneven site replication; not a causal ecological test",
        },
        {
            "hypothesis": "morphomechanical_rematching",
            "current_support": "trait_specific_support",
            "key_result": "mouth width changes through island-specific allometric slopes whereas throat width is closer to common scaling",
            "metric_1": "mouth_interaction_BH_p",
            "value_1": float(mouth_test.site_interaction_permutation_p_bh),
            "metric_2": "mouth_interaction_partial_R2",
            "value_2": float(mouth_test.site_interaction_partial_r2),
            "metric_3": "throat_island_BH_p",
            "value_3": float(throat_test.site_ancova_permutation_p_bh),
            "metric_4": "throat_interaction_BH_p",
            "value_4": float(throat_test.site_interaction_permutation_p_bh),
            "guardrail": "mechanical function is inferred from morphology; visitor size/contact and pollen transfer are not yet measured",
        },
        {
            "hypothesis": "pollen_transfer_geometry_reorganisation",
            "current_support": "phenotypic_support_only",
            "key_result": "reproductive-organ length retains island divergence after body-size adjustment",
            "metric_1": "organ_site_ancova_BH_p",
            "value_1": float(organ_test.site_ancova_permutation_p_bh),
            "metric_2": "organ_site_ancova_partial_R2",
            "value_2": float(organ_test.site_ancova_partial_r2),
            "metric_3": "organ_interaction_BH_p",
            "value_3": float(organ_test.site_interaction_permutation_p_bh),
            "metric_4": "organ_common_support_p",
            "value_4": float(organ_test.common_support_site_residual_permutation_p),
            "guardrail": "reproductive geometry is not direct reproductive assurance",
        },
        {
            "hypothesis": "genetic_regulation_of_visual_signal",
            "current_support": "not_tested",
            "key_result": "phenotype pattern motivates a regulatory-expression hypothesis but the current neutral SNP/phenotype pipeline cannot identify causal guide loci",
            "metric_1": "not_applicable", "value_1": np.nan,
            "metric_2": "not_applicable", "value_2": np.nan,
            "metric_3": "not_applicable", "value_3": np.nan,
            "metric_4": "not_applicable", "value_4": np.nan,
            "guardrail": "requires matched dense genomics and/or common-garden developmental/expression work",
        },
    ])
    summary.to_csv(R / "syndrome_signal_mechanical_hypotheses.csv", index=False)

    print("\n=== physical matching vs receiver-dependent visual signalling ===")
    print(f"guide spatial layout: proximal fraction guide/random={layout.mean_guide_fraction_in_proximal_third:.3f}/{layout.mean_random_fraction_in_proximal_third:.3f}; distal midline ratio={layout.mean_midline_ratio_distal_random_over_guide_distance:.3f}")
    print(f"guide amount: island shift beyond size BH={guide_test.site_ancova_permutation_p_bh:.4g}; within-island size-guide r={within.pearson_r:+.3f}; PST amount/basal/midline={amount:.3f}/{basal:.3f}/{mid:.3f}")
    print(f"mouth: slope-interaction BH={mouth_test.site_interaction_permutation_p_bh:.4g}; throat island BH={throat_test.site_ancova_permutation_p_bh:.4g}; organ island BH={organ_test.site_ancova_permutation_p_bh:.4g}")
    print("Guardrail: non-random guide placement supports the nectar-guide candidate label; pollinator use, selective cause, and genetic regulation remain hypotheses.")


if __name__ == "__main__":
    main()
