# Chapter 3 prespecified phenotype-axis contrast

Updated: 2026-08-27

## Question

The pre-existing `docs/THESIS_EMPIRICAL_BRIDGE.md` separated two phenotype groups before this contrast was run:

1. **absolute-size / investment axes** — corolla dimensions, mouth/throat width, reproductive-organ length and guide coverage;
2. **proportional-shape / spatial-allocation axes** — aspect, tube flare, organ/corolla ratio and guide spatial allocation.

The question is whether among-island phenotypic differentiation is distributed uniformly across those groups or concentrated in the first group.

## Fixed inputs

The analysis reads the already-final plant-level global `Pst` and site-corrected island tests from `results_shimask_all/island_analysis_stats.csv`. No trait was moved between groups after inspecting the contrast. `lobe_incision_mm` is excluded because the prior bridge did not assign it to either group.

## Result

| Summary | Absolute / investment | Proportional / spatial |
|---|---:|---:|
| n traits | 7 | 5 |
| mean Pst | 0.331 | 0.086 |
| median Pst | 0.332 | 0.082 |
| minimum Pst | 0.238 | 0.042 |
| maximum Pst | 0.475 | 0.129 |
| site-corrected significant traits | 7/7 | 0/5 |

The group mean difference is `0.2449 Pst` and the mean ratio is `3.85×`. More importantly, the two groups are **completely separated** at the retained trait level: the smallest absolute/investment value (`0.238`) exceeds the largest proportional/spatial value (`0.129`). Trait-level Cliff's delta is therefore `1.0`.

## Interpretation

The focal Izu lineage does not show equally strong differentiation across all measured floral architecture. Differentiation is concentrated in:

- absolute floral display size;
- mouth/throat access dimensions;
- reproductive-interface length;
- nectar-guide investment.

By contrast, several proportional-shape and within-guide spatial-allocation metrics remain much less differentiated.

This sharpens the Chapter 3 result from `multidimensional divergence exists` to:

> **Multidimensional divergence is strongly structured by phenotype channel: absolute/investment and interface axes carry substantially more among-island differentiation than the prespecified proportional/spatial axes.**

That pattern is compatible with the Chapter 2 idea that downstream realization can be channel-specific rather than one coordinated island syndrome. It does not identify which Chapter 2 mechanism caused any Chapter 3 axis.

## Inferential boundary

The trait axes are correlated measurements obtained from the same plants. They are not independent biological replicates. Therefore the analysis reports architecture-level descriptive contrasts and complete separation, but does not promote a trait-label permutation p-value as a biological significance test.

The result does not establish historical Bombus causation, effective pollination service, Qst, or selection. The direct mechanistic bridge still requires visitor identity/functional traits, plant-specific interaction weights, pollen deposition/effective service and reproductive dependency/outcome.
