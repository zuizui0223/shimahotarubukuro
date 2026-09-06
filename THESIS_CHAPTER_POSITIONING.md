# Thesis positioning — Chapter 3

Updated: 2026-09-06  
Chapter 2 boundary synchronized to `zuizui0223/izu-core` after world-saturation / Izu-continuity closure.

## Role in the dissertation

This repository is the **Chapter 3 / focal phenotypic-realization** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, why do those changes produce different response branches across islands and lineages, and what phenotype structure is realized within a focal island lineage?**

The three levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** asks **when and where** isolation-associated floral/reproductive filtering is detectable and where multivariate response vectors differ.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks **how / proximal why** altered interaction states can produce different response branches. It defines a conditional response geometry, confronts it with world-island evidence until the geography-first mechanism vocabulary saturates, identifies a structured historical-transition measurement bottleneck, and then uses Izu as the final continuity-system zoom because the same source-linked series can resolve the contemporary half of that chain while retaining contrary evidence.
- `shimahotarubukuro` — **Chapter 3:** owns the **directly measured focal phenotype**. It asks how much of *Campanula microdonta* among-island divergence follows a coordinated size/investment trajectory and what departures remain beyond common allometric scaling.

The Chapter 3 question is therefore:

> **How is among-island floral divergence structured within *Campanula microdonta*: as one coordinated size/investment trajectory, as independent trait-channel divergence, or as a mixture of shared scaling and selected departures?**

This is stronger than a generic `what phenotype differs?` question, but it remains a phenotype question: Chapter 3 does not infer the historical pollinator cause from morphology alone.

## Why Izu is the focal depth axis

The dissertation does **not** move from a selected handful of literature examples to an arbitrary nearby case study. Chapter 2 first separates a frozen formal identifiability audit from a broader descriptive programme, expands the world confrontation to **42 research entries across 37 exact geographic labels**, and then performs a geography-first island-system search with an outcome-independent stopping rule. The large-island search reaches the declared two-tranche zero-novelty condition; a separate small-island supplement partially recovers transition history but still does not close the full source-state → transition → realized-community → plant-response contract.

Only after that breadth step is saturated does the programme move to focal depth. Izu is selected by **measurement continuity across the bottleneck**, not because it is geographically close, logistically convenient, representative of all islands, or the strongest positive match to the Chapter 2 simulation.

The same Izu series can connect:

1. historical *Campanula microdonta* floral, mating-system and autonomous-capacity responses;
2. explicit population-history / colonisation alternatives that prevent a simple pollinator-history story;
3. repeated contemporary plant–pollinator network structure;
4. source-native quantitative pollinator functional traits;
5. signed source-position analyses, including model-facing failures rather than only successes;
6. contemporary functional-diversity → trait-matching and matching → pollen-response layers;
7. the directly measured within-lineage phenotype in this repository;
8. a feasible prospective extension to visitor effectiveness, reproductive dependency and mature reproductive output in tagged populations.

This is unusually valuable because the focal series can **discriminate among explanations**. In Chapter 2, the historical signed-position predictor does not explain null-corrected matching and the Oshima-source bridge is unsupported, whereas contemporary functional diversity is positively associated with corrected trait matching. Izu is therefore retained despite inconvenient results; it is not a hand-picked positive control.

The dissertation zoom is now:

```text
world island confrontation
        ↓
geography-first saturation + historical-transition bottleneck
        ↓
Izu selected by measurement continuity and falsification value
        ↓
contemporary functional realization resolved in Chapter 2
        ↓
C. microdonta directly measured phenotypic realization in Chapter 3
```

Chapter 3 is the **direct phenotype layer** of that zoom. It does not need to retroactively validate Chapter 2 for Chapter 2 to be complete.

## Empirical system

The publication pipeline measures flattened, pressed *Campanula microdonta* corollas from five Izu Islands:

- Oshima
- Toshima
- Niijima
- Shikinejima
- Kozushima

The retained dataset contains **218 corollas from 125 individual plants**. Inference is performed on plant-level means, with island effects evaluated after accounting for site structure. Global and pairwise `Pst` are used as phenotypic-divergence summaries.

The retained phenotype domains include:

- corolla length, width and area;
- throat and mouth width;
- aspect and tube-flare ratios;
- lobe incision;
- reproductive-organ length and organ/corolla ratio;
- nectar-guide coverage;
- guide spatial concentration.

## Raw among-island divergence

The strongest retained `Pst` values include:

| trait | Pst | Oshima mean | Kozushima mean |
| --- | ---: | ---: | ---: |
| reproductive-organ length | 0.475 | 22.68 mm | 14.07 mm |
| mouth width | 0.374 | 50.99 mm | 28.67 mm |
| corolla area | 0.370 | 1553.78 mm² | 611.94 mm² |
| corolla length | 0.332 | 35.92 mm | 22.88 mm |
| corolla width | 0.283 | 53.08 mm | 34.70 mm |
| guide coverage | 0.244 | 31.08% | 6.90% |
| throat width | 0.238 | 31.66 mm | 22.69 mm |

The prespecified contrast between seven absolute/investment/interface axes and five proportional/spatial axes is complete at the retained trait level: mean `Pst` is `0.331` versus `0.086`, the smallest absolute/investment `Pst` (`0.238`) exceeds the largest proportional/spatial value (`0.129`), and the site-corrected significant counts are `7/7` versus `0/5`.

That contrast is descriptive architecture, not independent-replicate evidence for separate mechanisms. Absolute traits share a size component and proportional traits remove part of that scale by construction.

## Size/allometry gate

The pre-existing multivariate/allometry pipeline provides the stricter interpretation.

- **throat width:** after floral body-size adjustment, no independent island effect or island-specific allometry is detected; its raw divergence is compatible with common floral scaling.
- **mouth width:** retains a size-adjusted island effect (site-permutation BH `p = 0.0368`) and a strong island × size interaction (BH `p = 0.020`), consistent with island-specific allometric reconfiguration over the full observed range. The multiple-testing-corrected common-support sensitivity is weaker.
- **reproductive-organ length:** retains an additive size-adjusted island shift (BH `p = 0.0216`), while the restricted common-support sensitivity is weaker.
- **nectar-guide coverage:** likewise retains a full-range size-adjusted island shift (BH `p = 0.0216`), with weaker restricted common-support sensitivity.

The absolute multivariate phenotype has a leading `Pst` analogue of `0.592` (bootstrap 95% `0.526–0.812`). After residualising mouth width, throat width, reproductive-organ length and guide coverage against common floral body size, a residual multivariate axis remains with `Pst` analogue `0.286` (bootstrap 95% `0.245–0.639`). Its largest loadings are reproductive-organ residual (`+0.745`), mouth-width residual (`−0.555`) and guide residual (`+0.365`).

Therefore simple floral miniaturisation is insufficient, but neither is the phenotype best described as freely independent trait branching.

A separate module analysis finds a rank-1 shared trajectory fraction of `0.775` (bootstrap 95% `0.546–0.886`), and the module × island permutation test is not significant (`p = 0.3907`). The current phenotype is thus best summarized as:

> **a strongly coordinated island size/investment trajectory plus selected departures from common size scaling in mouth allometry, reproductive-interface length, guide investment and a residual multivariate phenotype axis.**

See `docs/CHAPTER3_SIZE_ALLOMETRY_GATE_20260827.md` for the frozen interpretation gate.

## Relationship to Chapter 1

Chapter 1 establishes that isolation-associated floral/reproductive filtering is detectable in northern mid-latitude and tropical island floras, persists within native non-endemic assemblages, and is expressed as different multivariate response vectors between those contexts.

Chapter 3 does not retest that global boundary result. It asks how phenotype differentiation is structured within one focal northern-midlatitude lineage.

Thus Chapter 1 supplies **breadth at the assemblage/biogeographic level**, while Chapter 3 supplies **depth at the within-lineage phenotype level**. The two estimands must not be equated.

## Relationship to Chapter 2

Chapter 2 is now closed at the **continuity-system boundary**, not left waiting for Chapter 3 to validate it. Its current contribution is:

- a conditional post-establishment response geometry in which partner loss/arrival organize possible regimes;
- response direction emerging from starting state evaluated against the realized community, with consequential non-additivity;
- local filtering reallocating branches and autonomous assurance attenuating downstream magnitude in the declared synthetic envelope;
- a world confrontation showing multiple empirical response states and a persistent outcome-rich / transition-poor measurement structure;
- geography-first saturation showing that the bottleneck is not an artefact of the original literature frame;
- an Izu continuity zoom separating historical signed-position inference from robust contemporary functional-diversity → corrected-matching structure and weaker matching → pollen propagation.

Chapter 3 does **not** claim to have observed the synthetic response branches directly in phenotype modules, and it does not serve as a missing validation panel for Chapter 2. Instead, it supplies an **independently measured downstream phenotype** in the same continuity system—the biological object that any stronger historical mechanism would eventually need to explain.

The retained Chapter 3 traits connect to different parts of the Chapter 2 architecture as measurement candidates:

- **corolla length:** closest retained morphology candidate to a plant-side mechanical matching axis, but not calibrated to the exact source-native tube-length coordinate;
- **mouth width:** access geometry with evidence for island-specific allometric reconfiguration;
- **throat width:** access geometry whose current island divergence is largely compatible with common size scaling;
- **reproductive-organ length:** reproductive-interface/contact geometry with a residual island shift after size adjustment;
- **nectar-guide coverage:** visual guidance / attraction investment with a residual full-range island shift after size adjustment;
- **corolla area / width:** display and floral investment, strongly involved in the shared size trajectory;
- **proportional shape and guide-allocation metrics:** comparison axes that currently show weaker site-robust differentiation.

No single Chapter 3 trait is declared to be `the` Chapter 2 functional coordinate, and the current phenotype does not identify the historical mechanism.

## What Chapter 3 establishes

The current evidence supports:

> ***Campanula microdonta* populations across the Izu island series show pronounced floral divergence with a large coordinated size/investment component and selected departures from common size scaling in specific access, reproductive-interface and visual-investment channels.**

This is more informative than either `all traits change uniformly` or `every trait channel branches independently`.

The result is already a substantive empirical endpoint in its own right; it does not depend on completing the historical pollinator-causation bridge.

## What Chapter 3 does not establish

This repository should not claim that:

- the Oshima-versus-other-islands contrast is a causal Bombus experiment;
- `Pst` or the multivariate `Pst` analogue is `Qst`;
- phenotypic differentiation alone demonstrates natural selection;
- lower floral size, access width, reproductive-organ length or guide coverage was historically caused by Bombus loss;
- Chapter 1's northern-midlatitude response vector directly predicts the within-lineage Chapter 3 phenotype;
- module-level independent branching is demonstrated by the current phenotype;
- flattened corolla length is interchangeable with the exact signed tube-length coordinate used in the Izu functional-position analysis;
- a phenotype axis alone identifies visitor effectiveness, pollen deposition or reproductive dependency.

A stronger end-to-end causal bridge remains a **prospective extension**, not a prerequisite for the present Chapter 3 phenotype result or for the Chapter 2 conclusion:

```text
visitor identity + exact pollinator functional trait
    -> plant-specific visitor weights
        -> legitimate floral contact / realized matching
            -> single-visit pollen deposition / effective service
                -> controlled reproductive dependency
                    -> mature reproductive outcome
                        -> relation to the measured phenotype
```

Historical identification would additionally require either a true temporal partner-regime transition with matched pre/post plant response or an independently replicated bridge-state exposure. The present morphology must not be back-labelled as that causal proof.

## Three-chapter architecture

| Chapter | Repository | Scale | Primary question | Main contribution |
| --- | --- | --- | --- | --- |
| 1 | `island` | global island floras | **When / where?** | contexts where filtering is detectable and response vectors differ |
| 2 | `izu-core` | synthetic mechanism + world confrontation + Izu continuity zoom | **How / proximal why?** | conditional response geometry; world saturation and identifiability bottleneck; contemporary mechanism discrimination in Izu |
| 3 | `shimahotarubukuro` | one lineage across five Izu islands | **How is phenotype structured?** | directly measured coordinated size/investment divergence plus selected residual departures beyond common allometry |

Together:

```text
WHEN / WHERE island filtering is detectable                         [Chapter 1]
    ↓
HOW / WHY responses can branch + WHAT remains unidentifiable        [Chapter 2]
    ↓
WHAT phenotype is directly realised within the focal continuity system [Chapter 3]
```

The Chapter 3 contribution is **to supply the dissertation's directly measured within-lineage phenotypic realization in the same Izu continuity system selected independently by Chapter 2's measurement logic, while testing how much of that phenotype is shared scaling and how much remains beyond it.**
