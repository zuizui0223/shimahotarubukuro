# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal phenotypic-realization** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, why do those changes produce different response branches across islands and lineages, and what phenotype structure is realized within a focal island lineage?**

The three levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** asks **when and where** isolation-associated floral/reproductive filtering is detectable and where multivariate response vectors differ.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks **how** altered interaction states propagate and supplies a model-conditional **proximal why** for response branching. Its broad comparison universe is larger than the 13 strict manuscript challenges; Izu is the deepest mechanistic anchor rather than one exchangeable replicate among all systems.
- `shimahotarubukuro` — **Chapter 3:** measures the focal Izu lineage's phenotype and asks how much of the among-island divergence is coordinated with floral size versus retained as selected departures from common size scaling.

The Chapter 3 question is therefore:

> **How is among-island floral divergence structured within *Campanula microdonta*: as one coordinated size/investment trajectory, as independent trait-channel divergence, or as a mixture of shared scaling and selected departures?**

This is stronger than a generic `what phenotype differs?` question, but it remains a phenotype question: Chapter 3 does not infer the historical pollinator cause from morphology alone.

## Why Izu is the focal depth axis

The dissertation does not move from a 13-system comparison directly to an arbitrary case study. Chapter 2 has examined a broader comparative universe that includes the 13 strict external challenges plus additional cross-archipelago, source-gated and falsification systems. Those systems establish response-state breadth, counterexamples and identifiability boundaries.

Izu is then used asymmetrically as the **high-resolution historical plus contemporary mechanism anchor** because the same island series can connect:

1. historical *Campanula* phenotype and mating-system evidence;
2. contemporary plant–pollinator network structure;
3. source-native pollinator functional traits;
4. signed source-position / pollinator-centre-shift analyses;
5. prospective visitor-effectiveness and reproductive-dependency measurements;
6. the directly measured within-lineage phenotype in this repository.

The dissertation zoom is therefore:

```text
broad island comparison universe
        ↓
strict response-state / falsification boundaries
        ↓
Izu deep mechanistic anchor
        ↓
C. microdonta phenotypic realization
```

Chapter 3 is the final depth layer of that zoom.

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

The current Chapter 2 result is not a universal island syndrome. It shows that response geometry is conditional on partner loss/arrival balance, starting functional position, realized pollinator community and local interaction filtering. Starting position organizes mean response geometry, while realized community state dominates cell-level variation and combines non-additively with starting state.

Chapter 3 does **not** claim to have observed those model response branches directly in phenotype modules. Instead it supplies the focal phenotype that any proposed mechanism must explain. The phenotype contains both a large coordinated component and residual structure beyond simple size scaling.

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

The missing end-to-end empirical bridge remains:

```text
visitor identity + exact pollinator functional trait
    -> plant-specific visitor weights
        -> frozen signed plant position
            -> single-visit pollen deposition / effective service
                -> controlled reproductive dependency
                    -> mature reproductive outcome
```

Chapter 3 can later be tested as a phenotypic endpoint of that chain, but its current morphology must not be back-labelled as causal proof.

## Three-chapter architecture

| Chapter | Repository | Scale | Primary question | Main contribution |
| --- | --- | --- | --- | --- |
| 1 | `island` | global island floras | **When / where?** | contexts where filtering is detectable and response vectors differ |
| 2 | `izu-core` | broad comparison + mechanistic response architecture + Izu deep bridge | **How / proximal why?** | conditional response geometry, community contingency, falsification boundaries and focal empirical triangulation |
| 3 | `shimahotarubukuro` | one lineage across five Izu islands | **How is phenotype structured?** | coordinated size/investment divergence plus selected residual departures beyond common allometry |

Together:

```text
WHEN / WHERE island filtering is detectable                  [Chapter 1]
    ↓
HOW / WHY ecological responses can branch conditionally      [Chapter 2]
    ↓
WHAT coordinated and residual phenotype structure is realised [Chapter 3]
```

The Chapter 3 contribution is **to anchor the dissertation's broad comparative and mechanistic arguments in a directly measured phenotype within the same Izu island series, while explicitly testing how much of that phenotype is shared scaling and how much remains beyond it.**
