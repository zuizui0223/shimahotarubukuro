# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal phenotypic-realization** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, why do those changes produce different response branches across islands and lineages, and how are those branches realized in floral phenotype?**

The three levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** asks **when and where** isolation-associated floral/reproductive filtering is detectable and where multivariate response vectors differ.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks **how** altered interaction states propagate and supplies a model-conditional **proximal why** for response branching. Its broad comparison universe is larger than the 13 strict manuscript challenges; Izu is the deepest mechanistic anchor rather than one exchangeable replicate among all systems.
- `shimahotarubukuro` — **Chapter 3:** asks how that conditional-response problem is **realized as unequal multidimensional floral divergence within one focal Izu lineage**.

The Chapter 3 question is therefore:

> **How is a condition-dependent island response architecture realized as multidimensional floral divergence within *Campanula microdonta*, and which phenotype axes carry the strongest among-island differentiation?**

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

## Current empirical pattern

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

The divergence spans **display size, floral access geometry, reproductive-interface length and nectar-guide investment** rather than one generic size variable.

Several proportional or allocation traits remain weaker after site correction. The phenotype is therefore explicitly multidimensional and unequal rather than collapsed into one island-adaptation score.

## Relationship to Chapter 1

Chapter 1 establishes that isolation-associated floral/reproductive filtering is detectable in northern mid-latitude and tropical island floras, persists within native non-endemic assemblages, and is expressed as different multivariate response vectors between those contexts.

Chapter 3 does not retest that global boundary result. It asks whether one focal northern-midlatitude lineage also carries strong but unequal among-island phenotype differentiation.

Thus Chapter 1 supplies **breadth at the assemblage/biogeographic level**, while Chapter 3 supplies **depth at the within-lineage phenotype level**. The two estimands must not be equated.

## Relationship to Chapter 2

The current Chapter 2 result is not a universal island syndrome. It shows that response geometry is conditional on partner loss/arrival balance, starting functional position, realized pollinator community and local interaction filtering. Starting position organizes mean response geometry, while realized community state dominates cell-level variation and combines non-additively with starting state.

Chapter 3 asks what a realized phenotype looks like when the system is allowed to be multidimensional rather than forced into one response axis.

The retained Chapter 3 traits connect to different parts of the Chapter 2 architecture:

- **corolla length:** closest retained morphology candidate to a plant-side mechanical matching axis, but not calibrated to the exact source-native tube-length coordinate;
- **mouth / throat width:** floral access geometry;
- **reproductive-organ length:** reproductive-interface/contact geometry;
- **nectar-guide coverage:** visual guidance / attraction investment;
- **corolla area / width:** display and floral investment;
- **proportional shape and guide-allocation metrics:** comparison axes that currently show weaker site-robust differentiation.

The correspondence is intentionally many-channel. No single Chapter 3 trait is declared to be `the` Chapter 2 functional coordinate.

See `docs/CHAPTER2_CHAPTER3_MECHANISTIC_BRIDGE_20260827.md` for the explicit trait-axis bridge and claim boundaries.

## What Chapter 3 establishes

The current evidence supports:

> ***Campanula microdonta* populations across the Izu island series show pronounced, multidimensional floral divergence concentrated in particular absolute-size, access, reproductive-interface and nectar-guide axes rather than expressed uniformly across all floral-shape metrics.**

This directly establishes that the focal lineage's island phenotype is not well summarized by one uniformly changing floral syndrome.

It is therefore consistent with the dissertation's conditional-response architecture: different response channels can carry different amounts of divergence.

## What Chapter 3 does not establish

This repository should not claim that:

- the Oshima-versus-other-islands contrast is a causal Bombus experiment;
- `Pst` is `Qst`;
- phenotypic differentiation alone demonstrates natural selection;
- lower floral size, access width, reproductive-organ length or guide coverage was historically caused by Bombus loss;
- Chapter 1's northern-midlatitude response vector directly predicts the within-lineage Chapter 3 phenotype;
- all floral dimensions constitute one coordinated island syndrome;
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
| 3 | `shimahotarubukuro` | one lineage across five Izu islands | **How is it phenotypically realized?** | direct multidimensional, unequal within-lineage divergence |

Together:

```text
WHEN / WHERE island filtering is detectable             [Chapter 1]
    ↓
HOW / WHY responses branch under altered interactions   [Chapter 2]
    ↓
HOW that conditionality is realized across floral axes  [Chapter 3]
```

The Chapter 3 contribution is: **to anchor the dissertation's broad comparative and mechanistic arguments in a directly measured, multidimensional phenotype within the same Izu island series that provides the strongest mechanistic bridge.**
