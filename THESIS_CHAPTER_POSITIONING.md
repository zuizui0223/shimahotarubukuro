# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal empirical phenotype** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, and why do those changes produce different floral outcomes across islands and lineages?**

The three empirical levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** asks **when and where** isolation-associated floral/reproductive filtering is detectable and where multivariate response vectors differ.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks **why** supported contexts generate different response architectures.
- `shimahotarubukuro` — **Chapter 3:** measures **what floral phenotype axes actually diverge** within one focal Izu lineage.

Chapter 3 asks:

> **Which floral and reproductive-interface traits actually diverge among island populations within *Campanula microdonta*, and which phenotypic axes carry the strongest island differentiation?**

## Why Chapter 3 is needed after the Chapter 1 when/where result

Chapter 1 now establishes that isolation-associated filtering is detectable in both northern mid-latitude and tropical island floras, persists in native non-endemic assemblages, and is expressed as significantly different multivariate response vectors between those regions.

But Chapter 1 is an assemblage-level analysis. It cannot demonstrate that one extant lineage changes its phenotype across islands.

Chapter 3 supplies that missing observation layer:

```text
Chapter 1
WHERE does assemblage-level filtering occur?
        ↓
Chapter 2
WHY can ecological interaction states generate different response architectures?
        ↓
Chapter 3
WHAT phenotype axes actually diverge within one lineage?
```

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

Several proportional or allocation traits remain weaker after site correction, so the phenotype is explicitly treated as multidimensional rather than collapsed into one island-adaptation score.

## Relationship to Chapter 1

The canonical Chapter 1 result is now:

> **Isolation-associated floral/reproductive filtering is confirmatorily detectable in northern mid-latitude and tropical island floras, persists within native non-endemic assemblages, and is expressed as different multivariate response vectors between those two contexts.**

Chapter 3 does not retest that global boundary result.

Instead it asks whether a **single northern-midlatitude lineage** also shows multidimensional among-island phenotype differentiation. The answer is yes: different phenotype axes diverge strongly but unequally.

Thus Chapter 3 provides a focal realization of the broader dissertation problem without claiming that the global assemblage vector and the *C. microdonta* phenotype are the same estimand.

## Relationship to Chapter 2

Chapter 2 distinguishes mechanisms such as pollinator functional diversity, trait matching, effective service, reproductive assurance, functional replacement, network state and non-pollination history.

Chapter 3 supplies the phenotype those mechanisms would need to explain in the focal lineage.

Different Chapter 3 axes show different divergence magnitudes, which is consistent with a response-branching framework but does not identify the mechanism by itself.

## Working interpretation

The current evidence supports:

> ***Campanula microdonta* populations across the Izu island series show pronounced, multidimensional floral divergence concentrated in particular absolute-size, access, reproductive-interface and nectar-guide axes rather than expressed uniformly across all floral-shape metrics.**

This is stronger and more defensible than claims such as:

- all floral traits become smaller or more generalized with island isolation;
- Bombus loss caused the observed phenotype;
- geographic island order uniquely identifies the historical driver.

In particular, reductions in mouth and throat width are **not automatically evidence of increased floral accessibility or generalization**; they may partly reflect overall floral scaling.

## Claim boundary

This repository should not claim that:

- the Oshima-versus-other-islands contrast is a causal Bombus experiment;
- `Pst` is `Qst`;
- phenotypic differentiation alone demonstrates natural selection;
- lower floral size, access width, reproductive-organ length or guide coverage was historically caused by Bombus loss;
- Chapter 1's northern-midlatitude response vector directly predicts the within-lineage Chapter 3 phenotype; or
- all floral dimensions constitute one coordinated island syndrome.

## Three-chapter architecture

| Chapter | Repository | Scale | Primary question | Main contribution |
| --- | --- | --- | --- | --- |
| 1 | `island` | global island floras | **When / where?** | contexts where filtering is detectable and response vectors differ |
| 2 | `izu-core` | mechanistic response architecture | **Why / how?** | candidate mechanisms, branching, propagation and buffering |
| 3 | `shimahotarubukuro` | one lineage across five islands | **What phenotype?** | direct multidimensional within-lineage divergence |

Together:

```text
WHEN / WHERE isolation filtering occurs                 [Chapter 1]
    ↓
WHY supported contexts generate different responses     [Chapter 2]
    ↓
WHAT phenotype axes diverge in one focal lineage         [Chapter 3]
```

The Chapter 3 contribution is: **to provide a directly measured within-lineage phenotype that anchors the dissertation's global boundary and mechanistic arguments in a concrete island system.**
