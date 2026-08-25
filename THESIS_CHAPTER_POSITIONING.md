# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal empirical phenotype** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, and why do those changes produce different floral outcomes across islands and lineages?**

The three empirical levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** identifies when and where floral/reproductive island syndromes differ among biogeographic contexts.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks why those responses branch, propagate, buffer or decouple under altered ecological interactions.
- `shimahotarubukuro` — **Chapter 3:** measures the realized within-lineage floral phenotype across a concrete Izu island series in *Campanula microdonta*.

Chapter 3 therefore asks:

> **Which floral and reproductive-interface traits actually diverge among island populations within one focal lineage, and which phenotypic axes carry the strongest island differentiation?**

## Why Chapter 3 is needed

Chapter 1 works at the assemblage level and tests context dependence; it cannot demonstrate that an extant lineage changes its phenotype across islands. Chapter 2 addresses mechanisms and response branching, but it deliberately refuses to infer morphology from pollinator identity alone.

Chapter 3 supplies a different kind of evidence: direct, reproducible measurement of the phenotype itself within a focal lineage.

```text
Chapter 1: when / where do different regional trait syndromes appear?
        ↓
Chapter 2: why can altered ecological interactions generate different responses?
        ↓
Chapter 3: what floral phenotype is actually differentiated within one lineage?
```

It is therefore an **empirical realization / focal-case chapter**, not a third global test and not a substitute for Chapter 2 mechanistic evidence.

## Empirical system

The publication pipeline measures flattened, pressed *Campanula microdonta* corollas from five Izu Islands:

- Oshima
- Toshima
- Niijima
- Shikinejima
- Kozushima

The retained dataset contains **218 corollas from 125 individual plants**. Inference is performed on plant-level means, with island effects evaluated after accounting for site structure. Global and pairwise `Pst` are used as phenotypic-divergence summaries.

The retained phenotype domains include:

- corolla length, width, and area;
- throat and mouth width;
- aspect and tube-flare ratios;
- lobe incision;
- reproductive-organ length and organ/corolla ratio;
- nectar-guide coverage; and
- guide spatial concentration.

## Current empirical pattern

The current site-corrected analysis identifies island differentiation in several biologically interpretable axes. The strongest retained `Pst` values include:

| trait | Pst | Oshima mean | Kozushima mean |
| --- | ---: | ---: | ---: |
| reproductive-organ length | 0.475 | 22.68 mm | 14.07 mm |
| mouth width | 0.374 | 50.99 mm | 28.67 mm |
| corolla area | 0.370 | 1553.78 mm² | 611.94 mm² |
| corolla length | 0.332 | 35.92 mm | 22.88 mm |
| corolla width | 0.283 | 53.08 mm | 34.70 mm |
| guide coverage | 0.244 | 31.08% | 6.90% |
| throat width | 0.238 | 31.66 mm | 22.69 mm |

These results show that island divergence is not confined to a single generic size metric. It spans **display size, floral access geometry, reproductive-interface length, and nectar-guide investment**.

The geographic pattern is not forced into one perfectly monotonic syndrome: individual islands can depart from a simple ordered gradient, and the chapter preserves trait-specific and pairwise structure rather than collapsing all measurements into one island-adaptation score.

## Relationship to Chapter 1

Chapter 1 now tests a **biogeographically contingent floral island syndrome hypothesis**:

> comparable isolation need not generate one universal floral/reproductive response; regional trait vectors may differ after status, lineage/source-pool, climate and observation structure are represented.

Chapter 3 does not retest that global assemblage hypothesis. It provides a focal within-lineage observation unavailable to Chapter 1:

> island populations of the same lineage can differ strongly in several floral and reproductive-interface traits.

Some Chapter 3 directions — such as lower corolla dimensions and lower guide coverage toward Kozushima relative to Oshima — may later be compared with broader regional trait directions from Chapter 1. However, Chapter 3 does not force those measurements into a predeclared Bombus or generalization syndrome.

In particular, reductions in mouth and throat width are **not automatically evidence of increased floral accessibility or generalization**; they may partly reflect overall floral scaling.

## Relationship to Chapter 2

Chapter 2 predicts **response branching**: a shared ecological perturbation need not generate one common downstream trait response across lineages or trait channels.

Chapter 3 adds a within-lineage version of the same lesson. Different phenotype axes have different divergence magnitudes and statistical support. Absolute dimensions and guide coverage diverge strongly, whereas several ratios and spatial guide-allocation metrics do not show equally strong site-corrected differences.

This makes Chapter 3 the phenotype endpoint of the thesis:

```text
Chapter 1: identify where regional trait syndromes differ
        ->
Chapter 2: identify why ecological responses can branch / propagate / buffer
        ->
Chapter 3: measure which phenotype axes actually carry divergence in one focal lineage
```

## Working Chapter 3 interpretation

The current evidence supports:

> *Campanula microdonta* populations across the Izu island series show pronounced, multidimensional floral divergence, but the divergence is concentrated in particular absolute-size, access, reproductive-interface and nectar-guide axes rather than expressed as one uniform change across all floral-shape metrics.

This is stronger and more defensible than claims such as:

- "all floral traits become smaller or more generalized with island isolation";
- "Bombus loss caused the observed phenotype."

## What Chapter 3 identifies

Chapter 3 can support claims about:

- reproducible among-island phenotypic differentiation within *C. microdonta*;
- which floral dimensions carry the strongest divergence;
- whether divergence is concentrated in size, access geometry, reproductive-interface or signal traits;
- the spatial structure of nectar-guide pigment within flowers;
- site-corrected island differences; and
- the relative magnitude of phenotypic differentiation using `Pst` and pairwise `Pst`.

It provides the **phenotypic observation layer** that Chapter 1 cannot obtain from assemblage composition and that Chapter 2 should not infer from pollinator identity.

## Claim boundary

This repository should not claim that:

- the Oshima-versus-other-islands contrast is a causal Bombus experiment;
- `Pst` is `Qst`;
- phenotypic differentiation alone demonstrates natural selection;
- a decrease in floral size, access width, reproductive-organ length or guide coverage was caused historically by Bombus loss;
- latitude or island order uniquely identifies the causal driver; or
- all floral dimensions constitute one coordinated island syndrome.

The Chapter 3 contribution is narrower and stronger: **to provide a directly measured, within-lineage empirical example of multidimensional floral divergence across an island series, and to locate precisely which phenotype axes require mechanistic explanation from the broader dissertation framework.**

## Three-chapter thesis architecture

| Chapter | Repository | Scale | Primary question | Main evidential contribution |
| --- | --- | --- | --- | --- |
| 1 | `island` | global island floras | **When and where?** | biogeographic contingency of floral/reproductive trait syndromes |
| 2 | `izu-core` | mechanistic island-response architecture | **How and why?** | branching, propagation, buffering and candidate mechanisms |
| 3 | `shimahotarubukuro` | one focal lineage across five islands | **What phenotype actually diverges?** | direct multidimensional floral measurement and within-lineage phenotypic divergence |

Together:

```text
Isolation × biogeographic context
    ↓
regional trait syndromes                          [Chapter 1]
    ↓
candidate ecological mechanisms / response modes [Chapter 2]
    ↓
realized, trait-specific floral divergence        [Chapter 3]
```

The chapters are complementary rather than nested proofs: Chapter 1 establishes the pattern boundary, Chapter 2 tests mechanism and response architecture, and Chapter 3 supplies a high-resolution empirical phenotype those explanations must account for.
