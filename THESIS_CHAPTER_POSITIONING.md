# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal empirical phenotype** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction and floral phenotype, and why do different phenotype components respond differently across islands and lineages?**

The three empirical levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** tests whether isolation produces one coherent floral/reproductive syndrome or component-specific reorganization at the global assemblage scale.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks which ecological interaction states make those response components branch, propagate, buffer or decouple.
- `shimahotarubukuro` — **Chapter 3:** measures the realized within-lineage floral phenotype across a concrete Izu island series in *Campanula microdonta*.

Chapter 3 therefore asks:

> **Which floral and reproductive-interface traits actually diverge among island populations within one focal lineage, and which phenotypic axes carry the strongest island differentiation?**

## Why Chapter 3 is needed

Chapter 1 works at the assemblage level and now shows that isolation-associated change is better represented as **component-specific reorganization** than as one coherent classic syndrome. It cannot demonstrate that an extant lineage changes its phenotype across islands. Chapter 2 addresses why response components can branch, but it deliberately refuses to infer morphology from pollinator identity alone.

Chapter 3 supplies a different kind of evidence: direct, reproducible measurement of the phenotype itself within a focal lineage.

```text
Chapter 1: which floral components reorganize under isolation?
        ↓
Chapter 2: why can ecological interactions make components respond differently?
        ↓
Chapter 3: which phenotype axes actually diverge within one lineage?
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

Chapter 1 no longer has a headline **biogeographically contingent island syndrome** result. Its current frozen evidence supports:

> **Isolation-associated floral change is component-specific rather than one coherent `generalized + plain + SC` syndrome; strong northern-midlatitude within-context signals exist, but confirmatory regional slope heterogeneity is not established.**

Chapter 3 does not retest that assemblage result. Instead, it provides a focal within-lineage observation unavailable to Chapter 1:

> **Different phenotype components within the same lineage can also diverge by very different amounts.**

This makes Chapter 3 a particularly natural empirical continuation of the Chapter 1 result. Chapter 1 finds decoupling among floral categories across floras; Chapter 3 asks whether decoupling is also visible across phenotype axes within one lineage.

Some Chapter 3 directions — such as lower corolla dimensions and lower guide coverage toward Kozushima relative to Oshima — may later be compared with Chapter 1 atomic component directions. However, Chapter 3 does not force those measurements into a predeclared Bombus, generalization or island-syndrome score.

In particular, reductions in mouth and throat width are **not automatically evidence of increased floral accessibility or generalization**; they may partly reflect overall floral scaling.

## Relationship to Chapter 2

Chapter 2 predicts **response branching**: a shared ecological perturbation need not generate one common downstream trait response across lineages or trait channels.

Chapter 3 adds a within-lineage version of the same lesson. Different phenotype axes have different divergence magnitudes and statistical support. Absolute dimensions and guide coverage diverge strongly, whereas several ratios and spatial guide-allocation metrics do not show equally strong site-corrected differences.

The three chapters therefore align around the same structural idea:

```text
Chapter 1: assemblage components do not collapse into one syndrome
        ->
Chapter 2: interaction states determine response branching / propagation / buffering
        ->
Chapter 3: phenotype axes within one lineage also diverge non-uniformly
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
| 1 | `island` | global island floras | **Which components reorganize?** | component-specific isolation-associated floral reorganization, with formal regional boundary tests |
| 2 | `izu-core` | mechanistic island-response architecture | **How and why?** | branching, propagation, buffering and candidate mechanisms |
| 3 | `shimahotarubukuro` | one focal lineage across five islands | **What phenotype actually diverges?** | direct multidimensional floral measurement and within-lineage phenotypic divergence |

Together:

```text
component-specific assemblage reorganization         [Chapter 1]
    ↓
candidate interaction states / response modes        [Chapter 2]
    ↓
realized, trait-specific floral divergence            [Chapter 3]
```

The chapters are complementary rather than nested proofs: Chapter 1 establishes the component structure of the macroecological pattern, Chapter 2 tests why components can decouple mechanistically, and Chapter 3 supplies a high-resolution empirical phenotype those explanations must account for.
