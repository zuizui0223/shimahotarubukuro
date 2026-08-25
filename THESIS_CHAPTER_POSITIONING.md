# Thesis positioning — Chapter 3

## Role in the dissertation

This repository is the **Chapter 3 / focal empirical phenotype** component of the dissertation.

The shared dissertation-level question is:

> **How does geographic isolation alter plant reproduction through changes in ecological interactions, and why do those changes produce different floral outcomes across islands and lineages?**

The three empirical levels are intentionally separated:

- [`zuizui0223/island`](https://github.com/zuizui0223/island) — **Chapter 1:** identifies the macroecological and biogeographic conditions under which floral and reproductive assemblage filtering appears.
- [`zuizui0223/izu-core`](https://github.com/zuizui0223/izu-core) — **Chapter 2:** asks how pollination-regime change is translated into reproductive and interaction responses and applies a stricter causal-identification standard.
- `shimahotarubukuro` — **Chapter 3:** measures the realized within-lineage floral phenotype across a concrete Izu island series in *Campanula microdonta*.

Chapter 3 therefore asks:

> **Which floral and reproductive-interface traits actually diverge among island populations within one focal lineage, and which phenotypic axes carry the strongest island differentiation?**

## Why Chapter 3 is needed

Chapter 1 works at the assemblage level and cannot by itself demonstrate that an extant lineage changes its phenotype across islands. Chapter 2 resolves mechanisms and response branching, but it deliberately refuses to infer morphology from pollinator identity alone.

Chapter 3 supplies a different kind of evidence: a direct, reproducible measurement of the phenotype itself within a focal lineage.

```text
Chapter 1: where / when does filtering appear?
        ↓
Chapter 2: how can pollination-regime change generate different biological responses?
        ↓
Chapter 3: what floral phenotype is actually differentiated within a focal island lineage?
```

It is therefore an **empirical realization / focal-case chapter**, not a third global test and not a substitute for the Chapter 2 causal measurements.

## Empirical system

The current publication pipeline measures flattened, pressed *Campanula microdonta* corollas from five Izu Islands:

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

## Relationship to the Channel-gated island assembly hypothesis

Chapter 1 proposes that a classical floral island syndrome should be conditional on loss or weakening of a pollination channel that is regionally available, rather than on pollinator absence alone.

Chapter 3 provides a focal phenotype that is **compatible with testing the biological expression of that idea**, because *C. microdonta* occurs across an Izu island series where pollination regimes differ and the measured traits include floral-access and signal components.

However, the present `shimahotarubukuro` pipeline does **not** estimate a causal Bombus-present versus Bombus-absent effect. The observed island differentiation can be discussed alongside the Chapter 1 channel-gated prediction and the Chapter 2 Izu functional evidence, but Chapter 3 does not convert that contextual agreement into historical causal proof.

The appropriate thesis-level interpretation is:

```text
Chapter 1 identifies a conditional macroecological pattern.
Chapter 2 identifies which pollination/reproductive mechanisms are or are not supported.
Chapter 3 demonstrates that a focal lineage carries substantial, trait-specific phenotypic divergence across the same island context.
```

## Relationship to Chapter 2 response branching

Chapter 2 predicts that a shared change in pollination function need not generate one universal morphological response. Chapter 3 is therefore not expected to show that every measured floral trait changes in the same direction or with the same effect size.

The relevant empirical question is instead whether divergence is **trait specific**. The current results support that view: reproductive-organ length, mouth width, corolla area, corolla length, guide coverage and throat width show substantial differentiation, whereas aspect ratio, tube flare, lobe incision and guide spatial metrics are much weaker after site correction.

This makes Chapter 3 a concrete example of the response-branching principle: different parts of the same flower can carry very different amounts of island-associated divergence.

## What Chapter 3 identifies

Chapter 3 can support claims about:

- reproducible among-island phenotypic differentiation within *C. microdonta*;
- which floral dimensions carry the strongest divergence;
- whether divergence is concentrated in size, access geometry, reproductive-interface, or signal traits;
- the spatial structure of nectar-guide pigment within flowers;
- site-corrected island differences; and
- the relative magnitude of phenotypic differentiation using `Pst` and pairwise `Pst`.

It provides the **phenotypic observation layer** that Chapter 1 cannot obtain from assemblage composition and that Chapter 2 should not infer from pollinator identity.

## Claim boundary

This repository should not claim that:

- the Oshima-versus-other-islands contrast is a causal Bombus experiment;
- `Pst` is `Qst`;
- phenotypic differentiation alone demonstrates natural selection;
- a decrease in floral size, access width, reproductive-organ length, or guide coverage was caused historically by Bombus loss;
- latitude or island order uniquely identifies the causal environmental driver; or
- all floral dimensions constitute one coordinated island syndrome.

The Chapter 3 contribution is narrower and stronger: **to provide a directly measured, within-lineage empirical example of multidimensional floral divergence across an island series, and to locate precisely which phenotype axes require mechanistic explanation from the broader dissertation framework.**

## Three-chapter thesis architecture

| Chapter | Repository | Scale | Primary question | Main evidential contribution |
| --- | --- | --- | --- | --- |
| 1 | `island` | global island floras | **Where and when?** | boundary conditions for channel-gated assemblage filtering |
| 2 | `izu-core` | Izu networks / lineages / plants | **How and why?** | mechanistic identification and response branching |
| 3 | `shimahotarubukuro` | one focal lineage across five islands | **What phenotype actually diverges?** | direct multidimensional floral measurement and within-lineage phenotypic divergence |

Together the intended argument is:

```text
Geographic isolation
    ↓
conditional pollination-channel context          [Chapter 1]
    ↓
plant-specific reproductive / interaction response [Chapter 2]
    ↓
realized, trait-specific floral divergence          [Chapter 3]
```

The chapters are complementary rather than nested proofs: Chapter 1 sets the boundary conditions, Chapter 2 sets the mechanistic admission standard, and Chapter 3 supplies a high-resolution empirical phenotype that those broader explanations must account for.
