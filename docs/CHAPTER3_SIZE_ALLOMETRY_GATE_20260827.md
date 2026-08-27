# Chapter 3 size/allometry interpretation gate

Updated: 2026-08-27

## Why this gate is necessary

The prespecified phenotype-axis contrast shows a large separation between absolute/investment traits and proportional/spatial traits. That is a real description of the retained `Pst` values, but it is not by itself evidence that multiple ecological mechanisms produced independent trait branches: absolute dimensions can co-diverge under a common floral-size shift, while ratios mathematically remove part of that scale variation.

The repository already contained a stronger allometry and multivariate analysis before the 2026-08-27 Chapter 2–3 reframing. This gate therefore uses the existing analysis, without retuning it, to ask a stricter question:

> **After accounting for shared floral body size, which Chapter 3 channels still show among-island structure?**

The frozen values below come from successful `reproduce-pipeline` run `32233626007` at commit `4a286534569d366470fc1556b8380aa8ef4f0b69`, artifact `trait-tables` (SHA-256 `1921a6168b8c7c296d8b76e33bc23acb920bb05c524d3180fe0591593aca07de`).

## Trait-level allometry

The existing analysis defines floral body size from corolla length and full-equivalent corolla width, then tests functional traits against log body size with site-level permutation inference.

| Trait | Size-adjusted island result | Island×size interaction | Common-support sensitivity | Reading |
|---|---:|---:|---:|---|
| mouth width | BH p = **0.0368**, partial R² = **0.533** | BH p = **0.020**, partial R² = **0.499** | BH p = 0.133 | strong evidence for island-specific allometric reconfiguration over the full observed range; common-support restriction weakens certainty |
| throat width | BH p = 0.985, partial R² = 0.046 | BH p = 0.572 | BH p = 0.933 | largely compatible with common size scaling / unresolved independent change |
| reproductive-organ length | BH p = **0.0216**, partial R² = **0.464** | BH p = 0.723 | BH p = 0.221 | additive island shift beyond the full-range common allometry; restricted common-support sensitivity weaker |
| guide coverage | BH p = **0.0216**, partial R² = **0.464** | BH p = 0.723 | BH p = 0.221 | island-level guide-amount shift beyond the full-range size relation; restricted common-support sensitivity weaker |

This immediately changes the interpretation of the raw `Pst` pattern. The large throat-width `Pst` should not be promoted as an independent access-geometry response: after size adjustment it is consistent with shared floral scaling. Mouth width is different: its island effect is expressed partly as a change in allometric relationship itself. Reproductive-organ length and guide amount retain full-range island shifts after size adjustment, although the narrower common-support subset is less decisive.

## Multivariate residual structure

The absolute six-trait phenotype has a leading multivariate `Pst` analogue of **0.592** (hierarchical-bootstrap 95% interval **0.526–0.812**).

After removing the common body-size relation from mouth width, throat width, reproductive-organ length and guide coverage, the leading residual axis still has `Pst` analogue **0.286** with bootstrap interval **0.245–0.639**.

Its leading loadings are:

- reproductive-organ residual: `+0.745`;
- mouth-width residual: `−0.555`;
- guide-coverage residual: `+0.365`;
- throat-width residual: `+0.067`.

Therefore simple floral miniaturisation is insufficient to erase the among-island multivariate phenotype. A substantial residual axis remains, concentrated especially in reproductive-interface length, mouth allometry and guide investment.

## But the modules are not proven to branch independently

A separate pre-existing module analysis gives a rank-1 shared trajectory fraction of **0.775** (bootstrap 95% **0.546–0.886**). The module × island permutation test is not significant (`p = 0.3907`; partial R² `0.160`). Its own interpretation is that the current data **do not reject a shared three-module syndrome trajectory despite heterogeneous trait-level response modes**.

That result matters for the Chapter 2 connection. Chapter 3 should not claim that Chapter 2 response branching has already been observed as statistically independent trait-module branching within *C. microdonta*.

The stronger and more accurate Chapter 3 statement is:

> ***Campanula microdonta* shows a strongly coordinated island size/investment shift plus selected departures from simple size scaling in mouth allometry, reproductive-interface length, guide investment and a residual multivariate phenotype axis.**

This is a better empirical endpoint for Chapter 2 than either extreme:

- `all traits form one uniform syndrome`; or
- `every trait channel branches independently`.

## Chapter 2 → Chapter 3 interpretation

Chapter 2 establishes that ecological response geometry can be conditional on source state and realised community. Chapter 3 now contributes a compatible but distinct empirical observation:

```text
large coordinated phenotype component
        +
selected channel-specific departures from common size scaling
```

That pattern leaves room for conditional ecological realization without pretending that morphology alone identifies the responsible mechanism.

The direct bridge still requires plant-linked visitor identity/functional traits, interaction weights, pollen deposition/effective service and reproductive dependency/outcome.

## Claim ceiling

Do not use the raw absolute-versus-ratio contrast alone as evidence for mechanistic response branching. Do not call throat-width divergence independent of size. Do not treat the residual multivariate `Pst` analogue as Qst or selection. Do not infer Bombus causation from the allometric departures. The present evidence supports coordinated scaling **plus** selected residual phenotype structure, not a fully identified ecological mechanism.