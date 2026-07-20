#!/usr/bin/env python3
"""Test whether nectar-guide divergence tracks the loss of bumblebee pollinators.

Hypothesis (reviewer's): purple nectar guides signal bumblebees into the bell. The
Izu Islands lack Bombus except Oshima (Bombus ardens), so on the bumblebee-free
islands the guide is a cost with no receiver. If divergence has run past neutral
drift, guide investment should be reduced there relative to Oshima, and that
reduction should be DECOUPLED from corolla size (i.e. not just passive allometry).

Three complementary tests, honest about their limits:

1. Bombus contrast. Guide coverage / spot count on Oshima (Bombus present) vs the
   pooled bumblebee-free islands (Mann-Whitney U, rank-based).

2. Size-controlled (ANCOVA-style). Regress guide coverage on corolla length across
   all corollas; add a Bombus-present indicator. If the indicator stays significant
   after size, the guide difference is not explained by Oshima simply being larger -
   the signature expected under adaptive reduction, not neutral allometric scaling.

3. Pst-analog. Among- vs within-island variance component for each trait
   (Pst = Vb / (Vb + 2 Vw), assuming c/h^2 = 1). Guide traits are compared with
   size traits: if guide Pst greatly exceeds size Pst, guide divergence is stronger
   than the shared size divergence - consistent with differential selection. This is
   a PHENOTYPIC surrogate; a definitive Qst-Fst test needs neutral genetic markers,
   which we do not have, so this is presented as suggestive, not conclusive.

Writes results_shimask_all/guide_divergence_stats.csv and prints a summary.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from plot_island_traits import ORDER, island_of

BOMBUS_PRESENT = {"Oshima"}  # only island with a bumblebee (Bombus ardens)


def load() -> list[dict]:
    p = Path("results_shimask_all/pollination_traits.csv")
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    for r in rows:
        r["island"] = island_of(r["sheet"])
        r["bombus"] = 1 if r["island"] in BOMBUS_PRESENT else 0
    return rows


def col(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] not in ("", "nan")])


def pst(rows, key):
    """Pst = Vb / (Vb + 2 Vw) from a one-way island decomposition."""
    groups = [col([r for r in rows if r["island"] == i], key) for i in ORDER]
    groups = [g for g in groups if g.size >= 2]
    grand = np.concatenate(groups)
    n = grand.size
    k = len(groups)
    means = [g.mean() for g in groups]
    ni = [g.size for g in groups]
    ss_b = sum(nj * (mj - grand.mean()) ** 2 for nj, mj in zip(ni, means))
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n - k)
    n0 = (n - sum(x * x for x in ni) / n) / (k - 1)
    vb = max((ms_b - ms_w) / n0, 0.0)
    vw = ms_w
    f, p = stats.f_oneway(*groups)
    return vb / (vb + 2 * vw) if (vb + vw) > 0 else 0.0, f, p


def ols_ancova(rows):
    """coverage = b0 + b1*length + b2*bombus ; return coeffs, ses, p-values, R^2."""
    rr = [r for r in rows if r["guide_coverage_pct"] not in ("", "nan")
          and r["corolla_length_mm"] not in ("", "nan")]
    y = np.array([float(r["guide_coverage_pct"]) for r in rr])
    length = np.array([float(r["corolla_length_mm"]) for r in rr])
    bombus = np.array([float(r["bombus"]) for r in rr])
    X = np.column_stack([np.ones_like(y), length, bombus])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return beta, se, pvals, r2, len(y)


def main() -> None:
    rows = load()
    out_rows = []

    print("=== 1. Bombus present (Oshima) vs bumblebee-free islands ===")
    present = [r for r in rows if r["bombus"] == 1]
    absent = [r for r in rows if r["bombus"] == 0]
    for key in ("guide_coverage_pct", "n_guide_spots", "guide_contrast_dE", "guide_saturation"):
        a, b = col(present, key), col(absent, key)
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {key:20s} present {a.mean():7.1f} (n={a.size})  "
              f"absent {b.mean():7.1f} (n={b.size})  U p={p:.2e}")
        out_rows.append({"test": "bombus_vs_free", "trait": key,
                         "present_mean": round(a.mean(), 2), "absent_mean": round(b.mean(), 2),
                         "stat": round(u, 1), "p": f"{p:.3g}"})

    print("\n=== 2. Size-controlled ANCOVA: coverage ~ length + Bombus ===")
    beta, se, pv, r2, n = ols_ancova(rows)
    names = ["intercept", "corolla_length_mm", "bombus_present"]
    for nm, b, s, p in zip(names, beta, se, pv):
        print(f"  {nm:20s} coef {b:8.3f}  se {s:6.3f}  p {p:.2e}")
    print(f"  R^2 = {r2:.3f}  (n = {n})")
    print(f"  -> After controlling corolla size, Bombus-present adds "
          f"{beta[2]:+.1f} pct-points of guide coverage (p={pv[2]:.2e}).")
    out_rows.append({"test": "ancova_coverage", "trait": "bombus_present_coef",
                     "present_mean": round(beta[2], 2), "absent_mean": "",
                     "stat": round(r2, 3), "p": f"{pv[2]:.3g}"})

    print("\n=== 3. Pst-analog: among-island divergence (guide vs size traits) ===")
    guide_traits = ["guide_coverage_pct", "n_guide_spots", "guide_contrast_dE", "guide_saturation"]
    size_traits = ["corolla_length_mm", "throat_width_mm", "mouth_width_mm", "style_length_mm"]
    for label, keys in (("GUIDE", guide_traits), ("SIZE", size_traits)):
        for key in keys:
            val, f, p = pst(rows, key)
            print(f"  [{label:5s}] {key:20s} Pst {val:5.2f}  (ANOVA F={f:6.1f}, p={p:.2e})")
            out_rows.append({"test": f"pst_{label.lower()}", "trait": key,
                             "present_mean": round(val, 3), "absent_mean": "",
                             "stat": round(f, 1), "p": f"{p:.3g}"})

    out = Path("results_shimask_all/guide_divergence_stats.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "trait", "present_mean", "absent_mean", "stat", "p"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
