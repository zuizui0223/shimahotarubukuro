#!/usr/bin/env python3
"""Compare among-island divergence axes with within-island phenotypic covariance.

This is a site-level phenotypic P-matrix diagnostic, not a genetic G-matrix test.
It asks whether island divergence follows the dominant direction of within-island
phenotypic variation or is reoriented relative to it. It cannot distinguish
selection from plasticity or population history.
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
ABS = ["corolla_length_mm", "corolla_width_fulleq_mm", "mouth_width_mm",
       "throat_width_mm", "organ_length_mm", "guide_coverage_pct"]
ADJ = ["mouth_width_mm", "throat_width_mm", "organ_length_mm", "guide_coverage_pct"]


def zframe(d, cols):
    x = d[cols].astype(float)
    return (x - x.mean()) / x.std(ddof=1)


def variance_components(d, cols):
    z = zframe(d, cols)
    z["_g"] = d.island.to_numpy()
    arrays = [x[cols].to_numpy(float) for _, x in z.groupby("_g") if len(x) >= 1]
    ni = np.array([len(a) for a in arrays], float)
    n, k, p = int(ni.sum()), len(arrays), len(cols)
    grand = np.vstack(arrays).mean(0)
    ssw = np.zeros((p, p)); ssb = np.zeros((p, p))
    for nn, a in zip(ni, arrays):
        m = a.mean(0); c = a - m
        ssw += c.T @ c
        delta = (m - grand)[:, None]
        ssb += nn * (delta @ delta.T)
    W = ssw / (n - k)
    MSB = ssb / (k - 1)
    n0 = (n - (ni @ ni) / n) / (k - 1)
    B = (MSB - W) / n0
    B = (B + B.T) / 2
    vals, vecs = np.linalg.eigh(B)
    B = vecs @ np.diag(np.clip(vals, 0, None)) @ vecs.T
    return (B + B.T) / 2, W


def leading(M):
    vals, vecs = np.linalg.eigh((M + M.T) / 2)
    j = int(np.argmax(vals))
    v = vecs[:, j]
    return v / np.linalg.norm(v), float(vals[j] / np.clip(vals.sum(), np.finfo(float).tiny, None))


def metrics(d, cols):
    B, W = variance_components(d, cols)
    b, bfrac = leading(B)
    w, wfrac = leading(W)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(b, w)), -1, 1))))
    return angle, bfrac, wfrac, b, w


def bootstrap(d, cols, nboot=2500, seed=20260829):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nboot):
        pieces = []
        for _, di in d.groupby("island", observed=True):
            pieces.append(di.iloc[rng.integers(0, len(di), len(di))].copy())
        try:
            a, bf, wf, _, _ = metrics(pd.concat(pieces, ignore_index=True), cols)
            rows.append((a, bf, wf))
        except Exception:
            pass
    return np.asarray(rows, float)


def run_one(d, cols, label, seed):
    a, bf, wf, b, w = metrics(d, cols)
    boot = bootstrap(d, cols, 2500, seed)
    row = {
        "analysis": label,
        "n_sites": len(d),
        "n_traits": len(cols),
        "angle_deg_B1_vs_W1": a,
        "angle_boot_lo": np.quantile(boot[:, 0], .025),
        "angle_boot_hi": np.quantile(boot[:, 0], .975),
        "B1_variance_fraction": bf,
        "W1_variance_fraction": wf,
        "B1_fraction_boot_lo": np.quantile(boot[:, 1], .025),
        "B1_fraction_boot_hi": np.quantile(boot[:, 1], .975),
        "W1_fraction_boot_lo": np.quantile(boot[:, 2], .025),
        "W1_fraction_boot_hi": np.quantile(boot[:, 2], .975),
        "n_bootstrap": len(boot),
        "interpretation": (
            "angle near 0 = divergence follows dominant within-island site-level phenotypic covariance; "
            "angle near 90 = reoriented divergence. Phenotypic P-matrix only, not evidence of selection."
        ),
    }
    for c, x in zip(cols, b):
        row[f"B1_loading_{c}"] = float(x)
    for c, x in zip(cols, w):
        row[f"W1_loading_{c}"] = float(x)
    return row


def main():
    plant = pd.read_csv(R / "plant_means.csv", encoding="utf-8-sig")
    plant = plant[plant.island.isin(ISLANDS)].copy()
    plant["site"] = plant.island + "_" + plant.no.astype(str)
    absolute = plant.groupby(["island", "site"], observed=True)[ABS].mean().reset_index()

    sr = pd.read_csv(R / "multivariate_site_residuals.csv")
    adjusted = sr[sr.trait.isin(ADJ)].pivot_table(
        index=["island", "site"], columns="trait",
        values="mean_size_adjusted_residual"
    ).reset_index()

    rows = [
        run_one(absolute, ABS, "absolute_site_means", 20260829),
        run_one(adjusted, ADJ, "body_size_adjusted_functional_site_residuals", 20260830),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(R / "syndrome_divergence_covariance_alignment.csv", index=False)

    print("\n=== divergence vs within-island phenotypic covariance ===")
    for _, r in out.iterrows():
        print(
            f"{r.analysis:45s} angle={r.angle_deg_B1_vs_W1:.1f} deg "
            f"[{r.angle_boot_lo:.1f},{r.angle_boot_hi:.1f}] "
            f"B1frac={r.B1_variance_fraction:.3f} W1frac={r.W1_variance_fraction:.3f}"
        )
    print("Guardrail: W is a site-level phenotypic P-matrix, not a genetic G-matrix.")


if __name__ == "__main__":
    main()
