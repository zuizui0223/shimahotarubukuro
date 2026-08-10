#!/usr/bin/env python3
"""Exploratory multivariate phenotypic divergence across Izu islands.

This is deliberately labelled a phenotypic P_ST analogue, not a Q_ST-F_ST test.
It uses plant means to separate within-island covariance from among-island
variance components and asks whether nectar-guide coverage is coordinated with
floral size after removing island means.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats
import statsmodels.formula.api as smf

RESULTS = Path("results_shimask_all")
ISLAND_ORDER = ["oshima", "toshima", "niijima", "shikine", "kozu"]

SIZE_TRAITS = [
    "corolla_length_mm",
    "corolla_width_fulleq_mm",
    "mouth_width_mm",
    "throat_width_mm",
    "organ_length_mm",
]
SHAPE_TRAITS = [
    "corolla_aspect_L_W",
    "tube_flare_W_throat",
    "lobe_incision_ratio",
    "organ_corolla_ratio",
]
GUIDE = "guide_coverage_pct"
CORE_TRAITS = SIZE_TRAITS + [GUIDE]


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=1)


def pc1_scores(frame: pd.DataFrame, columns: list[str], prefix: str):
    clean = frame[columns].dropna()
    z = zscore(clean)
    cov = np.cov(z.values, rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    loading = vectors[:, 0].copy()
    if np.nanmean(loading) < 0:
        loading *= -1
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    scores.loc[clean.index] = z.values @ loading
    table = pd.DataFrame({
        "module": prefix,
        "trait": columns,
        "pc1_loading": loading,
        "pc1_variance_fraction": values[0] / values.sum(),
    })
    return scores, table


def correlation_row(label: str, x: pd.Series, y: pd.Series):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 4:
        return {"comparison": label, "n": len(d), "pearson_r": np.nan,
                "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(d.x, d.y)
    sr, sp = stats.spearmanr(d.x, d.y)
    return {"comparison": label, "n": len(d), "pearson_r": pr,
            "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp}


def variance_components(z: pd.DataFrame, groups: pd.Series):
    data = z.copy()
    data["_group"] = groups
    data = data.dropna()
    cols = list(z.columns)
    group_data = [(g, d[cols].values) for g, d in data.groupby("_group") if len(d) >= 2]
    if len(group_data) < 2:
        raise ValueError("Need at least two islands with >=2 complete plants")
    n_i = np.array([len(a) for _, a in group_data], float)
    arrays = [a for _, a in group_data]
    n = int(n_i.sum())
    k = len(arrays)
    grand = np.vstack(arrays).mean(axis=0)
    ssw = np.zeros((len(cols), len(cols)))
    ssb = np.zeros_like(ssw)
    for ni, arr in zip(n_i, arrays):
        mean = arr.mean(axis=0)
        centered = arr - mean
        ssw += centered.T @ centered
        delta = (mean - grand)[:, None]
        ssb += ni * (delta @ delta.T)
    msw = ssw / (n - k)
    msb = ssb / (k - 1)
    n0 = (n - (n_i @ n_i) / n) / (k - 1)
    b_raw = (msb - msw) / n0
    b_raw = (b_raw + b_raw.T) / 2
    evals, evecs = np.linalg.eigh(b_raw)
    b_psd = evecs @ np.diag(np.clip(evals, 0, None)) @ evecs.T
    b_psd = (b_psd + b_psd.T) / 2
    return b_raw, b_psd, msw, n, k


def directional_pst(b_psd: np.ndarray, w: np.ndarray, trait_names: list[str]):
    ridge = max(float(np.trace(w)) / len(trait_names), 1.0) * 1e-8
    vals, vecs = linalg.eigh(b_psd, w + np.eye(len(trait_names)) * ridge)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rows = []
    for j, lam in enumerate(vals):
        v = vecs[:, j].copy()
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
        pst = lam / (lam + 2.0) if lam >= 0 else np.nan
        row = {"axis": j + 1, "lambda_B_over_W": lam, "pst_analogue": pst}
        row.update({f"loading_{name}": value for name, value in zip(trait_names, v)})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_top_axis(df: pd.DataFrame, n_boot: int = 1000, seed: int = 20260811):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sampled = []
        for island, d in df.groupby("island"):
            idx = rng.integers(0, len(d), len(d))
            sampled.append(d.iloc[idx].assign(island=island))
        boot = pd.concat(sampled, ignore_index=True)
        complete = boot[["island"] + CORE_TRAITS].dropna()
        if complete.island.nunique() < 2:
            continue
        z = zscore(complete[CORE_TRAITS])
        try:
            _br, bp, w, _n, _k = variance_components(z, complete.island)
            top = directional_pst(bp, w, CORE_TRAITS).iloc[0]
            out.append(float(top.pst_analogue))
        except Exception:
            continue
    if not out:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi), len(out)


def main() -> None:
    df = pd.read_csv(RESULTS / "plant_means.csv", encoding="utf-8-sig")
    df = df[df.island.isin(ISLAND_ORDER)].copy()

    df["guide_asin"] = np.arcsin(np.sqrt(np.clip(df[GUIDE] / 100.0, 0, 1)))
    df["guide_z"] = (df.guide_asin - df.guide_asin.mean()) / df.guide_asin.std(ddof=1)

    df["size_pc1"], size_load = pc1_scores(df, SIZE_TRAITS, "size")
    df["shape_pc1"], shape_load = pc1_scores(df, SHAPE_TRAITS, "shape")
    pd.concat([size_load, shape_load], ignore_index=True).to_csv(
        RESULTS / "multivariate_module_loadings.csv", index=False
    )

    df["size_within"] = df.size_pc1 - df.groupby("island").size_pc1.transform("mean")
    df["guide_within"] = df.guide_z - df.groupby("island").guide_z.transform("mean")
    island_means = df.groupby("island", observed=True)[["size_pc1", "guide_z", "shape_pc1"]].mean()

    corrs = [
        correlation_row("raw plant-level sizePC1 vs guide", df.size_pc1, df.guide_z),
        correlation_row("within-island centered sizePC1 vs guide", df.size_within, df.guide_within),
        correlation_row("between-island mean sizePC1 vs guide", island_means.size_pc1, island_means.guide_z),
        correlation_row("between-island mean shapePC1 vs guide", island_means.shape_pc1, island_means.guide_z),
    ]
    pd.DataFrame(corrs).to_csv(RESULTS / "multivariate_size_guide_correlations.csv", index=False)

    model_data = df[["guide_z", "size_pc1", "island", "no"]].dropna().copy()
    model_data["site"] = model_data.island.astype(str) + "_" + model_data.no.astype(str)
    ols = smf.ols("guide_z ~ size_pc1 + C(island)", data=model_data).fit(cov_type="HC3")
    interaction = smf.ols("guide_z ~ size_pc1 * C(island)", data=model_data).fit(cov_type="HC3")
    model_table = pd.DataFrame({
        "model": ["island-adjusted common slope", "island-specific slopes"],
        "n": [int(ols.nobs), int(interaction.nobs)],
        "r2": [ols.rsquared, interaction.rsquared],
        "aic": [ols.aic, interaction.aic],
        "size_slope": [ols.params.get("size_pc1", np.nan), interaction.params.get("size_pc1", np.nan)],
        "size_p": [ols.pvalues.get("size_pc1", np.nan), interaction.pvalues.get("size_pc1", np.nan)],
    })
    model_table.to_csv(RESULTS / "multivariate_size_guide_models.csv", index=False)

    complete = df[["island"] + CORE_TRAITS].dropna().copy()
    z = zscore(complete[CORE_TRAITS])
    b_raw, b_psd, w, n, k = variance_components(z, complete.island)
    directional = directional_pst(b_psd, w, CORE_TRAITS)
    lo, hi, n_boot = bootstrap_top_axis(df)
    directional["top_axis_boot_lo"] = np.nan
    directional["top_axis_boot_hi"] = np.nan
    directional["top_axis_boot_n"] = n_boot
    directional.loc[directional.axis == 1, "top_axis_boot_lo"] = lo
    directional.loc[directional.axis == 1, "top_axis_boot_hi"] = hi
    directional.to_csv(RESULTS / "multivariate_pst_axes.csv", index=False)

    pd.DataFrame(b_raw, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(
        RESULTS / "multivariate_B_raw.csv"
    )
    pd.DataFrame(b_psd, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(
        RESULTS / "multivariate_B_psd.csv"
    )
    pd.DataFrame(w, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(
        RESULTS / "multivariate_W.csv"
    )
    island_means.to_csv(RESULTS / "multivariate_island_module_means.csv")

    print("\n=== multivariate phenotypic divergence ===")
    print(f"plant means: {len(df)}; complete core-trait plants: {n}; islands: {k}")
    print(f"size PC1 variance explained: {size_load.pc1_variance_fraction.iloc[0]:.3f}")
    print(f"shape PC1 variance explained: {shape_load.pc1_variance_fraction.iloc[0]:.3f}")
    for row in corrs:
        print(
            f"{row['comparison']}: n={row['n']} "
            f"r={row['pearson_r']:+.3f} p={row['pearson_p']:.3g}; "
            f"rho={row['spearman_rho']:+.3f} p={row['spearman_p']:.3g}"
        )
    print(
        "island-adjusted sizePC1 slope on guide: "
        f"beta={ols.params.get('size_pc1', np.nan):+.3f}, "
        f"p={ols.pvalues.get('size_pc1', np.nan):.3g}"
    )
    top = directional.iloc[0]
    print(
        f"top multivariate P_ST analogue={top.pst_analogue:.3f} "
        f"(bootstrap 95% CI {lo:.3f}-{hi:.3f}; {n_boot} resamples)"
    )
    print("top-axis loadings:")
    for trait in CORE_TRAITS:
        print(f"  {trait:30s} {top[f'loading_{trait}']:+.3f}")


if __name__ == "__main__":
    main()
