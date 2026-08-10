#!/usr/bin/env python3
"""Exploratory multivariate phenotypic divergence across Izu islands.

This is deliberately a phenotypic P_ST analogue, not a Q_ST-F_ST test.
The key null is simple floral miniaturisation: corolla body size is defined
without mouth, throat, reproductive-organ or guide traits, and those functional
traits are then tested for island divergence beyond body-size allometry.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, stats
import statsmodels.formula.api as smf

RESULTS = Path("results_shimask_all")
ISLAND_ORDER = ["oshima", "toshima", "niijima", "shikine", "kozu"]
BODY_SIZE_TRAITS = ["corolla_length_mm", "corolla_width_fulleq_mm"]
SHAPE_TRAITS = ["corolla_aspect_L_W", "tube_flare_W_throat", "organ_corolla_ratio"]
GUIDE = "guide_coverage_pct"
FUNCTIONAL_TRAITS = [
    "mouth_width_mm", "throat_width_mm", "organ_length_mm", GUIDE,
    "organ_corolla_ratio", "corolla_aspect_L_W", "tube_flare_W_throat",
    "lobe_incision_mm",
]
CORE_TRAITS = BODY_SIZE_TRAITS + ["mouth_width_mm", "throat_width_mm", "organ_length_mm", GUIDE]


def zscore(frame):
    return (frame - frame.mean()) / frame.std(ddof=1)


def pc1_scores(frame, columns, prefix):
    clean = frame[columns].dropna()
    z = zscore(clean)
    values, vectors = np.linalg.eigh(np.cov(z.values, rowvar=False, ddof=1))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    loading = vectors[:, 0].copy()
    if np.nanmean(loading) < 0:
        loading *= -1
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    scores.loc[clean.index] = z.values @ loading
    return scores, pd.DataFrame({
        "module": prefix, "trait": columns, "pc1_loading": loading,
        "pc1_variance_fraction": values[0] / values.sum(),
    })


def correlation_row(label, x, y):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 4:
        return {"comparison": label, "n": len(d), "pearson_r": np.nan,
                "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(d.x, d.y)
    sr, sp = stats.spearmanr(d.x, d.y)
    return {"comparison": label, "n": len(d), "pearson_r": pr,
            "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp}


def variance_components(z, groups):
    data = z.copy(); data["_group"] = groups; data = data.dropna()
    cols = list(z.columns)
    group_data = [(g, d[cols].values) for g, d in data.groupby("_group") if len(d) >= 2]
    if len(group_data) < 2:
        raise ValueError("Need at least two islands with >=2 complete plants")
    n_i = np.array([len(a) for _, a in group_data], float)
    arrays = [a for _, a in group_data]; n = int(n_i.sum()); k = len(arrays)
    grand = np.vstack(arrays).mean(axis=0)
    ssw = np.zeros((len(cols), len(cols))); ssb = np.zeros_like(ssw)
    for ni, arr in zip(n_i, arrays):
        mean = arr.mean(axis=0); centered = arr - mean
        ssw += centered.T @ centered
        delta = (mean - grand)[:, None]; ssb += ni * (delta @ delta.T)
    msw = ssw / (n - k); msb = ssb / (k - 1)
    n0 = (n - (n_i @ n_i) / n) / (k - 1)
    b_raw = (msb - msw) / n0; b_raw = (b_raw + b_raw.T) / 2
    evals, evecs = np.linalg.eigh(b_raw)
    b_psd = evecs @ np.diag(np.clip(evals, 0, None)) @ evecs.T
    return b_raw, (b_psd + b_psd.T) / 2, msw, n, k


def directional_pst(b_psd, w, trait_names):
    ridge = max(float(np.trace(w)) / len(trait_names), 1.0) * 1e-8
    vals, vecs = linalg.eigh(b_psd, w + np.eye(len(trait_names)) * ridge)
    order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    rows = []
    for j, lam in enumerate(vals):
        v = vecs[:, j].copy()
        if v[np.argmax(np.abs(v))] < 0: v *= -1
        # Euclidean-normalised coefficients are easier to compare across traits.
        norm = np.linalg.norm(v)
        if norm > 0: v /= norm
        row = {"axis": j + 1, "lambda_B_over_W": lam,
               "pst_analogue": lam / (lam + 2.0) if lam >= 0 else np.nan}
        row.update({f"loading_{name}": value for name, value in zip(trait_names, v)})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_top_axis(df, n_boot=1000, seed=20260811):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        sampled = []
        for island, d in df.groupby("island"):
            sampled.append(d.iloc[rng.integers(0, len(d), len(d))].assign(island=island))
        boot = pd.concat(sampled, ignore_index=True)
        complete = boot[["island"] + CORE_TRAITS].dropna()
        if complete.island.nunique() < 2: continue
        try:
            _br, bp, w, _n, _k = variance_components(zscore(complete[CORE_TRAITS]), complete.island)
            out.append(float(directional_pst(bp, w, CORE_TRAITS).iloc[0].pst_analogue))
        except Exception:
            continue
    if not out: return np.nan, np.nan, 0
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi), len(out)


def allometry_models(df):
    """Test functional traits for island divergence beyond corolla body size."""
    rows = []
    for target in FUNCTIONAL_TRAITS:
        d = df[[target, "size_pc1", "island", "site"]].dropna().copy()
        d["y"] = (np.arcsin(np.sqrt(np.clip(d[target] / 100.0, 0, 1)))
                  if target == GUIDE else d[target])
        size_only = smf.ols("y ~ size_pc1", d).fit(cov_type="HC3")
        # Cluster-robust SEs protect against plants sharing a sampling site.
        island_add = smf.ols("y ~ size_pc1 + C(island)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["site"]})
        interaction = smf.ols("y ~ size_pc1 * C(island)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["site"]})
        levels = sorted(d.island.unique())
        restrictions = [f"C(island)[T.{x}] = 0" for x in levels if x != levels[0]]
        try:
            island_p = float(island_add.f_test(restrictions).pvalue)
        except Exception:
            island_p = np.nan
        rows.append({
            "trait": target, "n": len(d), "n_sites": d.site.nunique(),
            "size_only_r2": size_only.rsquared, "size_island_r2": island_add.rsquared,
            "size_island_delta_r2": island_add.rsquared - size_only.rsquared,
            "common_size_slope": island_add.params.get("size_pc1", np.nan),
            "common_size_p_cluster": island_add.pvalues.get("size_pc1", np.nan),
            "island_joint_p_given_size_cluster": island_p,
            "aic_size_only": size_only.aic, "aic_size_island": island_add.aic,
            "aic_interaction": interaction.aic,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RESULTS / "plant_means.csv", encoding="utf-8-sig")
    df = df[df.island.isin(ISLAND_ORDER)].copy()
    df["site"] = df.island.astype(str) + "_" + df.no.astype(str)
    df["guide_asin"] = np.arcsin(np.sqrt(np.clip(df[GUIDE] / 100.0, 0, 1)))
    df["guide_z"] = (df.guide_asin - df.guide_asin.mean()) / df.guide_asin.std(ddof=1)
    df["size_pc1"], size_load = pc1_scores(df, BODY_SIZE_TRAITS, "corolla_body_size")
    df["shape_pc1"], shape_load = pc1_scores(df, SHAPE_TRAITS, "shape")
    pd.concat([size_load, shape_load], ignore_index=True).to_csv(
        RESULTS / "multivariate_module_loadings.csv", index=False)

    df["size_within"] = df.size_pc1 - df.groupby("island").size_pc1.transform("mean")
    df["guide_within"] = df.guide_z - df.groupby("island").guide_z.transform("mean")
    island_means = df.groupby("island", observed=True)[["size_pc1", "guide_z", "shape_pc1"]].mean()
    corrs = [
        correlation_row("raw plant-level bodySizePC1 vs guide", df.size_pc1, df.guide_z),
        correlation_row("within-island centered bodySizePC1 vs guide", df.size_within, df.guide_within),
        correlation_row("between-island mean bodySizePC1 vs guide", island_means.size_pc1, island_means.guide_z),
        correlation_row("between-island mean shapePC1 vs guide", island_means.shape_pc1, island_means.guide_z),
    ]
    pd.DataFrame(corrs).to_csv(RESULTS / "multivariate_size_guide_correlations.csv", index=False)

    allom = allometry_models(df)
    allom.to_csv(RESULTS / "multivariate_allometry_tests.csv", index=False)

    complete = df[["island"] + CORE_TRAITS].dropna()
    b_raw, b_psd, w, n, k = variance_components(zscore(complete[CORE_TRAITS]), complete.island)
    directional = directional_pst(b_psd, w, CORE_TRAITS)
    lo, hi, n_boot = bootstrap_top_axis(df)
    directional["top_axis_boot_lo"] = np.nan; directional["top_axis_boot_hi"] = np.nan
    directional["top_axis_boot_n"] = n_boot
    directional.loc[directional.axis == 1, "top_axis_boot_lo"] = lo
    directional.loc[directional.axis == 1, "top_axis_boot_hi"] = hi
    directional.to_csv(RESULTS / "multivariate_pst_axes.csv", index=False)
    pd.DataFrame(b_raw, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(RESULTS / "multivariate_B_raw.csv")
    pd.DataFrame(b_psd, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(RESULTS / "multivariate_B_psd.csv")
    pd.DataFrame(w, index=CORE_TRAITS, columns=CORE_TRAITS).to_csv(RESULTS / "multivariate_W.csv")
    island_means.to_csv(RESULTS / "multivariate_island_module_means.csv")

    print("\n=== multivariate phenotypic divergence: body-size null ===")
    print(f"plant means: {len(df)}; complete core-trait plants: {n}; islands: {k}")
    print(f"corolla body-size PC1 variance explained: {size_load.pc1_variance_fraction.iloc[0]:.3f}")
    for row in corrs:
        print(f"{row['comparison']}: n={row['n']} r={row['pearson_r']:+.3f} p={row['pearson_p']:.3g}; rho={row['spearman_rho']:+.3f} p={row['spearman_p']:.3g}")
    print("functional traits: island effect after body-size adjustment (site-cluster robust)")
    for _, r in allom.iterrows():
        print(f"  {r.trait:28s} deltaR2={r.size_island_delta_r2:.3f} island p={r.island_joint_p_given_size_cluster:.3g}")
    top = directional.iloc[0]
    print(f"top multivariate P_ST analogue={top.pst_analogue:.3f} (bootstrap 95% CI {lo:.3f}-{hi:.3f}; {n_boot} resamples)")
    print("top-axis unit-norm loadings:")
    for trait in CORE_TRAITS:
        print(f"  {trait:30s} {top[f'loading_{trait}']:+.3f}")


if __name__ == "__main__":
    main()
