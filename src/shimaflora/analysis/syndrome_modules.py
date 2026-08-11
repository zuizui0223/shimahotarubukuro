#!/usr/bin/env python3
"""Test whether floral syndrome formation is modular rather than a single package.

Cross-sectional island data cannot establish temporal asynchrony. This analysis
therefore tests the directly estimable prediction: attraction, mechanical-fit
and reproductive modules should show heterogeneous/non-parallel island
trajectories if a pollination syndrome is assembled modularly.

Corolla length and width define the body-size null. Functional traits are
residualised against log body size before module scores are constructed.
Sampling sites are the primary replication unit for island-level inference.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
GUIDE = "guide_coverage_pct"
MODULES = {
    "attraction": [GUIDE],
    "mechanical_fit": ["mouth_width_mm", "throat_width_mm", "tube_flare_W_throat"],
    "reproductive": ["organ_length_mm", "organ_corolla_ratio"],
}
LOG_TRAITS = {"mouth_width_mm", "throat_width_mm", "organ_length_mm"}


def zscore(x):
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def transformed(df, trait):
    if trait == GUIDE:
        return np.arcsin(np.sqrt(np.clip(df[trait] / 100.0, 0, 1)))
    if trait in LOG_TRAITS:
        return np.log(df[trait].clip(lower=np.finfo(float).tiny))
    return df[trait].astype(float)


def residualise(df, trait):
    d = df[[trait, "log_body_size"]].dropna()
    y = transformed(d, trait).to_numpy(float)
    x = d.log_body_size.to_numpy(float)
    X = np.column_stack([np.ones(len(d)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[d.index] = y - X @ beta
    return out, float(beta[1])


def module_score(df, module, traits):
    residual_cols = []
    slope_rows = []
    for trait in traits:
        col = f"_resid_{trait}"
        df[col], slope = residualise(df, trait)
        residual_cols.append(col)
        slope_rows.append({"module": module, "trait": trait, "body_size_slope": slope})

    clean = df[residual_cols].dropna()
    z = clean.apply(zscore)
    if len(traits) == 1:
        load = np.array([1.0])
        score_values = z.iloc[:, 0].to_numpy(float)
        variance_fraction = 1.0
    else:
        cov = np.cov(z.to_numpy(float), rowvar=False, ddof=1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        load = vectors[:, 0].copy()
        if np.mean(load) < 0:
            load *= -1
        score_values = z.to_numpy(float) @ load
        variance_fraction = float(values[0] / values.sum())
    score_values = (score_values - score_values.mean()) / score_values.std(ddof=1)
    score = pd.Series(np.nan, index=df.index, dtype=float)
    score.loc[clean.index] = score_values
    loadings = pd.DataFrame({
        "module": module,
        "trait": traits,
        "pc1_loading": load,
        "pc1_variance_fraction": variance_fraction,
    }).merge(pd.DataFrame(slope_rows), on=["module", "trait"], how="left")
    return score, loadings


def pst_value(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) >= 1]
    if len(groups) < 2:
        return np.nan
    all_values = np.concatenate(groups)
    n, k = len(all_values), len(groups)
    if n <= k:
        return np.nan
    sizes = np.array([len(g) for g in groups], float)
    grand = all_values.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msw = ssw / (n - k)
    n0 = (n - (sizes @ sizes) / n) / (k - 1)
    vb = max((ssb / (k - 1) - msw) / n0, 0.0)
    return vb / (vb + 2 * msw) if (vb + msw) > 0 else 0.0


def profile_interaction_test(site_wide, nperm=9999, seed=20260811):
    """Repeated-profile test: do relative module scores depend on island?"""
    complete = site_wide.dropna(subset=list(MODULES)).copy()
    long = complete.melt(
        id_vars=["island", "site"], value_vars=list(MODULES),
        var_name="module", value_name="score"
    )
    sites = sorted(long.site.unique())
    modules = list(MODULES)
    islands = sorted(long.island.unique())
    site_dummy = np.column_stack([(long.site == x).to_numpy(float) for x in sites[1:]])
    module_dummy = np.column_stack([(long.module == x).to_numpy(float) for x in modules[1:]])
    X0 = np.column_stack([np.ones(len(long)), site_dummy, module_dummy])

    def full_matrix(labels_by_site):
        row_island = long.site.map(labels_by_site)
        island_dummy = np.column_stack([
            (row_island == x).to_numpy(float) for x in islands[1:]
        ])
        interactions = np.column_stack([
            module_dummy[:, m] * island_dummy[:, g]
            for m in range(module_dummy.shape[1])
            for g in range(island_dummy.shape[1])
        ])
        return np.column_stack([X0, interactions])

    observed_labels = complete.set_index("site").island.to_dict()
    y = long.score.to_numpy(float)
    X1 = full_matrix(observed_labels)
    H0 = X0 @ np.linalg.pinv(X0)
    H1 = X1 @ np.linalg.pinv(X1)
    M0 = np.eye(len(y)) - H0
    M1 = np.eye(len(y)) - H1
    s0 = float(y @ M0 @ y)
    s1 = float(y @ M1 @ y)
    dfn = np.linalg.matrix_rank(X1) - np.linalg.matrix_rank(X0)
    dfd = len(y) - np.linalg.matrix_rank(X1)
    F = ((s0 - s1) / dfn) / (s1 / dfd)
    rng = np.random.default_rng(seed)
    site_labels = complete[["site", "island"]].drop_duplicates().sort_values("site")
    original = site_labels.island.to_numpy()
    exceed = 0
    for _ in range(nperm):
        permuted = rng.permutation(original)
        mapping = dict(zip(site_labels.site, permuted))
        Xp = full_matrix(mapping)
        Hp = Xp @ np.linalg.pinv(Xp)
        Mp = np.eye(len(y)) - Hp
        sp = float(y @ Mp @ y)
        Fp = ((s0 - sp) / dfn) / (sp / dfd)
        exceed += Fp >= F - 1e-12
    return {
        "n_islands": complete.island.nunique(),
        "n_sites": len(complete),
        "n_modules": len(modules),
        "module_by_island_f": F,
        "module_by_island_df_num": dfn,
        "module_by_island_df_den": dfd,
        "module_by_island_parametric_p": float(stats.f.sf(max(F, 0), dfn, dfd)),
        "module_by_island_permutation_p": (exceed + 1) / (nperm + 1),
        "module_by_island_partial_r2": (s0 - s1) / s0 if s0 > 0 else np.nan,
        "n_permutations": nperm,
    }


def island_matrix(site_wide):
    means = site_wide.groupby("island", observed=True)[list(MODULES)].mean()
    means = means.reindex([x for x in ISLANDS if x in means.index])
    z = means.apply(zscore)
    return means, z


def rank1_metrics(z):
    arr = z.to_numpy(float)
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    total = float(np.sum(s * s))
    fraction = float((s[0] ** 2) / total) if total > 0 else np.nan
    fitted = np.outer(u[:, 0] * s[0], vt[0])
    residual = arr - fitted
    return fraction, fitted, residual


def pairwise_trajectory_rows(z):
    rows = []
    for a, b in combinations(z.columns, 2):
        r, p = stats.pearsonr(z[a], z[b])
        rho, sp = stats.spearmanr(z[a], z[b])
        rows.append({
            "module_a": a,
            "module_b": b,
            "pearson_r_across_islands": r,
            "pearson_p_descriptive": p,
            "spearman_rho_across_islands": rho,
            "spearman_p_descriptive": sp,
            "trajectory_angle_deg": float(np.degrees(np.arccos(np.clip(r, -1, 1)))),
        })
    return rows


def bootstrap(site_wide, nboot=2000, seed=20260812):
    rng = np.random.default_rng(seed)
    rank1_values = []
    coherence_values = []
    pst_values = {m: [] for m in MODULES}
    pair_values = {(a, b): [] for a, b in combinations(MODULES, 2)}
    for _ in range(nboot):
        pieces = []
        for _, di in site_wide.groupby("island", observed=True):
            idx = rng.integers(0, len(di), len(di))
            pieces.append(di.iloc[idx].copy())
        boot = pd.concat(pieces, ignore_index=True)
        _, z = island_matrix(boot)
        try:
            fraction, _, _ = rank1_metrics(z)
            rank1_values.append(fraction)
            rs = []
            for a, b in pair_values:
                r = float(stats.pearsonr(z[a], z[b]).statistic)
                pair_values[(a, b)].append(r)
                rs.append(r)
            coherence_values.append(float(np.mean(rs)))
            for m in MODULES:
                groups = [g[m].dropna().to_numpy(float) for _, g in boot.groupby("island")]
                pst_values[m].append(float(pst_value(groups)))
        except Exception:
            continue
    return rank1_values, coherence_values, pst_values, pair_values


def response_modes():
    """Classify trait responses as level shifts, slope changes, or unresolved scaling."""
    path = R / "multivariate_allometry_tests.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    tests = pd.read_csv(path)
    trait_module = {
        GUIDE: "attraction",
        "mouth_width_mm": "mechanical_fit",
        "throat_width_mm": "mechanical_fit",
        "tube_flare_W_throat": "mechanical_fit",
        "organ_length_mm": "reproductive",
        "organ_corolla_ratio": "reproductive",
    }
    out = tests[tests.trait.isin(trait_module)].copy()
    out["module"] = out.trait.map(trait_module)

    def classify(row):
        if row.site_interaction_permutation_p_bh < 0.05:
            return "allometric_reconfiguration"
        if row.site_ancova_permutation_p_bh < 0.05:
            return "island_level_shift"
        return "simple_scaling_or_unresolved"

    out["response_mode"] = out.apply(classify, axis=1)
    keep = [
        "module", "trait", "response_mode",
        "site_ancova_partial_r2", "site_ancova_permutation_p",
        "site_ancova_permutation_p_bh",
        "site_interaction_partial_r2", "site_interaction_permutation_p",
        "site_interaction_permutation_p_bh",
        "common_support_site_residual_permutation_p",
        "common_support_site_residual_permutation_p_bh",
    ]
    out = out[keep].sort_values(["module", "trait"]).reset_index(drop=True)
    rows = []
    for module, d in out.groupby("module", observed=True):
        counts = d.response_mode.value_counts()
        rows.append({
            "module": module,
            "n_traits": len(d),
            "n_island_level_shift": int(counts.get("island_level_shift", 0)),
            "n_allometric_reconfiguration": int(counts.get("allometric_reconfiguration", 0)),
            "n_simple_scaling_or_unresolved": int(counts.get("simple_scaling_or_unresolved", 0)),
            "response_modes_present": ";".join(sorted(d.response_mode.unique())),
        })
    return out, pd.DataFrame(rows)


def q025975(values):
    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def main():
    df = pd.read_csv(R / "plant_means.csv", encoding="utf-8-sig")
    df = df[df.island.isin(ISLANDS)].copy()
    df["site"] = df.island.astype(str) + "_" + df.no.astype(str)
    df["body_size_mm"] = np.sqrt(df.corolla_length_mm * df.corolla_width_fulleq_mm)
    df["log_body_size"] = np.log(df.body_size_mm)

    loading_tables = []
    for module, traits in MODULES.items():
        df[module], loadings = module_score(df, module, traits)
        loading_tables.append(loadings)
    pd.concat(loading_tables, ignore_index=True).to_csv(
        R / "syndrome_module_loadings.csv", index=False
    )

    plant_cols = ["island", "site", "no", "id", "body_size_mm"] + list(MODULES)
    df[plant_cols].to_csv(R / "syndrome_module_plant_scores.csv", index=False)

    aggregations = {m: (m, "mean") for m in MODULES}
    site = df.groupby(["island", "site"], observed=True).agg(
        n_plants=("id", "size"), **aggregations
    ).reset_index()
    site.to_csv(R / "syndrome_module_site_scores.csv", index=False)

    means, z = island_matrix(site)
    rank1_fraction, fitted, residual = rank1_metrics(z)
    island_rows = []
    for i, island in enumerate(z.index):
        row = {
            "island": island,
            "syndrome_axis_score": np.nan,
            "orthogonal_residual_norm": float(np.linalg.norm(residual[i])),
        }
        for j, module in enumerate(z.columns):
            row[f"mean_{module}"] = float(means.loc[island, module])
            row[f"z_{module}"] = float(z.loc[island, module])
            row[f"rank1_fitted_{module}"] = float(fitted[i, j])
            row[f"rank1_residual_{module}"] = float(residual[i, j])
        island_rows.append(row)
    arr = z.to_numpy(float)
    u, s, _ = np.linalg.svd(arr, full_matrices=False)
    axis_score = u[:, 0] * s[0]
    if np.corrcoef(axis_score, z.mean(axis=1))[0, 1] < 0:
        axis_score *= -1
    for row, score in zip(island_rows, axis_score):
        row["syndrome_axis_score"] = float(score)
    pd.DataFrame(island_rows).to_csv(
        R / "syndrome_module_island_trajectories.csv", index=False
    )

    interaction = profile_interaction_test(site)
    pairs = pairwise_trajectory_rows(z)
    rank_boot, coherence_boot, pst_boot, pair_boot = bootstrap(site)
    rank_lo, rank_hi = q025975(rank_boot)
    het_lo, het_hi = (
        (1 - rank_hi, 1 - rank_lo) if np.isfinite(rank_lo) else (np.nan, np.nan)
    )
    observed_rs = [row["pearson_r_across_islands"] for row in pairs]
    coherence = float(np.mean(observed_rs))
    coherence_lo, coherence_hi = q025975(coherence_boot)
    summary = {
        **interaction,
        "rank1_shared_trajectory_fraction": rank1_fraction,
        "rank1_shared_trajectory_boot_lo": rank_lo,
        "rank1_shared_trajectory_boot_hi": rank_hi,
        "module_heterogeneity_fraction": 1 - rank1_fraction,
        "module_heterogeneity_boot_lo": het_lo,
        "module_heterogeneity_boot_hi": het_hi,
        "mean_pairwise_module_trajectory_r": coherence,
        "mean_pairwise_module_trajectory_r_boot_lo": coherence_lo,
        "mean_pairwise_module_trajectory_r_boot_hi": coherence_hi,
        "n_bootstrap": len(rank_boot),
        "interpretation": (
            "Significant module-by-island interaction supports heterogeneous/non-parallel "
            "module divergence. Cross-sectional data do not by themselves establish temporal asynchrony."
        ),
    }
    pd.DataFrame([summary]).to_csv(
        R / "syndrome_module_heterogeneity.csv", index=False
    )

    for row in pairs:
        vals = pair_boot[(row["module_a"], row["module_b"])]
        lo, hi = q025975(vals)
        row["pearson_r_boot_lo"] = lo
        row["pearson_r_boot_hi"] = hi
    pd.DataFrame(pairs).to_csv(
        R / "syndrome_module_pairwise_trajectories.csv", index=False
    )

    pst_rows = []
    for module in MODULES:
        groups = [g[module].dropna().to_numpy(float) for _, g in site.groupby("island")]
        observed = float(pst_value(groups))
        lo, hi = q025975(pst_boot[module])
        pst_rows.append({
            "module": module,
            "site_level_pst_analogue": observed,
            "bootstrap_lo": lo,
            "bootstrap_hi": hi,
            "n_bootstrap": len(pst_boot[module]),
        })
    pd.DataFrame(pst_rows).to_csv(R / "syndrome_module_pst.csv", index=False)

    modes, mode_summary = response_modes()
    if len(modes):
        modes.to_csv(R / "syndrome_module_response_modes.csv", index=False)
        mode_summary.to_csv(R / "syndrome_module_response_summary.csv", index=False)

    print("\n=== pollination-syndrome module heterogeneity ===")
    print("modules:", ", ".join(MODULES))
    print(
        f"module x island profile interaction: F={interaction['module_by_island_f']:.3f}, "
        f"permutation p={interaction['module_by_island_permutation_p']:.4g}, "
        f"partial R2={interaction['module_by_island_partial_r2']:.3f}"
    )
    print(
        f"shared rank-1 trajectory fraction={rank1_fraction:.3f} "
        f"(site-bootstrap 95% CI {rank_lo:.3f}-{rank_hi:.3f}); "
        f"heterogeneity={1-rank1_fraction:.3f}"
    )
    print(f"mean pairwise module trajectory r={coherence:+.3f}")
    for row in pst_rows:
        print(
            f"  {row['module']:16s} site-P_ST analogue={row['site_level_pst_analogue']:.3f} "
            f"[{row['bootstrap_lo']:.3f},{row['bootstrap_hi']:.3f}]"
        )
    if len(modes):
        print("trait-level response modes:")
        for _, row in modes.iterrows():
            print(f"  {row['module']:16s} {row['trait']:24s} {row['response_mode']}")
    print(
        "Interpretation: this tests heterogeneous/non-parallel module assembly; "
        "temporal asynchrony requires historical/phylogenetic ordering."
    )


if __name__ == "__main__":
    main()
