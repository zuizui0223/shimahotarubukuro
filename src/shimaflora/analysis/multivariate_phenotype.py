#!/usr/bin/env python3
"""Multivariate phenotypic divergence under a simple floral-miniaturisation null.

This is an exploratory phenotypic P_ST analogue, not a Q_ST-F_ST test. Corolla
length and width define body size; guide, mechanical-fit and reproductive traits
are tested for island divergence beyond log body-size allometry. Site-level
permutation tests are the primary inference because plants share sampling sites.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, stats
import statsmodels.formula.api as smf

R = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
BODY = ["corolla_length_mm", "corolla_width_fulleq_mm"]
SHAPE = ["corolla_aspect_L_W", "tube_flare_W_throat", "organ_corolla_ratio"]
GUIDE = "guide_coverage_pct"
FUNCTIONAL = ["mouth_width_mm", "throat_width_mm", "organ_length_mm", GUIDE,
              "organ_corolla_ratio", "corolla_aspect_L_W",
              "tube_flare_W_throat", "lobe_incision_mm"]
LOG_TRAITS = {"mouth_width_mm", "throat_width_mm", "organ_length_mm",
              "lobe_incision_mm"}
ABS_TRAITS = BODY + ["mouth_width_mm", "throat_width_mm", "organ_length_mm", GUIDE]
RES_TRAITS = ["mouth_width_mm", "throat_width_mm", "organ_length_mm", GUIDE]


def zscore(x):
    return (x - x.mean()) / x.std(ddof=1)


def transformed(df, trait):
    if trait == GUIDE:
        return np.arcsin(np.sqrt(np.clip(df[trait] / 100.0, 0, 1)))
    if trait in LOG_TRAITS:
        return np.log(df[trait].clip(lower=np.finfo(float).tiny))
    return df[trait]


def pc1(df, cols, module):
    clean = df[cols].dropna(); z = zscore(clean)
    values, vectors = np.linalg.eigh(np.cov(z.values, rowvar=False, ddof=1))
    order = np.argsort(values)[::-1]; values = values[order]; vectors = vectors[:, order]
    load = vectors[:, 0].copy()
    if np.mean(load) < 0: load *= -1
    score = pd.Series(np.nan, index=df.index, dtype=float)
    score.loc[clean.index] = z.values @ load
    table = pd.DataFrame({"module": module, "trait": cols, "pc1_loading": load,
                          "pc1_variance_fraction": values[0] / values.sum()})
    return score, table


def corr(label, x, y, omitted=""):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    row = {"comparison": label, "omitted_island": omitted, "n": len(d)}
    if len(d) < 4:
        return {**row, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(d.x, d.y); sr, sp = stats.spearmanr(d.x, d.y)
    return {**row, "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp}


def variance_components(z, groups):
    d = z.copy(); d["_g"] = groups; d = d.dropna(); cols = list(z.columns)
    arrays = [x[cols].values for _, x in d.groupby("_g") if len(x) >= 2]
    if len(arrays) < 2: raise ValueError("need >=2 islands")
    ni = np.array([len(x) for x in arrays], float); n = int(ni.sum()); k = len(arrays)
    grand = np.vstack(arrays).mean(0); p = len(cols)
    ssw = np.zeros((p, p)); ssb = np.zeros((p, p))
    for nn, x in zip(ni, arrays):
        mean = x.mean(0); c = x - mean; ssw += c.T @ c
        delta = (mean - grand)[:, None]; ssb += nn * (delta @ delta.T)
    msw = ssw / (n - k); msb = ssb / (k - 1)
    n0 = (n - (ni @ ni) / n) / (k - 1)
    b = (msb - msw) / n0; b = (b + b.T) / 2
    val, vec = np.linalg.eigh(b); bpsd = vec @ np.diag(np.clip(val, 0, None)) @ vec.T
    return b, (bpsd + bpsd.T) / 2, msw, n, k


def directional_axes(b, w, names):
    ridge = max(float(np.trace(w)) / len(names), 1.0) * 1e-8
    val, vec = linalg.eigh(b, w + np.eye(len(names)) * ridge)
    order = np.argsort(val)[::-1]; rows = []
    for axis, j in enumerate(order, 1):
        lam = val[j]; v = vec[:, j].copy()
        if v[np.argmax(np.abs(v))] < 0: v *= -1
        v /= np.linalg.norm(v)
        row = {"axis": axis, "lambda_B_over_W": lam,
               "pst_analogue": lam / (lam + 2) if lam >= 0 else np.nan}
        row.update({f"loading_{n}": x for n, x in zip(names, v)}); rows.append(row)
    return pd.DataFrame(rows)


def residual(df, trait):
    d = df[[trait, "log_body_size"]].dropna(); y = transformed(d, trait).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d.log_body_size.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = pd.Series(np.nan, index=df.index, dtype=float); out.loc[d.index] = y - X @ beta
    return out


def add_residuals(df, traits):
    out = df.copy()
    for t in traits: out[f"resid_{t}"] = residual(out, t)
    return out


def hierarchical_bootstrap(df, traits, residualise, nboot=500, seed=20260811):
    rng = np.random.default_rng(seed); values = []
    for _ in range(nboot):
        pieces = []
        for island, di in df.groupby("island"):
            sites = di.site.unique()
            for j, site in enumerate(rng.choice(sites, len(sites), replace=True)):
                ds = di[di.site == site]; take = rng.integers(0, len(ds), len(ds))
                piece = ds.iloc[take].copy(); piece["site"] = f"{island}_boot_{j}"
                pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        if residualise:
            boot = add_residuals(boot, traits); cols = [f"resid_{t}" for t in traits]
        else: cols = traits
        complete = boot[["island"] + cols].dropna()
        try:
            _, b, w, _, _ = variance_components(zscore(complete[cols]), complete.island)
            values.append(float(directional_axes(b, w, cols).iloc[0].pst_analogue))
        except Exception: pass
    if not values: return np.nan, np.nan, 0
    lo, hi = np.quantile(values, [0.025, 0.975]); return float(lo), float(hi), len(values)


def eta2(groups, values):
    grand = np.mean(values); total = np.sum((values - grand) ** 2)
    if total <= 0: return 0.0
    between = sum(np.sum(groups == g) * (np.mean(values[groups == g]) - grand) ** 2
                  for g in np.unique(groups))
    return float(between / total)


def label_permutation(groups, values, observed, nperm, seed):
    rng = np.random.default_rng(seed); exceed = 0
    for _ in range(nperm):
        exceed += eta2(rng.permutation(groups), values) >= observed - 1e-12
    return (exceed + 1) / (nperm + 1)


def bh(pvalues):
    p = np.asarray([x if x == x else 1.0 for x in pvalues]); order = np.argsort(p)
    out = np.empty(len(p)); running = 1.0
    for rev, i in enumerate(order[::-1], 1):
        rank = len(p) - rev + 1; running = min(running, p[i] * len(p) / rank); out[i] = running
    return np.clip(out, 0, 1)


def site_ancova(site, nperm, seed):
    y = site.mean_y.to_numpy(float); x = site.mean_log_body_size.to_numpy(float)
    levels = sorted(site.island.unique())
    dummy = np.column_stack([(site.island == lev).to_numpy(float) for lev in levels[1:]])
    X0 = np.column_stack([np.ones(len(site)), x]); X1 = np.column_stack([X0, dummy])
    H0 = X0 @ np.linalg.pinv(X0); H1 = X1 @ np.linalg.pinv(X1)
    M0 = np.eye(len(y)) - H0; M1 = np.eye(len(y)) - H1
    s0 = float(y @ M0 @ y); s1 = float(y @ M1 @ y)
    dfn = np.linalg.matrix_rank(X1) - np.linalg.matrix_rank(X0)
    dfd = len(y) - np.linalg.matrix_rank(X1)
    F = ((s0 - s1) / dfn) / (s1 / dfd); param_p = float(stats.f.sf(max(F, 0), dfn, dfd))
    fitted = H0 @ y; resid0 = M0 @ y; rng = np.random.default_rng(seed); exceed = 0
    for _ in range(nperm):
        yp = fitted + rng.permutation(resid0); a = float(yp @ M0 @ yp); b = float(yp @ M1 @ yp)
        exceed += (((a - b) / dfn) / (b / dfd)) >= F - 1e-12
    total = float(np.sum((y - y.mean()) ** 2)); r20 = 1 - s0 / total; r21 = 1 - s1 / total
    return {"site_ancova_f": F, "site_ancova_parametric_p": param_p,
            "site_ancova_permutation_p": (exceed + 1) / (nperm + 1),
            "site_ancova_size_only_r2": r20, "site_ancova_size_island_r2": r21,
            "site_ancova_delta_r2": r21 - r20,
            "site_ancova_partial_r2": (s0 - s1) / s0 if s0 > 0 else 0.0}


def site_interaction(site, nperm, seed):
    """Test island-specific allometric slopes after fitting additive island shifts."""
    y = site.mean_y.to_numpy(float); x = site.mean_log_body_size.to_numpy(float)
    levels = sorted(site.island.unique())
    dummy = np.column_stack([(site.island == lev).to_numpy(float) for lev in levels[1:]])
    interactions = dummy * x[:, None]
    X0 = np.column_stack([np.ones(len(site)), x, dummy])
    X1 = np.column_stack([X0, interactions])
    H0 = X0 @ np.linalg.pinv(X0); H1 = X1 @ np.linalg.pinv(X1)
    M0 = np.eye(len(y)) - H0; M1 = np.eye(len(y)) - H1
    s0 = float(y @ M0 @ y); s1 = float(y @ M1 @ y)
    dfn = np.linalg.matrix_rank(X1) - np.linalg.matrix_rank(X0)
    dfd = len(y) - np.linalg.matrix_rank(X1)
    F = ((s0 - s1) / dfn) / (s1 / dfd); param_p = float(stats.f.sf(max(F, 0), dfn, dfd))
    fitted = H0 @ y; resid0 = M0 @ y; rng = np.random.default_rng(seed); exceed = 0
    for _ in range(nperm):
        yp = fitted + rng.permutation(resid0); a = float(yp @ M0 @ yp); b = float(yp @ M1 @ yp)
        exceed += (((a - b) / dfn) / (b / dfd)) >= F - 1e-12
    return {"site_interaction_f": F, "site_interaction_parametric_p": param_p,
            "site_interaction_permutation_p": (exceed + 1) / (nperm + 1),
            "site_interaction_partial_r2": (s0 - s1) / s0 if s0 > 0 else 0.0}


def allometry(df, nperm=4999, interaction_perm=1999, loo_perm=1999):
    summary = []; site_rows = []; island_rows = []; loo_rows = []
    for i, trait in enumerate(FUNCTIONAL):
        d = df[[trait, "log_body_size", "island", "site"]].dropna().copy()
        d["y"] = transformed(d, trait)
        m0 = smf.ols("y ~ log_body_size", d).fit(cov_type="HC3")
        m1 = smf.ols("y ~ log_body_size + C(island)", d).fit(cov_type="HC3")
        m2 = smf.ols("y ~ log_body_size * C(island)", d).fit(cov_type="HC3")
        d["size_adjusted_residual"] = residual(d, trait).loc[d.index]
        site = d.groupby(["island", "site"], observed=True).agg(
            n_plants=("y", "size"), mean_y=("y", "mean"),
            mean_log_body_size=("log_body_size", "mean"),
            mean_size_adjusted_residual=("size_adjusted_residual", "mean")).reset_index()
        site["trait"] = trait; site_rows.extend(site.to_dict("records"))
        e = eta2(site.island.to_numpy(), site.mean_size_adjusted_residual.to_numpy())
        sensitivity_p = label_permutation(site.island.to_numpy(),
                                          site.mean_size_adjusted_residual.to_numpy(),
                                          e, nperm, 20260811 + i)
        anc = site_ancova(site, nperm, 20261811 + i)
        interaction = site_interaction(site, interaction_perm, 20262811 + i)
        for omitted in sorted(site.island.unique()):
            reduced = site[site.island != omitted].copy()
            loo = site_ancova(reduced, loo_perm, 20263811 + i * 10 + ISLANDS.index(omitted))
            loo_rows.append({"trait": trait, "omitted_island": omitted,
                             "n_sites": len(reduced), "n_islands": reduced.island.nunique(),
                             **loo, "n_permutations": loo_perm})
        for island, di in site.groupby("island", observed=True):
            island_rows.append({"trait": trait, "island": island, "n_sites": len(di),
                "mean_site_residual": di.mean_size_adjusted_residual.mean(),
                "sd_site_residual": di.mean_size_adjusted_residual.std(ddof=1),
                "min_site_residual": di.mean_size_adjusted_residual.min(),
                "max_site_residual": di.mean_size_adjusted_residual.max()})
        summary.append({"trait": trait, "n_plants": len(d), "n_sites": len(site),
            "size_only_r2_plant": m0.rsquared, "size_island_r2_plant": m1.rsquared,
            "size_island_delta_r2_plant": m1.rsquared - m0.rsquared,
            "common_log_size_slope": m1.params.get("log_body_size", np.nan),
            "common_log_size_p_hc3": m1.pvalues.get("log_body_size", np.nan),
            "aic_size_only_plant": m0.aic, "aic_size_island_plant": m1.aic,
            "aic_interaction_plant": m2.aic,
            "delta_aic_interaction_vs_additive_plant": m2.aic - m1.aic,
            "plant_allometry_site_residual_eta2": e,
            "plant_allometry_site_residual_permutation_p": sensitivity_p,
            **anc, **interaction, "n_permutations": nperm,
            "n_interaction_permutations": interaction_perm})
    tab = pd.DataFrame(summary)
    tab["site_ancova_permutation_p_bh"] = bh(tab.site_ancova_permutation_p.tolist())
    tab["site_interaction_permutation_p_bh"] = bh(tab.site_interaction_permutation_p.tolist())
    tab["plant_allometry_site_residual_permutation_p_bh"] = bh(
        tab.plant_allometry_site_residual_permutation_p.tolist())
    return tab, pd.DataFrame(site_rows), pd.DataFrame(island_rows), pd.DataFrame(loo_rows)


def write_axes(df, traits, prefix, residualise, seed):
    analysed = add_residuals(df, traits) if residualise else df
    cols = [f"resid_{t}" for t in traits] if residualise else traits
    complete = analysed[["island"] + cols].dropna()
    raw, b, w, n, k = variance_components(zscore(complete[cols]), complete.island)
    ax = directional_axes(b, w, cols); lo, hi, nb = hierarchical_bootstrap(
        df, traits, residualise, seed=seed)
    ax["top_axis_boot_lo"] = np.nan; ax["top_axis_boot_hi"] = np.nan; ax["top_axis_boot_n"] = nb
    ax.loc[ax.axis == 1, ["top_axis_boot_lo", "top_axis_boot_hi"]] = [lo, hi]
    ax.to_csv(R / f"{prefix}_pst_axes.csv", index=False)
    for name, matrix in [("B_raw", raw), ("B_psd", b), ("W", w)]:
        pd.DataFrame(matrix, index=cols, columns=cols).to_csv(R / f"{prefix}_{name}.csv")
    return ax, lo, hi, nb, n, k


def main():
    df = pd.read_csv(R / "plant_means.csv", encoding="utf-8-sig")
    df = df[df.island.isin(ISLANDS)].copy(); df["site"] = df.island + "_" + df.no.astype(str)
    df["body_size_mm"] = np.sqrt(df.corolla_length_mm * df.corolla_width_fulleq_mm)
    df["log_body_size"] = np.log(df.body_size_mm); df["guide_asin"] = transformed(df, GUIDE)
    df["guide_z"] = zscore(df[["guide_asin"]]).guide_asin
    df["size_pc1"], size_load = pc1(df, BODY, "corolla_body_size")
    df["shape_pc1"], shape_load = pc1(df, SHAPE, "shape")
    pd.concat([size_load, shape_load], ignore_index=True).to_csv(R / "multivariate_module_loadings.csv", index=False)

    df["size_within"] = df.size_pc1 - df.groupby("island").size_pc1.transform("mean")
    df["guide_within"] = df.guide_z - df.groupby("island").guide_z.transform("mean")
    islands = df.groupby("island", observed=True)[["size_pc1", "log_body_size", "guide_z", "shape_pc1"]].mean()
    sites = df.groupby(["island", "site"], observed=True)[["size_pc1", "guide_z"]].mean().reset_index()
    sites["size_within"] = sites.size_pc1 - sites.groupby("island").size_pc1.transform("mean")
    sites["guide_within"] = sites.guide_z - sites.groupby("island").guide_z.transform("mean")
    correlations = [corr("raw plant-level bodySizePC1 vs guide", df.size_pc1, df.guide_z),
        corr("within-island centred plants", df.size_within, df.guide_within),
        corr("raw site means", sites.size_pc1, sites.guide_z),
        corr("within-island centred sites", sites.size_within, sites.guide_within),
        corr("between-island means", islands.size_pc1, islands.guide_z),
        corr("between-island shapePC1 vs guide", islands.shape_pc1, islands.guide_z)]
    for omitted in islands.index:
        keep = islands.drop(index=omitted)
        correlations.append(corr("between-island means leave-one-island-out",
                                 keep.size_pc1, keep.guide_z, omitted))
    pd.DataFrame(correlations).to_csv(R / "multivariate_size_guide_correlations.csv", index=False)

    tests, site_resid, island_resid, loo = allometry(df)
    tests.to_csv(R / "multivariate_allometry_tests.csv", index=False)
    site_resid.to_csv(R / "multivariate_site_residuals.csv", index=False)
    island_resid.to_csv(R / "multivariate_island_residual_means.csv", index=False)
    loo.to_csv(R / "multivariate_allometry_leave_one_island_out.csv", index=False)
    absolute, alo, ahi, anb, n, k = write_axes(df, ABS_TRAITS, "multivariate_absolute", False, 20260811)
    adjusted, rlo, rhi, rnb, _, _ = write_axes(df, RES_TRAITS, "multivariate_size_adjusted", True, 20260812)
    absolute.to_csv(R / "multivariate_pst_axes.csv", index=False)
    for suffix in ("B_raw", "B_psd", "W"):
        (R / f"multivariate_{suffix}.csv").write_text(
            (R / f"multivariate_absolute_{suffix}.csv").read_text(), encoding="utf-8")
    islands.to_csv(R / "multivariate_island_module_means.csv")

    print("\n=== multivariate phenotypic divergence: body-size null ===")
    print(f"plant means: {len(df)}; complete absolute-trait plants: {n}; islands: {k}")
    print(f"corolla body-size PC1 variance explained: {size_load.pc1_variance_fraction.iloc[0]:.3f}")
    for row in correlations[:6]:
        print(f"{row['comparison']}: n={row['n']} r={row['pearson_r']:+.3f} p={row['pearson_p']:.3g}; "
              f"rho={row['spearman_rho']:+.3f} p={row['spearman_p']:.3g}")
    print("functional traits after log body-size adjustment: site-level inference")
    for _, row in tests.iterrows():
        print(f"  {row.trait:28s} site-ANCOVA partialR2={row.site_ancova_partial_r2:.3f} "
              f"perm p={row.site_ancova_permutation_p:.3g} BH={row.site_ancova_permutation_p_bh:.3g}; "
              f"plant-residual sensitivity p={row.plant_allometry_site_residual_permutation_p:.3g}; "
              f"slope-interaction perm p={row.site_interaction_permutation_p:.3g}; "
              f"deltaAIC(interaction-additive)={row.delta_aic_interaction_vs_additive_plant:+.2f}")
    atop = absolute.iloc[0]; rtop = adjusted.iloc[0]
    print(f"absolute-trait multivariate P_ST analogue={atop.pst_analogue:.3f} "
          f"(hierarchical bootstrap 95% CI {alo:.3f}-{ahi:.3f}; {anb})")
    print(f"size-adjusted functional P_ST analogue={rtop.pst_analogue:.3f} "
          f"(hierarchical bootstrap 95% CI {rlo:.3f}-{rhi:.3f}; {rnb})")
    print("size-adjusted top-axis unit-norm loadings:")
    for trait in [f"resid_{t}" for t in RES_TRAITS]:
        print(f"  {trait:36s} {rtop[f'loading_{trait}']:+.3f}")


if __name__ == "__main__":
    main()
