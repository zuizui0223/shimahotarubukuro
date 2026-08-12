#!/usr/bin/env python3
"""Extended diagnostics for complex floral-syndrome assembly.

Runs after multivariate_phenotype.py and syndrome_modules.py. The goal is to
extract the strongest defensible hypotheses from the current cross-sectional
data while keeping a strict boundary between measured reproductive geometry and
future direct reproductive-assurance assays.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

R = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
MODULES = ["attraction", "mechanical_fit", "reproductive"]
MODULE_TRAITS = {
    "attraction": ["guide_coverage_pct"],
    "mechanical_fit": ["mouth_width_mm", "throat_width_mm", "tube_flare_W_throat"],
    "reproductive": ["organ_length_mm", "organ_corolla_ratio"],
}
CORE_TRAITS = sum(MODULE_TRAITS.values(), [])


def zframe(df, cols):
    out = df[cols].astype(float).copy()
    return (out - out.mean()) / out.std(ddof=1)


def profile_permanova(df, cols, nperm=4999, seed=20260824):
    """Test non-parallel profiles by permuting whole-site island labels."""
    d = df.dropna(subset=cols).copy()
    X = zframe(d, cols).to_numpy(float)
    X = X - X.mean(axis=1, keepdims=True)
    labels = d.island.to_numpy()
    groups = np.unique(labels)
    grand = X.mean(axis=0)

    def pseudo_f(lab):
        ssb = 0.0
        ssw = 0.0
        for g in groups:
            xx = X[lab == g]
            centre = xx.mean(axis=0)
            ssb += len(xx) * float(np.sum((centre - grand) ** 2))
            ssw += float(np.sum((xx - centre) ** 2))
        return (ssb / (len(groups) - 1)) / (ssw / (len(X) - len(groups)))

    observed = pseudo_f(labels)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(nperm):
        exceed += pseudo_f(rng.permutation(labels)) >= observed - 1e-12
    df_num = (len(groups) - 1) * (len(cols) - 1)
    df_den = (len(X) - len(groups)) * (len(cols) - 1)
    return {
        "n_islands": len(groups),
        "n_sites": len(d),
        "n_profile_units": len(cols),
        "pseudo_f": observed,
        "approx_parametric_df_num": df_num,
        "approx_parametric_df_den": df_den,
        "approx_parametric_p": float(stats.f.sf(max(observed, 0), df_num, df_den)),
        "site_label_permutation_p": (exceed + 1) / (nperm + 1),
        "n_permutations": nperm,
    }


def profile_wide_from_site_residuals(site_resid, traits):
    d = site_resid[site_resid.trait.isin(traits)]
    return d.pivot_table(
        index=["island", "site"], columns="trait",
        values="mean_size_adjusted_residual"
    ).reset_index()


def integration_levels(site_scores, nboot=1500, seed=20260825):
    pairs = list(combinations(MODULES, 2))

    def metrics(d):
        within = d[MODULES] - d.groupby("island")[MODULES].transform("mean")
        between = d.groupby("island", observed=True)[MODULES].mean()
        out = {}
        for a, b in pairs:
            rw = float(within[[a, b]].corr().iloc[0, 1])
            rb = float(between[[a, b]].corr().iloc[0, 1])
            out[(a, b)] = (rw, rb, rb - rw)
        return out

    observed = metrics(site_scores)
    rng = np.random.default_rng(seed)
    boots = {p: [] for p in pairs}
    for _ in range(nboot):
        pieces = []
        for _, di in site_scores.groupby("island", observed=True):
            pieces.append(di.iloc[rng.integers(0, len(di), len(di))].copy())
        vals = metrics(pd.concat(pieces, ignore_index=True))
        for p in pairs:
            boots[p].append(vals[p])

    rows = []
    for p in pairs:
        arr = np.asarray(boots[p], float)
        rw, rb, delta = observed[p]
        rows.append({
            "module_a": p[0],
            "module_b": p[1],
            "within_island_centered_site_r": rw,
            "between_island_mean_r": rb,
            "between_minus_within_r": delta,
            "within_boot_lo": np.nanquantile(arr[:, 0], 0.025),
            "within_boot_hi": np.nanquantile(arr[:, 0], 0.975),
            "between_boot_lo": np.nanquantile(arr[:, 1], 0.025),
            "between_boot_hi": np.nanquantile(arr[:, 1], 0.975),
            "difference_boot_lo": np.nanquantile(arr[:, 2], 0.025),
            "difference_boot_hi": np.nanquantile(arr[:, 2], 0.975),
            "bootstrap_pr_between_gt_within": float(np.mean(arr[:, 2] > 0)),
            "n_bootstrap": len(arr),
            "interpretation": "positive delta = stronger population-level co-divergence than within-island phenotypic integration",
        })

    loo = []
    for omitted in sorted(site_scores.island.unique()):
        vals = metrics(site_scores[site_scores.island != omitted])
        for p, (rw, rb, delta) in vals.items():
            loo.append({
                "omitted_island": omitted,
                "module_a": p[0],
                "module_b": p[1],
                "within_r": rw,
                "between_r": rb,
                "between_minus_within_r": delta,
            })
    return pd.DataFrame(rows), pd.DataFrame(loo)


def matrix_variance_components(z, groups):
    d = z.copy()
    d["_group"] = groups.to_numpy()
    d = d.dropna()
    cols = list(z.columns)
    arrays = [x[cols].to_numpy(float) for _, x in d.groupby("_group") if len(x) >= 1]
    if len(arrays) < 2:
        raise ValueError("need at least two groups")
    ni = np.array([len(a) for a in arrays], float)
    n, k, p = int(ni.sum()), len(arrays), len(cols)
    grand = np.vstack(arrays).mean(axis=0)
    ssw = np.zeros((p, p))
    ssb = np.zeros((p, p))
    for nn, arr in zip(ni, arrays):
        centre = arr.mean(axis=0)
        c = arr - centre
        ssw += c.T @ c
        delta = (centre - grand)[:, None]
        ssb += nn * (delta @ delta.T)
    msw = ssw / (n - k)
    msb = ssb / (k - 1)
    n0 = (n - (ni @ ni) / n) / (k - 1)
    b = (msb - msw) / n0
    b = (b + b.T) / 2
    vals, vecs = np.linalg.eigh(b)
    b = vecs @ np.diag(np.clip(vals, 0, None)) @ vecs.T
    return (b + b.T) / 2, msw


def max_axis_pst(wide, cols):
    d = wide[["island"] + cols].dropna().copy()
    z = zframe(d, cols)
    b, w = matrix_variance_components(z, d.island)
    ridge = max(float(np.trace(w)) / len(cols), 1.0) * 1e-8
    vals, vecs = linalg.eigh(b, w + np.eye(len(cols)) * ridge)
    j = int(np.argmax(vals))
    lam = float(vals[j])
    v = vecs[:, j].copy()
    if v[np.argmax(np.abs(v))] < 0:
        v *= -1
    v /= np.linalg.norm(v)
    return lam / (lam + 2.0), v


def module_multivariate_pst(site_resid, nboot=800, seed=20260826):
    rng = np.random.default_rng(seed)
    rows = []
    for module, traits in MODULE_TRAITS.items():
        wide = profile_wide_from_site_residuals(site_resid, traits)
        observed, loading = max_axis_pst(wide, traits)
        vals = []
        for _ in range(nboot):
            pieces = []
            for _, di in wide.groupby("island", observed=True):
                pieces.append(di.iloc[rng.integers(0, len(di), len(di))].copy())
            try:
                value, _ = max_axis_pst(pd.concat(pieces, ignore_index=True), traits)
                vals.append(float(value))
            except Exception:
                pass
        lo, hi = np.quantile(vals, [0.025, 0.975])
        row = {
            "module": module,
            "scope": "reproductive geometry proxy; not reproductive assurance" if module == "reproductive" else "current measured proxy",
            "site_level_multivariate_max_axis_pst": observed,
            "bootstrap_lo": lo,
            "bootstrap_hi": hi,
            "n_bootstrap": len(vals),
        }
        for trait, coef in zip(traits, loading):
            row[f"top_axis_loading_{trait}"] = float(coef)
        rows.append(row)
    return pd.DataFrame(rows)


def sampling_gaps(plant):
    d = plant.copy()
    d["site"] = d.island.astype(str) + "_" + d.no.astype(str)
    d["body_size_mm"] = np.sqrt(d.corolla_length_mm * d.corolla_width_fulleq_mm)
    support = d.groupby("island").body_size_mm.agg(["min", "max"])
    lo, hi = float(support["min"].max()), float(support["max"].min())
    d["common"] = d.body_size_mm.between(lo, hi)
    rows = []
    for island in ISLANDS:
        di = d[d.island == island]
        dc = di[di.common]
        row = {
            "island": island,
            "n_plants": len(di),
            "n_sites": di.site.nunique(),
            "common_support_lo_mm": lo,
            "common_support_hi_mm": hi,
            "common_support_plants": len(dc),
            "common_support_sites": dc.site.nunique(),
        }
        for target in [3, 5, 8, 10]:
            row[f"additional_sites_to_{target}"] = max(target - di.site.nunique(), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def pseudo_f_only(df, cols):
    d = df.dropna(subset=cols)
    X = zframe(d, cols).to_numpy(float)
    X -= X.mean(axis=1, keepdims=True)
    labels = d.island.to_numpy()
    groups = np.unique(labels)
    grand = X.mean(axis=0)
    ssb = 0.0
    ssw = 0.0
    for g in groups:
        xx = X[labels == g]
        centre = xx.mean(axis=0)
        ssb += len(xx) * float(np.sum((centre - grand) ** 2))
        ssw += float(np.sum((xx - centre) ** 2))
    return (ssb / (len(groups)-1)) / (ssw / (len(X)-len(groups)))


def planning_power(site_scores, nsim=500, seed=20260827):
    means = site_scores.groupby("island", observed=True)[MODULES].mean().reindex(ISLANDS)
    residuals = pd.concat([
        di[MODULES] - di[MODULES].mean()
        for _, di in site_scores.groupby("island")
    ])
    cov = residuals.cov().to_numpy(float)
    eig, vec = np.linalg.eigh((cov + cov.T) / 2)
    cov = vec @ np.diag(np.clip(eig, 1e-8, None)) @ vec.T
    rng = np.random.default_rng(seed)
    rows = []
    for nsite in [2, 3, 4, 5, 6, 8, 10, 12]:
        hits = 0
        pvals = []
        for sim in range(nsim):
            records = []
            for island in ISLANDS:
                x = rng.multivariate_normal(means.loc[island].to_numpy(float), cov, size=nsite)
                for j, vector in enumerate(x):
                    records.append({
                        "island": island,
                        "site": f"{island}_{sim}_{j}",
                        **{m: vector[k] for k, m in enumerate(MODULES)},
                    })
            dd = pd.DataFrame(records)
            F = pseudo_f_only(dd, MODULES)
            df1 = (len(ISLANDS)-1)*(len(MODULES)-1)
            df2 = (len(dd)-len(ISLANDS))*(len(MODULES)-1)
            p = float(stats.f.sf(max(F, 0), df1, df2))
            pvals.append(p)
            hits += p < 0.05
        rows.append({
            "sites_per_island": nsite,
            "total_sites": nsite*len(ISLANDS),
            "approx_power_alpha_0_05": hits/nsim,
            "median_parametric_p": np.median(pvals),
            "n_simulations": nsim,
            "assumption": "current island module means and pooled within-island site covariance; planning only",
        })
    return pd.DataFrame(rows)


def mosaic_states():
    path = R / "syndrome_module_island_trajectories.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    rows = []
    for _, r in d.iterrows():
        residuals = {m: r[f"rank1_residual_{m}"] for m in MODULES}
        rows.append({
            "island": r.island,
            "syndrome_axis_score": r.syndrome_axis_score,
            "mosaic_deviation_norm": r.orthogonal_residual_norm,
            "largest_positive_module_deviation": max(residuals, key=residuals.get),
            "largest_negative_module_deviation": min(residuals, key=residuals.get),
            **{f"deviation_{m}": residuals[m] for m in MODULES},
        })
    return pd.DataFrame(rows).sort_values("mosaic_deviation_norm", ascending=False)


def scope_table():
    return pd.DataFrame([
        {"component": "attraction", "current": "nectar-guide coverage", "evidence_status": "direct signal-amount proxy", "priority_next": "visible/UV contrast; display; nectar; scent"},
        {"component": "pollination_function", "current": "mouth width; throat width; tube flare", "evidence_status": "morphological fit proxies", "priority_next": "visitor body/proboscis size; stigma/anther contact; pollen removal/deposition"},
        {"component": "reproductive_assurance", "current": "organ length and organ/corolla ratio", "evidence_status": "NOT direct assurance; reproductive geometry proxy only", "priority_next": "bagged autonomous seed set; hand-self; hand-cross; open + supplemental cross; herkogamy; dichogamy; SC/SI"},
        {"component": "population_history", "current": "no population-matched neutral baseline used here", "evidence_status": "must remain separate until sampling is matched", "priority_next": "same-plant/same-site neutral SNPs; coancestry; common garden/families"},
    ])


def fix_existing_interpretation():
    path = R / "syndrome_module_heterogeneity.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    p = float(d.loc[0, "module_by_island_permutation_p"])
    if p < 0.05:
        text = "Module-by-island interaction supports non-parallel module divergence; temporal asynchrony remains untested."
    elif p < 0.10:
        text = "Module heterogeneity is suggestive but not decisive; a shared syndrome trajectory remains plausible."
    else:
        text = "Current data do not reject a shared three-module syndrome trajectory, despite heterogeneous trait-level response modes."
    d.loc[0, "interpretation"] = text
    d.to_csv(path, index=False)


def main():
    fix_existing_interpretation()
    site = pd.read_csv(R / "syndrome_module_site_scores.csv")
    site_resid = pd.read_csv(R / "multivariate_site_residuals.csv")
    plant = pd.read_csv(R / "plant_means.csv", encoding="utf-8-sig")

    module_profile = profile_permanova(site, MODULES, 4999, 20260824)
    module_profile.update({"profile_level": "module", "interpretation": "shared-axis vs non-parallel module profile"})
    pd.DataFrame([module_profile]).to_csv(R / "syndrome_module_profile_permanova.csv", index=False)

    trait_wide = profile_wide_from_site_residuals(site_resid, CORE_TRAITS)
    trait_profile = profile_permanova(trait_wide, CORE_TRAITS, 4999, 20260825)
    trait_profile.update({"profile_level": "trait", "interpretation": "body-size-adjusted trait mosaic / non-parallel trait response"})
    pd.DataFrame([trait_profile]).to_csv(R / "syndrome_trait_profile_heterogeneity.csv", index=False)
    trait_wide.groupby("island", observed=True)[CORE_TRAITS].mean().reset_index().to_csv(R / "syndrome_trait_island_profiles.csv", index=False)

    integ, loo = integration_levels(site)
    integ.to_csv(R / "syndrome_integration_levels.csv", index=False)
    loo.to_csv(R / "syndrome_integration_leave_one_island_out.csv", index=False)

    module_multivariate_pst(site_resid).to_csv(R / "syndrome_module_multivariate_pst.csv", index=False)
    sampling_gaps(plant).to_csv(R / "syndrome_sampling_gaps.csv", index=False)
    planning_power(site).to_csv(R / "syndrome_module_power.csv", index=False)
    mosaic_states().to_csv(R / "syndrome_mosaic_states.csv", index=False)
    scope_table().to_csv(R / "syndrome_measurement_scope.csv", index=False)

    print("\n=== extended complex-adaptation diagnostics ===")
    print(f"module profile permutation p={module_profile['site_label_permutation_p']:.4g}")
    print(f"trait profile permutation p={trait_profile['site_label_permutation_p']:.4g}")
    print("within-island vs between-island module integration:")
    for _, r in integ.iterrows():
        print(f"  {r.module_a:16s} x {r.module_b:16s} within={r.within_island_centered_site_r:+.3f} between={r.between_island_mean_r:+.3f} delta={r.between_minus_within_r:+.3f} bootPr(delta>0)={r.bootstrap_pr_between_gt_within:.3f}")
    print("module multivariate site-P_ST:")
    mm = pd.read_csv(R / "syndrome_module_multivariate_pst.csv")
    for _, r in mm.iterrows():
        print(f"  {r.module:16s} {r.site_level_multivariate_max_axis_pst:.3f} [{r.bootstrap_lo:.3f},{r.bootstrap_hi:.3f}]")
    print("planning power for module x island interaction:")
    pw = pd.read_csv(R / "syndrome_module_power.csv")
    for _, r in pw.iterrows():
        print(f"  {int(r.sites_per_island):2d} sites/island -> power~{r.approx_power_alpha_0_05:.2f}")
    print("Guardrail: reproductive assurance itself is not yet measured; current reproductive results are geometry proxies.")


if __name__ == "__main__":
    main()
