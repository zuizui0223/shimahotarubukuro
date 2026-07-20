#!/usr/bin/env python3
"""Among-island analysis of the floral traits: Pst and site-corrected comparisons.

Consumes the finalised extraction output (results_shimask_all/corolla_master.csv,
plus the colour-free guide spatial metrics in guide_spatial.csv) and asks how the
traits diverge among the five Izu-island populations.

Two levels of non-independence are handled:
  - flowers within a plant  -> everything is aggregated to PLANT MEANS first
    (individual = island x site(no) x id; 1-2 flowers/plant; 125 plants).
  - plants within a site     -> sites are uneven (e.g. Niijima), so the island
    comparison is a LINEAR MIXED MODEL with island as a fixed effect and site as a
    random intercept (statsmodels), tested by a likelihood-ratio test. This
    "site-corrects" the island effect instead of letting a heavily-sampled site
    dominate its island.

Per trait it reports:
  - Pst = Vb/(Vb+2Vw) on plant means, with a 95% bootstrap CI (a PHENOTYPIC
    surrogate for Qst - it ranks divergence, it is not a Qst-Fst test).
  - the site-corrected island test (mixed-model LRT p, Benjamini-Hochberg adjusted).
  - a plant-level Kruskal-Wallis (uncorrected) for comparison.
  - a latitude cline (Spearman; Oshima 34.75 -> Kozushima 34.22 degN).

Nectar-guide traits are the amount (coverage, spot count, density) and the COLOUR-
FREE spatial distribution of the spots (basal concentration and petal-midline
concentration); guide colour/contrast is dropped because it is not reliably
measurable on dried, pressed specimens.

Writes results_shimask_all/island_analysis_stats.csv and plant_means.csv.
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from plot_island_traits import island_of  # noqa: F401

RESULTS = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
LABEL = {"oshima": "Oshima", "toshima": "Toshima", "niijima": "Niijima",
         "shikine": "Shikinejima", "kozu": "Kozushima"}

TRAITS = [
    ("corolla_length_mm", "corolla length"),
    ("corolla_width_fulleq_mm", "corolla width"),
    ("corolla_area_fulleq_mm2", "corolla area"),
    ("throat_width_mm", "throat width"),
    ("mouth_width_mm", "mouth width"),
    ("corolla_aspect_L_W", "corolla aspect"),
    ("tube_flare_W_throat", "tube flare"),
    ("lobe_incision_mm", "lobe incision"),
    ("style_length_mm", "organ/style length"),
    ("style_corolla_ratio", "style / corolla"),
    ("guide_coverage_pct", "guide coverage"),
    ("n_guide_spots", "guide spot count"),
    ("guide_density_per_cm2", "guide density"),
    # colour-free spatial distribution of the spots (robust on dried specimens)
    ("guide_basal_frac", "guide basal concentration"),
    ("guide_midline_ratio", "guide midline concentration"),
]


def load_plant_means():
    """One row per plant (island, site, id): mean of each trait over its flowers."""
    rows = list(csv.DictReader((RESULTS / "corolla_master.csv").open(encoding="utf-8-sig")))
    # Attach colour-free guide spatial-distribution metrics (per guided corolla).
    sp = {(g["sheet"], g["corolla_id"]): g for g in
          csv.DictReader((RESULTS / "guide_spatial.csv").open(encoding="utf-8-sig"))}
    for r in rows:
        g = sp.get((r["sheet"], r["sheet_corolla_id"]))
        r["guide_basal_frac"] = g["basal_frac_prox_third"] if g else ""
        r["guide_midline_ratio"] = g["midline_ratio_distal"] if g else ""

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["island"], r["no"], r["id"])].append(r)
    plants = []
    for (isl, site, pid), fl in groups.items():
        rec = {"island": isl, "no": site, "site": f"{isl}_{site}",
               "id": pid, "lat": float(fl[0]["lat"]), "n_flowers": len(fl)}
        for key, _ in TRAITS:
            vals = [float(f[key]) for f in fl if f.get(key, "") not in ("", "nan")]
            rec[key] = float(np.mean(vals)) if vals else np.nan
        ph = [f["status"] for f in fl if f["status"] in ("s", "p")]
        rec["frac_female"] = (ph.count("p") / len(ph)) if ph else np.nan
        plants.append(rec)
    return plants


def pst_value(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan
    grand = np.concatenate(groups)
    n, k = grand.size, len(groups)
    ni = [g.size for g in groups]
    ss_b = sum(nj * (g.mean() - grand.mean()) ** 2 for nj, g in zip(ni, groups))
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_w = ss_w / (n - k)
    n0 = (n - sum(x * x for x in ni) / n) / (k - 1)
    vb = max((ss_b / (k - 1) - ms_w) / n0, 0.0)
    return vb / (vb + 2 * ms_w) if (vb + ms_w) > 0 else 0.0


def pst_bootstrap(by_isl, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    obs = pst_value(list(by_isl.values()))
    boots = [pst_value([rng.choice(g, len(g), replace=True)
                        for g in by_isl.values() if len(g) >= 2]) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, lo, hi


def site_corrected_p(df, key):
    """Likelihood-ratio p for the island fixed effect, site as random intercept."""
    sub = df[["island", "site", key]].dropna().rename(columns={key: "y"})
    if sub["island"].nunique() < 2 or len(sub) < 8:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = smf.mixedlm("y ~ C(island)", sub, groups=sub["site"]).fit(reml=False)
            null = smf.mixedlm("y ~ 1", sub, groups=sub["site"]).fit(reml=False)
        lr = 2.0 * (full.llf - null.llf)
        ddf = sub["island"].nunique() - 1
        return float(lr), float(stats.chi2.sf(max(lr, 0), ddf))
    except Exception:
        return np.nan, np.nan


def bh_adjust(pvals):
    p = np.array([x if x == x else 1.0 for x in pvals], float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, p[idx] * m / (m - rank))
        adj[idx] = prev
    return adj


def main() -> None:
    plants = load_plant_means()
    df = pd.DataFrame(plants)

    cols = ["island", "no", "id", "lat", "n_flowers", "frac_female"] + [k for k, _ in TRAITS]
    with (RESULTS / "plant_means.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in plants:
            w.writerow({c: (round(p[c], 4) if isinstance(p.get(c), float) and p[c] == p[c]
                            else (p.get(c, "") if p.get(c) == p.get(c) else "")) for c in cols})
    print(f"wrote {RESULTS/'plant_means.csv'}  ({len(plants)} plants)")
    n_sites = {i: df[df.island == i]["site"].nunique() for i in ISLANDS}
    n_pl = {i: int((df.island == i).sum()) for i in ISLANDS}
    print("plants/island:", {LABEL[i]: n_pl[i] for i in ISLANDS})
    print("sites/island: ", {LABEL[i]: n_sites[i] for i in ISLANDS})

    results, kw_p, site_p = [], [], []
    for key, label in TRAITS:
        by_isl = {i: df[(df.island == i)][key].dropna().values for i in ISLANDS}
        by_isl = {i: v for i, v in by_isl.items() if len(v)}
        pst, lo, hi = pst_bootstrap(by_isl)
        groups = [v for v in by_isl.values() if len(v) >= 2]
        H, p_kw = stats.kruskal(*groups)
        lr, p_site = site_corrected_p(df, key)
        sub = df[[key, "lat"]].dropna()
        rho, p_lat = stats.spearmanr(sub["lat"], sub[key])
        results.append({"trait": label, "key": key,
                        "pst": round(pst, 3), "pst_lo": round(lo, 3), "pst_hi": round(hi, 3),
                        "kw_p": p_kw, "site_lrt": round(lr, 2) if lr == lr else "",
                        "site_p": p_site, "lat_rho": round(rho, 2), "lat_p": p_lat,
                        **{f"mean_{i}": (round(float(np.mean(by_isl[i])), 2) if i in by_isl else "")
                           for i in ISLANDS}})
        kw_p.append(p_kw)
        site_p.append(p_site)

    for r, a, b in zip(results, bh_adjust(kw_p), bh_adjust(site_p)):
        r["kw_p_adj"], r["site_p_adj"] = a, b

    fields = ["trait", "key", "pst", "pst_lo", "pst_hi", "kw_p", "kw_p_adj",
              "site_lrt", "site_p", "site_p_adj", "lat_rho", "lat_p"] + [f"mean_{i}" for i in ISLANDS]
    with (RESULTS / "island_analysis_stats.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.2e}" if k in ("kw_p", "kw_p_adj", "site_p", "site_p_adj", "lat_p")
                            and r[k] == r[k] else r[k]) for k in fields})
    print(f"wrote {RESULTS/'island_analysis_stats.csv'}")

    print("\n=== among-island analysis (plant means; site-corrected island test) ===")
    print(f"{'trait':26} {'Pst [95% CI]':>18} {'site p(adj)':>12} {'KW p(adj)':>11} {'lat rho':>8}")
    for r in sorted(results, key=lambda r: -r["pst"]):
        sp = f"{r['site_p_adj']:.1e}" if r["site_p_adj"] == r["site_p_adj"] else "  n/a"
        star = "" if (r["site_p_adj"] == r["site_p_adj"] and r["site_p_adj"] < 0.05) else " n.s."
        print(f"{r['trait']:26} {r['pst']:.2f} [{r['pst_lo']:.2f},{r['pst_hi']:.2f}]"
              f"  {sp}{star:5} {r['kw_p_adj']:.1e}  {r['lat_rho']:+.2f}")


if __name__ == "__main__":
    main()
