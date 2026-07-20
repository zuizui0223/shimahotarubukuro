#!/usr/bin/env python3
"""Among-island analysis of the floral traits: Pst and island comparisons.

Consumes the finalised extraction output (results_shimask_all/corolla_master.csv)
and asks how the traits diverge among the five Izu-island populations.

Pseudoreplication matters here: a plant contributes 1-2 flowers, so every test is
run on PLANT MEANS (one value per individual = island x site(no) x id), not per
flower. The individual id is numbered within a site, so the key includes the site
number. n = 47/38/23/5/12 plants for Oshima/Toshima/Niijima/Shikinejima/Kozushima
(125 plants, 218 flowers).

Per continuous trait it reports:
  - Pst = Vb / (Vb + 2 Vw) from a one-way island variance decomposition on plant
    means, with a 95% bootstrap CI (resampling plants within islands). Pst is a
    PHENOTYPIC surrogate for Qst; a definitive Qst-Fst test needs neutral markers,
    so Pst is read as "how strongly the trait diverges among islands", compared
    across traits, not as proof of selection.
  - Kruskal-Wallis across the five islands, its p, a Benjamini-Hochberg adjusted p
    (across traits), and the epsilon^2 effect size.
  - Oshima (the only island with a bumblebee, Bombus ardens) vs the pooled
    bumblebee-free islands (Mann-Whitney).
  - Spearman correlation of the plant values with latitude (a north-south cline;
    Oshima 34.75 -> Kozushima 34.22 degN).

Also tests the sexual phase (protandry): proportion of female-phase flowers per
island (chi-square). Writes results_shimask_all/island_analysis_stats.csv and
results_shimask_all/plant_means.csv.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from plot_island_traits import ORDER, island_of  # noqa: F401  (ORDER order/labels)

RESULTS = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
BOMBUS = {"oshima"}
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
    ("guide_reach_frac", "guide reach"),
    ("guide_contrast_dE", "guide contrast"),
    ("guide_saturation", "guide chroma"),
]


def load_plant_means():
    """One row per plant (island, id): mean of each trait over its flowers."""
    rows = list(csv.DictReader((RESULTS / "corolla_master.csv").open(encoding="utf-8-sig")))
    # Individual = (island, site no, id); id is numbered within a site, so the site
    # number is part of the key. Each plant has 1-2 flowers.
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["island"], r["no"], r["id"])].append(r)
    plants = []
    for (isl, site, pid), fl in groups.items():
        rec = {"island": isl, "no": site, "id": pid, "lat": float(fl[0]["lat"]),
               "n_flowers": len(fl)}
        for key, _ in TRAITS:
            vals = [float(f[key]) for f in fl if f.get(key, "") not in ("", "nan")]
            rec[key] = float(np.mean(vals)) if vals else np.nan
        # sexual phase: fraction of this plant's flowers in female phase
        ph = [f["status"] for f in fl if f["status"] in ("s", "p")]
        rec["frac_female"] = (ph.count("p") / len(ph)) if ph else np.nan
        plants.append(rec)
    return plants


def pst_value(groups):
    """Pst = Vb/(Vb+2Vw) from a one-way decomposition of the group arrays."""
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
    boots = []
    for _ in range(n_boot):
        resampled = [rng.choice(g, len(g), replace=True) for g in by_isl.values() if len(g) >= 2]
        boots.append(pst_value(resampled))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, lo, hi


def epsilon_sq(groups):
    """Kruskal-Wallis effect size epsilon^2 = (H - k + 1) / (n - k)."""
    n = sum(len(g) for g in groups)
    k = len(groups)
    H, _ = stats.kruskal(*groups)
    return (H - k + 1) / (n - k) if n > k else np.nan


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = m - rank
        prev = min(prev, p[idx] * m / r)
        adj[idx] = prev
    return adj


def main() -> None:
    plants = load_plant_means()
    # write plant means
    pm = RESULTS / "plant_means.csv"
    cols = ["island", "no", "id", "lat", "n_flowers", "frac_female"] + [k for k, _ in TRAITS]
    with pm.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in plants:
            w.writerow({c: (round(p[c], 4) if isinstance(p.get(c), float) and p[c] == p[c] else
                            (p.get(c, "") if p.get(c) == p.get(c) else "")) for c in cols})
    print(f"wrote {pm}  ({len(plants)} plants)")

    n_by_isl = {i: sum(1 for p in plants if p["island"] == i) for i in ISLANDS}
    print("plants/island:", {LABEL[i]: n_by_isl[i] for i in ISLANDS})

    results, kw_p = [], []
    for key, label in TRAITS:
        by_isl = {}
        for i in ISLANDS:
            v = np.array([p[key] for p in plants if p["island"] == i and p[key] == p[key]])
            if v.size:
                by_isl[i] = v
        groups = [g for g in by_isl.values() if len(g) >= 2]
        pst, lo, hi = pst_bootstrap(by_isl)
        H, p_kw = stats.kruskal(*groups)
        eps = epsilon_sq(groups)
        # Bombus present vs free
        pres = np.concatenate([by_isl[i] for i in by_isl if i in BOMBUS])
        free = np.concatenate([by_isl[i] for i in by_isl if i not in BOMBUS])
        _, p_bombus = stats.mannwhitneyu(pres, free, alternative="two-sided")
        # latitude cline (plant values vs latitude)
        xs = np.array([p["lat"] for p in plants if p[key] == p[key]])
        ys = np.array([p[key] for p in plants if p[key] == p[key]])
        rho, p_lat = stats.spearmanr(xs, ys)
        means = {i: (by_isl[i].mean() if i in by_isl else np.nan) for i in ISLANDS}
        results.append({"trait": label, "key": key,
                        "pst": round(pst, 3), "pst_lo": round(lo, 3), "pst_hi": round(hi, 3),
                        "kw_H": round(H, 2), "kw_p": p_kw, "eps2": round(eps, 3),
                        "bombus_p": p_bombus, "lat_rho": round(rho, 2), "lat_p": p_lat,
                        **{f"mean_{i}": round(means[i], 2) for i in ISLANDS}})
        kw_p.append(p_kw)

    adj = bh_adjust(kw_p)
    for r, a in zip(results, adj):
        r["kw_p_adj"] = a

    out = RESULTS / "island_analysis_stats.csv"
    fields = ["trait", "key", "pst", "pst_lo", "pst_hi", "kw_H", "kw_p", "kw_p_adj",
              "eps2", "bombus_p", "lat_rho", "lat_p"] + [f"mean_{i}" for i in ISLANDS]
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.2e}" if k in ("kw_p", "kw_p_adj", "bombus_p", "lat_p")
                            else r[k]) for k in fields})
    print(f"wrote {out}")

    # sexual phase across islands (flower-level chi-square)
    rows = list(csv.DictReader((RESULTS / "corolla_master.csv").open(encoding="utf-8-sig")))
    tab = []
    for i in ISLANDS:
        s = sum(1 for r in rows if r["island"] == i and r["status"] == "s")
        p = sum(1 for r in rows if r["island"] == i and r["status"] == "p")
        tab.append([s, p])
    chi2, p_phase, _, _ = stats.chi2_contingency(np.array(tab).T)

    print("\n=== among-island analysis (plant means) ===")
    print(f"{'trait':20} {'Pst [95% CI]':>20} {'KW p(adj)':>11} {'eps2':>6} {'Bombus p':>10} {'lat rho':>8}")
    for r in sorted(results, key=lambda r: -r["pst"]):
        print(f"{r['trait']:20} {r['pst']:.2f} [{r['pst_lo']:.2f},{r['pst_hi']:.2f}]"
              f"   {r['kw_p_adj']:.1e}  {r['eps2']:.2f}   {r['bombus_p']:.1e}  {r['lat_rho']:+.2f}")
    print(f"\nsexual phase (female fraction) by island: "
          f"{[f'{LABEL[i]} {tab[j][1]/(sum(tab[j]) or 1):.0%}' for j,i in enumerate(ISLANDS)]}")
    print(f"  chi-square across islands: chi2={chi2:.1f}, p={p_phase:.3f}")


if __name__ == "__main__":
    main()
