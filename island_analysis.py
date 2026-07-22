#!/usr/bin/env python3
"""Final among-island analysis through global and pairwise Pst.

The analysis consumes ``corolla_master.csv`` plus the colour-free guide spatial
metrics. Two nested sampling levels are handled before island comparison:

* 1-2 flowers within a plant are averaged to one plant mean
  (plant = island x site x individual; 125 plants).
* unevenly sampled sites are handled with a mixed model containing island as a fixed
  effect and site as a random intercept.

For every retained trait the script reports plant-level Pst with a bootstrap 95% CI,
the site-corrected island likelihood-ratio test with BH correction, a plant-level
Kruskal-Wallis comparison, and a latitude Spearman correlation. It also calculates
Pst separately for every island pair. Pst is a phenotypic divergence surrogate used
to rank traits; it is not a Qst-Fst test.
"""
from __future__ import annotations

import csv
import itertools
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
LABEL = {
    "oshima": "Oshima",
    "toshima": "Toshima",
    "niijima": "Niijima",
    "shikine": "Shikinejima",
    "kozu": "Kozushima",
}

TRAITS = [
    ("corolla_length_mm", "corolla length"),
    ("corolla_width_fulleq_mm", "corolla width"),
    ("corolla_area_fulleq_mm2", "corolla area"),
    ("throat_width_mm", "throat width"),
    ("mouth_width_mm", "mouth width"),
    ("corolla_aspect_L_W", "corolla aspect"),
    ("tube_flare_W_throat", "tube flare"),
    ("lobe_incision_mm", "lobe incision"),
    ("organ_length_mm", "reproductive-organ length"),
    ("organ_corolla_ratio", "organ / corolla"),
    ("guide_coverage_pct", "guide coverage"),
    ("guide_basal_frac", "guide basal concentration"),
    ("guide_midline_ratio", "guide midline concentration"),
]


def load_plant_means():
    """Return one row per plant, averaging its 1-2 measured corollas."""
    rows = list(csv.DictReader((RESULTS / "corolla_master.csv").open(encoding="utf-8-sig")))
    spatial = {
        (r["sheet"], r["corolla_id"]): r
        for r in csv.DictReader((RESULTS / "guide_spatial.csv").open(encoding="utf-8-sig"))
    }
    for row in rows:
        guide = spatial.get((row["sheet"], row["sheet_corolla_id"]))
        row["guide_basal_frac"] = guide["basal_frac_prox_third"] if guide else ""
        row["guide_midline_ratio"] = guide["midline_ratio_distal"] if guide else ""

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["island"], row["no"], row["id"])].append(row)

    plants = []
    for (island, site, plant_id), flowers in groups.items():
        record = {
            "island": island,
            "no": site,
            "site": f"{island}_{site}",
            "id": plant_id,
            "lat": float(flowers[0]["lat"]),
            "n_flowers": len(flowers),
        }
        for key, _label in TRAITS:
            values = [
                float(flower[key])
                for flower in flowers
                if flower.get(key, "") not in ("", "nan")
            ]
            record[key] = float(np.mean(values)) if values else np.nan
        phases = [flower["status"] for flower in flowers if flower["status"] in ("s", "p")]
        record["frac_female"] = phases.count("p") / len(phases) if phases else np.nan
        plants.append(record)
    return plants


def pst_value(groups):
    groups = [np.asarray(group, float) for group in groups if len(group) >= 2]
    if len(groups) < 2:
        return np.nan
    grand = np.concatenate(groups)
    n, k = grand.size, len(groups)
    sizes = [group.size for group in groups]
    ss_between = sum(
        size * (group.mean() - grand.mean()) ** 2
        for size, group in zip(sizes, groups)
    )
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    ms_within = ss_within / (n - k)
    n0 = (n - sum(size * size for size in sizes) / n) / (k - 1)
    variance_between = max((ss_between / (k - 1) - ms_within) / n0, 0.0)
    return (
        variance_between / (variance_between + 2 * ms_within)
        if (variance_between + ms_within) > 0 else 0.0
    )


def pst_bootstrap(by_island, n_boot: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    observed = pst_value(list(by_island.values()))
    bootstrap = [
        pst_value([
            rng.choice(group, len(group), replace=True)
            for group in by_island.values() if len(group) >= 2
        ])
        for _ in range(n_boot)
    ]
    lower, upper = np.nanpercentile(bootstrap, [2.5, 97.5])
    return observed, lower, upper


def site_corrected_p(dataframe, key):
    """Likelihood-ratio test for island fixed effect with site random intercept."""
    subset = dataframe[["island", "site", key]].dropna().rename(columns={key: "y"})
    if subset["island"].nunique() < 2 or len(subset) < 8:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = smf.mixedlm("y ~ C(island)", subset, groups=subset["site"]).fit(reml=False)
            null = smf.mixedlm("y ~ 1", subset, groups=subset["site"]).fit(reml=False)
        likelihood_ratio = 2.0 * (full.llf - null.llf)
        degrees = subset["island"].nunique() - 1
        return float(likelihood_ratio), float(stats.chi2.sf(max(likelihood_ratio, 0), degrees))
    except Exception:
        return np.nan, np.nan


def bh_adjust(p_values):
    values = np.array([value if value == value else 1.0 for value in p_values], float)
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count)
    previous = 1.0
    for rank, index in enumerate(order[::-1]):
        previous = min(previous, values[index] * count / (count - rank))
        adjusted[index] = previous
    return adjusted


def main() -> None:
    plants = load_plant_means()
    dataframe = pd.DataFrame(plants)

    columns = ["island", "no", "id", "lat", "n_flowers", "frac_female"] + [
        key for key, _label in TRAITS
    ]
    with (RESULTS / "plant_means.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for plant in plants:
            writer.writerow({
                column: (
                    round(plant[column], 4)
                    if isinstance(plant.get(column), float) and plant[column] == plant[column]
                    else (plant.get(column, "") if plant.get(column) == plant.get(column) else "")
                )
                for column in columns
            })
    print(f"wrote {RESULTS / 'plant_means.csv'}  ({len(plants)} plants)")
    print(
        "plants/island:",
        {LABEL[island]: int((dataframe.island == island).sum()) for island in ISLANDS},
    )
    print(
        "sites/island: ",
        {LABEL[island]: dataframe[dataframe.island == island]["site"].nunique() for island in ISLANDS},
    )

    results, kw_p_values, site_p_values = [], [], []
    for key, label in TRAITS:
        by_island = {
            island: dataframe[dataframe.island == island][key].dropna().values
            for island in ISLANDS
        }
        by_island = {island: values for island, values in by_island.items() if len(values)}
        pst, lower, upper = pst_bootstrap(by_island)
        groups = [values for values in by_island.values() if len(values) >= 2]
        _h, kw_p = stats.kruskal(*groups)
        likelihood_ratio, site_p = site_corrected_p(dataframe, key)
        latitude_data = dataframe[[key, "lat"]].dropna()
        rho, latitude_p = stats.spearmanr(latitude_data["lat"], latitude_data[key])
        results.append({
            "trait": label,
            "key": key,
            "pst": round(pst, 3),
            "pst_lo": round(lower, 3),
            "pst_hi": round(upper, 3),
            "kw_p": kw_p,
            "site_lrt": round(likelihood_ratio, 2) if likelihood_ratio == likelihood_ratio else "",
            "site_p": site_p,
            "lat_rho": round(rho, 2),
            "lat_p": latitude_p,
            **{
                f"mean_{island}": (
                    round(float(np.mean(by_island[island])), 2) if island in by_island else ""
                )
                for island in ISLANDS
            },
        })
        kw_p_values.append(kw_p)
        site_p_values.append(site_p)

    for result, kw_adjusted, site_adjusted in zip(
        results, bh_adjust(kw_p_values), bh_adjust(site_p_values)
    ):
        result["kw_p_adj"] = kw_adjusted
        result["site_p_adj"] = site_adjusted

    fields = [
        "trait", "key", "pst", "pst_lo", "pst_hi", "kw_p", "kw_p_adj",
        "site_lrt", "site_p", "site_p_adj", "lat_rho", "lat_p",
    ] + [f"mean_{island}" for island in ISLANDS]
    with (RESULTS / "island_analysis_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                field: (
                    f"{result[field]:.2e}"
                    if field in ("kw_p", "kw_p_adj", "site_p", "site_p_adj", "lat_p")
                    and result[field] == result[field]
                    else result[field]
                )
                for field in fields
            })
    print(f"wrote {RESULTS / 'island_analysis_stats.csv'}")

    pairwise_rows = []
    for key, label in TRAITS:
        for island_a, island_b in itertools.combinations(ISLANDS, 2):
            values_a = dataframe[dataframe.island == island_a][key].dropna().values
            values_b = dataframe[dataframe.island == island_b][key].dropna().values
            pairwise = (
                pst_value([values_a, values_b])
                if len(values_a) >= 2 and len(values_b) >= 2 else np.nan
            )
            pairwise_rows.append({
                "trait": label,
                "key": key,
                "island_a": LABEL[island_a],
                "island_b": LABEL[island_b],
                "pst": round(pairwise, 3) if pairwise == pairwise else "",
                "n_a": len(values_a),
                "n_b": len(values_b),
            })
    with (RESULTS / "island_pst_pairwise.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["trait", "key", "island_a", "island_b", "pst", "n_a", "n_b"],
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(f"wrote {RESULTS / 'island_pst_pairwise.csv'}")

    print("\n=== among-island analysis (plant means; site-corrected island test) ===")
    print(f"{'trait':30} {'Pst [95% CI]':>18} {'site p(adj)':>12} {'KW p(adj)':>11} {'lat rho':>8}")
    for result in sorted(results, key=lambda row: -row["pst"]):
        site_label = (
            f"{result['site_p_adj']:.1e}"
            if result["site_p_adj"] == result["site_p_adj"] else "n/a"
        )
        significance = (
            "" if result["site_p_adj"] == result["site_p_adj"]
            and result["site_p_adj"] < 0.05 else " n.s."
        )
        print(
            f"{result['trait']:30} {result['pst']:.2f} "
            f"[{result['pst_lo']:.2f},{result['pst_hi']:.2f}]  "
            f"{site_label}{significance:5} {result['kw_p_adj']:.1e}  "
            f"{result['lat_rho']:+.2f}"
        )


if __name__ == "__main__":
    main()
