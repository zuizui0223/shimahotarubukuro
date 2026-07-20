#!/usr/bin/env python3
"""Among-island divergence figure: Pst forest + latitudinal clines.

Panel A: Pst (plant-level) for every trait with its 95 % bootstrap CI, sorted, and
coloured by trait group (size / reproductive / nectar-guide). Pst is a phenotypic
surrogate for Qst - it ranks how strongly traits diverge among islands, not proof of
selection.
Panel B: the three most-diverged traits against latitude (one point per plant,
coloured by island), showing the north-south cline (Oshima largest).

Reads results_shimask_all/island_analysis_stats.csv and plant_means.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from plot_island_traits import COLOUR, INK, MUTED, GRID  # noqa: E402

RESULTS = Path("results_shimask_all")
ISLANDS = ["oshima", "toshima", "niijima", "shikine", "kozu"]
LABEL = {"oshima": "Oshima", "toshima": "Toshima", "niijima": "Niijima",
         "shikine": "Shikinejima", "kozu": "Kozushima"}
ICOL = {i: COLOUR[LABEL[i]] for i in ISLANDS}

GROUP = {  # trait key -> group
    **{k: "size" for k in ["corolla_length_mm", "corolla_width_fulleq_mm", "corolla_area_fulleq_mm2",
                           "throat_width_mm", "mouth_width_mm", "corolla_aspect_L_W",
                           "tube_flare_W_throat", "lobe_incision_mm"]},
    **{k: "repro" for k in ["style_length_mm", "style_corolla_ratio"]},
    **{k: "guide" for k in ["guide_coverage_pct", "n_guide_spots", "guide_density_per_cm2",
                            "guide_basal_frac", "guide_midline_ratio"]},
}
GCOL = {"size": "#2878ff", "repro": "#d1495b", "guide": "#7B2D8E"}
GNAME = {"size": "corolla size/shape", "repro": "reproductive organ", "guide": "nectar guide"}


def main() -> None:
    stat_all = list(csv.DictReader((RESULTS / "island_analysis_stats.csv").open(encoding="utf-8-sig")))
    plants = list(csv.DictReader((RESULTS / "plant_means.csv").open(encoding="utf-8-sig")))
    # keep only traits that diverge significantly among islands after site correction
    stat = [r for r in stat_all if r.get("site_p_adj", "") not in ("", None)
            and float(r["site_p_adj"]) < 0.05]
    stat.sort(key=lambda r: float(r["pst"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], wspace=0.28, hspace=0.55,
                          left=0.16, right=0.985, top=0.9, bottom=0.09)

    # -- A: Pst forest --
    axa = fig.add_subplot(gs[:, 0])
    y = np.arange(len(stat))
    for i, r in enumerate(stat):
        c = GCOL[GROUP[r["key"]]]
        lo, hi, p = float(r["pst_lo"]), float(r["pst_hi"]), float(r["pst"])
        axa.plot([lo, hi], [i, i], color=c, lw=2.4, alpha=0.6, solid_capstyle="round")
        axa.scatter([p], [i], s=52, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        axa.text(hi + 0.012, i, f"{p:.2f}", va="center", fontsize=8, color=MUTED)
    axa.set_yticks(y)
    axa.set_yticklabels([r["trait"] for r in stat], fontsize=8.8)
    axa.set_xlim(0, max(float(r["pst_hi"]) for r in stat) * 1.12)
    axa.set_xlabel("Pst  (among-island divergence, plant-level; 95% bootstrap CI)")
    axa.set_title("A  Traits that diverge significantly among islands\n"
                  "    (site-corrected mixed model, BH p < 0.05)", fontsize=10.5,
                  fontweight="bold", loc="left", pad=8)
    axa.xaxis.grid(True, color=GRID, lw=0.8)
    axa.set_axisbelow(True)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=GCOL[g],
                               markeredgecolor="white", markersize=9, label=GNAME[g])
                        for g in ("size", "repro", "guide")],
               fontsize=8, loc="lower right", frameon=False, title="trait group", title_fontsize=8.5)

    # -- B: latitudinal clines for the 3 top-Pst traits --
    top3 = [r["key"] for r in sorted(stat, key=lambda r: -float(r["pst"]))[:3]]
    names = {r["key"]: r["trait"] for r in stat}
    for row, key in enumerate(top3):
        ax = fig.add_subplot(gs[row, 1])
        if row == 0:
            ax.set_title("B  Latitudinal cline (top-3 diverged traits)", fontsize=11,
                         fontweight="bold", loc="left", pad=8)
        xs, ys, cs = [], [], []
        for p in plants:
            if p[key] in ("", "nan"):
                continue
            xs.append(float(p["lat"])); ys.append(float(p[key])); cs.append(ICOL[p["island"]])
        xs, ys = np.array(xs), np.array(ys)
        ax.scatter(xs, ys, s=20, c=cs, edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3)
        a, b = np.polyfit(xs, ys, 1)
        xr = np.array([xs.min(), xs.max()])
        ax.plot(xr, a * xr + b, color=INK, lw=1.3, ls="--", zorder=2)
        rho, _ = stats.spearmanr(xs, ys)
        ax.text(0.03, 0.95, f"{names[key]}\nrho={rho:+.2f}", transform=ax.transAxes,
                va="top", fontsize=8.2, color=INK)
        ax.set_ylabel("mm" if key.endswith("mm") else ("mm2" if "area" in key else "%"),
                      fontsize=8)
        if row == 2:
            ax.set_xlabel("Latitude (degN)   S <-- --> N")
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle("Among-island divergence of floral traits - Izu-island Campanula microdonta "
                 "(125 plants)", x=0.16, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.16, 0.02, "Plant means; island test = mixed model with site as random effect (uneven sites "
             "corrected). Pst = Vb/(Vb+2Vw), a phenotypic surrogate, not Qst-Fst.", fontsize=7.3,
             color=MUTED, ha="left")
    fig.text(0.16, 0.005, "Guide colour/contrast excluded (unreliable on dried specimens); guide spatial "
             "structure (basal / petal-midline concentration) included.", fontsize=7.3, color=MUTED, ha="left")

    out = RESULTS / "island_divergence.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
