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
import warnings
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
    **{k: "guide" for k in ["guide_coverage_pct", "guide_basal_frac", "guide_midline_ratio"]},
}
GCOL = {"size": "#2878ff", "repro": "#d1495b", "guide": "#7B2D8E"}
GNAME = {"size": "corolla size/shape", "repro": "reproductive organ", "guide": "nectar guide"}


UNIT = {"organ/style length": "mm", "mouth width": "mm", "corolla area": "mm2",
        "corolla length": "mm", "corolla width": "mm", "guide coverage": "%",
        "throat width": "mm"}
ILAB = ["Oshima", "Toshima", "Niijima", "Shikinejima", "Kozushima"]
IKEY = ["mean_oshima", "mean_toshima", "mean_niijima", "mean_shikine", "mean_kozu"]


def make_table(stat_all):
    """Paper Table 1: the significantly-diverged traits, as a CSV and a rendered PNG."""
    sig = [r for r in stat_all if r.get("site_p_adj", "") not in ("", None)
           and float(r["site_p_adj"]) < 0.05]
    sig.sort(key=lambda r: -float(r["pst"]))

    # CSV
    fields = ["trait", "unit", "pst", "pst_ci", "site_p_adj", "lat_rho"] + ILAB
    with (RESULTS / "island_divergence_table.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in sig:
            w.writerow([r["trait"], UNIT.get(r["trait"], ""), r["pst"],
                        f"{r['pst_lo']}-{r['pst_hi']}", r["site_p_adj"], r["lat_rho"]]
                       + [r[k] for k in IKEY])
    print(f"wrote {RESULTS/'island_divergence_table.csv'}")

    # rendered PNG
    header = ["Trait (unit)", "Pst", "95% CI", "p (site-corr.)", "lat rho"] + ILAB
    cells, colours = [], []
    for r in sig:
        cells.append([f"{r['trait']} ({UNIT.get(r['trait'],'')})", r["pst"],
                      f"{r['pst_lo']}-{r['pst_hi']}", f"{float(r['site_p_adj']):.1e}",
                      f"{float(r['lat_rho']):+.2f}"] + [r[k] for k in IKEY])
        g = GROUP[r["key"]]
        colours.append(GCOL.get(g if g in GCOL else "size", "#2878ff"))

    fig, ax = plt.subplots(figsize=(12.4, 0.52 * (len(sig) + 1) + 0.8))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ncol = len(header)
    widths = [0.185, 0.058, 0.10, 0.10, 0.072] + [0.097] * 5  # wide trait column
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(widths[col])
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#33324a"); cell.set_text_props(color="white", fontweight="bold")
        else:
            if col == 0:
                cell.set_text_props(color=colours[row - 1], fontweight="bold", ha="left")
                cell.PAD = 0.03
            if col >= ncol - 5:  # island-mean columns, light shade
                cell.set_facecolor("#f4f4fa")
    ax.set_title("Table 1  Floral traits that diverge significantly among Izu islands "
                 "(site-corrected, BH p < 0.05; 125 plants)",
                 fontsize=11.5, fontweight="bold", loc="left", pad=12)
    ax.text(0, -0.04, "Pst = among-island divergence (phenotypic surrogate, not Qst-Fst); lat rho = Spearman "
            "with latitude; last five columns = island means (mm, mm2 or %).",
            transform=ax.transAxes, fontsize=7.6, color=MUTED)
    fig.savefig(RESULTS / "island_divergence_table.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS/'island_divergence_table.png'}")


def make_pairwise(stat_all):
    """Grid of pairwise-Pst heatmaps for the significantly-diverged traits (+ mean)."""
    from matplotlib.colors import LinearSegmentedColormap
    pw = list(csv.DictReader((RESULTS / "island_pst_pairwise.csv").open(encoding="utf-8-sig")))
    sig = [r for r in stat_all if r.get("site_p_adj", "") not in ("", None)
           and float(r["site_p_adj"]) < 0.05]
    sig.sort(key=lambda r: -float(r["pst"]))
    order = ["Oshima", "Toshima", "Niijima", "Shikinejima", "Kozushima"]
    short = ["Osh", "Tos", "Nii", "Shi", "Koz"]
    idx = {n: i for i, n in enumerate(order)}

    def matrix(key):
        m = np.full((5, 5), np.nan)
        for r in pw:
            if r["key"] == key and r["pst"] != "":
                i, j = idx[r["island_a"]], idx[r["island_b"]]
                m[max(i, j), min(i, j)] = float(r["pst"])  # lower triangle
        return m

    mats = [(r["trait"], matrix(r["key"])) for r in sig]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_m = np.nanmean(np.dstack([m for _, m in mats]), axis=2)
    panels = mats + [("MEAN (7 traits)", mean_m)]
    vmax = max(np.nanmax(m) for _, m in panels)
    cmap = LinearSegmentedColormap.from_list("pst", ["#f7f4fb", "#c9a5d6", "#7B2D8E", "#2a0a33"])
    cmap.set_bad("white")

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    im = None
    for ax, (name, m) in zip(axes, panels):
        ax.axis("on")
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(5):
            for j in range(5):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if m[i, j] > vmax * 0.55 else INK)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(short, fontsize=7.5); ax.set_yticklabels(short, fontsize=7.5)
        ax.set_title(name, fontsize=9.5, fontweight="bold", pad=4)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Pairwise Pst between islands (per significant trait; islands ordered N->S)",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02)
    cb.set_label("pairwise Pst", fontsize=8)
    fig.text(0.02, 0.005, "Each cell = Pst for that island pair alone (plant means). Higher = more "
             "differentiated; note the N-S extremes (Oshima vs Kozushima) are largest.",
             fontsize=7.4, color=MUTED, ha="left")
    out = RESULTS / "island_pst_pairwise.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


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

    make_table(stat_all)
    make_pairwise(stat_all)


if __name__ == "__main__":
    main()
