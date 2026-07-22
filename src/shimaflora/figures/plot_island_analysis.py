#!/usr/bin/env python3
"""Publication figures and table for the final island-divergence analysis.

Outputs:
* ``island_divergence.png``: Pst forest for site-corrected significant traits and
  plant-level latitude plots for the three most-diverged traits.
* ``island_divergence_table.csv/.png``: manuscript table of significant traits.
* ``island_pst_pairwise.png``: pairwise-Pst heatmaps for those traits plus their mean.

Pst is a phenotypic divergence surrogate used to rank traits, not evidence by itself
for selection and not a Qst-Fst test.
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
LABEL = {
    "oshima": "Oshima",
    "toshima": "Toshima",
    "niijima": "Niijima",
    "shikine": "Shikinejima",
    "kozu": "Kozushima",
}
ISLAND_COLOUR = {key: COLOUR[value] for key, value in LABEL.items()}

GROUP = {
    **{
        key: "size"
        for key in [
            "corolla_length_mm",
            "corolla_width_fulleq_mm",
            "corolla_area_fulleq_mm2",
            "throat_width_mm",
            "mouth_width_mm",
            "corolla_aspect_L_W",
            "tube_flare_W_throat",
            "lobe_incision_mm",
        ]
    },
    **{key: "repro" for key in ["organ_length_mm", "organ_corolla_ratio"]},
    **{
        key: "guide"
        for key in ["guide_coverage_pct", "guide_basal_frac", "guide_midline_ratio"]
    },
}
GROUP_COLOUR = {"size": "#2878ff", "repro": "#d1495b", "guide": "#7B2D8E"}
GROUP_NAME = {
    "size": "corolla size/shape",
    "repro": "reproductive organ",
    "guide": "nectar guide",
}

UNIT = {
    "reproductive-organ length": "mm",
    "mouth width": "mm",
    "corolla area": "mm2",
    "corolla length": "mm",
    "corolla width": "mm",
    "guide coverage": "%",
    "throat width": "mm",
}
ISLAND_LABELS = ["Oshima", "Toshima", "Niijima", "Shikinejima", "Kozushima"]
ISLAND_MEAN_KEYS = [
    "mean_oshima",
    "mean_toshima",
    "mean_niijima",
    "mean_shikine",
    "mean_kozu",
]


def significant_traits(statistics):
    return [
        row for row in statistics
        if row.get("site_p_adj", "") not in ("", None)
        and float(row["site_p_adj"]) < 0.05
    ]


def make_table(statistics):
    """Write the manuscript table of site-corrected significant traits."""
    significant = significant_traits(statistics)
    significant.sort(key=lambda row: -float(row["pst"]))

    fields = ["trait", "unit", "pst", "pst_ci", "site_p_adj", "lat_rho"] + ISLAND_LABELS
    with (RESULTS / "island_divergence_table.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for row in significant:
            writer.writerow([
                row["trait"],
                UNIT.get(row["trait"], ""),
                row["pst"],
                f"{row['pst_lo']}-{row['pst_hi']}",
                row["site_p_adj"],
                row["lat_rho"],
            ] + [row[key] for key in ISLAND_MEAN_KEYS])
    print(f"wrote {RESULTS / 'island_divergence_table.csv'}")

    header = ["Trait (unit)", "Pst", "95% CI", "p (site-corr.)", "lat rho"] + ISLAND_LABELS
    cells, colours = [], []
    for row in significant:
        cells.append([
            f"{row['trait']} ({UNIT.get(row['trait'], '')})",
            row["pst"],
            f"{row['pst_lo']}-{row['pst_hi']}",
            f"{float(row['site_p_adj']):.1e}",
            f"{float(row['lat_rho']):+.2f}",
        ] + [row[key] for key in ISLAND_MEAN_KEYS])
        colours.append(GROUP_COLOUR[GROUP[row["key"]]])

    fig, ax = plt.subplots(figsize=(12.4, 0.52 * (len(significant) + 1) + 0.8))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    n_columns = len(header)
    widths = [0.185, 0.058, 0.10, 0.10, 0.072] + [0.097] * 5
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_width(widths[column_index])
        cell.set_edgecolor("#dddddd")
        if row_index == 0:
            cell.set_facecolor("#33324a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if column_index == 0:
                cell.set_text_props(
                    color=colours[row_index - 1], fontweight="bold", ha="left"
                )
                cell.PAD = 0.03
            if column_index >= n_columns - 5:
                cell.set_facecolor("#f4f4fa")
    ax.set_title(
        "Table 1  Floral traits that diverge significantly among Izu islands "
        "(site-corrected, BH p < 0.05; 125 plants)",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.text(
        0,
        -0.04,
        "Pst = phenotypic among-island divergence (not Qst-Fst); lat rho = Spearman "
        "correlation with latitude; last five columns are island means.",
        transform=ax.transAxes,
        fontsize=7.6,
        color=MUTED,
    )
    fig.savefig(RESULTS / "island_divergence_table.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS / 'island_divergence_table.png'}")


def make_pairwise(statistics):
    """Plot pairwise Pst for each significant trait and their mean."""
    from matplotlib.colors import LinearSegmentedColormap

    pairwise = list(
        csv.DictReader((RESULTS / "island_pst_pairwise.csv").open(encoding="utf-8-sig"))
    )
    significant = significant_traits(statistics)
    significant.sort(key=lambda row: -float(row["pst"]))
    order = ISLAND_LABELS
    short = ["Osh", "Tos", "Nii", "Shi", "Koz"]
    index = {name: i for i, name in enumerate(order)}

    def matrix(key):
        values = np.full((5, 5), np.nan)
        for row in pairwise:
            if row["key"] == key and row["pst"] != "":
                i, j = index[row["island_a"]], index[row["island_b"]]
                values[max(i, j), min(i, j)] = float(row["pst"])
        return values

    matrices = [(row["trait"], matrix(row["key"])) for row in significant]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_matrix = np.nanmean(np.dstack([values for _name, values in matrices]), axis=2)
    panels = matrices + [(f"MEAN ({len(matrices)} traits)", mean_matrix)]
    maximum = max(np.nanmax(values) for _name, values in panels)
    colour_map = LinearSegmentedColormap.from_list(
        "pst", ["#f7f4fb", "#c9a5d6", "#7B2D8E", "#2a0a33"]
    )
    colour_map.set_bad("white")

    n_columns = 4
    n_rows = int(np.ceil(len(panels) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3.05 * n_columns, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    image = None
    for axis, (name, values) in zip(axes, panels):
        axis.axis("on")
        image = axis.imshow(values, cmap=colour_map, vmin=0, vmax=maximum)
        for i in range(5):
            for j in range(5):
                if not np.isnan(values[i, j]):
                    axis.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if values[i, j] > maximum * 0.55 else INK,
                    )
        axis.set_xticks(range(5))
        axis.set_yticks(range(5))
        axis.set_xticklabels(short, fontsize=7.5)
        axis.set_yticklabels(short, fontsize=7.5)
        axis.set_title(name, fontsize=9.5, fontweight="bold", pad=4)
        for spine in ("top", "right", "left", "bottom"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(length=0)
    fig.suptitle(
        "Pairwise Pst between islands (site-corrected significant traits; N to S order)",
        fontsize=12.5,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    colour_bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.015, pad=0.02)
    colour_bar.set_label("pairwise Pst", fontsize=8)
    fig.text(
        0.02,
        0.005,
        "Each cell is Pst calculated from plant means for that island pair alone. "
        "Higher values indicate stronger phenotypic differentiation.",
        fontsize=7.4,
        color=MUTED,
        ha="left",
    )
    out = RESULTS / "island_pst_pairwise.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    statistics = list(
        csv.DictReader((RESULTS / "island_analysis_stats.csv").open(encoding="utf-8-sig"))
    )
    plants = list(csv.DictReader((RESULTS / "plant_means.csv").open(encoding="utf-8-sig")))
    significant = significant_traits(statistics)
    significant.sort(key=lambda row: float(row["pst"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.6, 6.6))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.25, 1],
        wspace=0.28,
        hspace=0.55,
        left=0.16,
        right=0.985,
        top=0.9,
        bottom=0.09,
    )

    forest = fig.add_subplot(grid[:, 0])
    y_positions = np.arange(len(significant))
    for index, row in enumerate(significant):
        colour = GROUP_COLOUR[GROUP[row["key"]]]
        lower, upper, value = float(row["pst_lo"]), float(row["pst_hi"]), float(row["pst"])
        forest.plot(
            [lower, upper],
            [index, index],
            color=colour,
            lw=2.4,
            alpha=0.6,
            solid_capstyle="round",
        )
        forest.scatter(
            [value], [index], s=52, color=colour, edgecolor="white", linewidth=0.8, zorder=3
        )
        forest.text(upper + 0.012, index, f"{value:.2f}", va="center", fontsize=8, color=MUTED)
    forest.set_yticks(y_positions)
    forest.set_yticklabels([row["trait"] for row in significant], fontsize=8.8)
    forest.set_xlim(0, max(float(row["pst_hi"]) for row in significant) * 1.12)
    forest.set_xlabel("Pst (plant-level phenotypic divergence; 95% bootstrap CI)")
    forest.set_title(
        "A  Traits that diverge significantly among islands\n"
        "    (site-corrected mixed model, BH p < 0.05)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    forest.xaxis.grid(True, color=GRID, lw=0.8)
    forest.set_axisbelow(True)
    for spine in ("top", "right"):
        forest.spines[spine].set_visible(False)
    forest.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GROUP_COLOUR[group],
                markeredgecolor="white",
                markersize=9,
                label=GROUP_NAME[group],
            )
            for group in ("size", "repro", "guide")
        ],
        fontsize=8,
        loc="lower right",
        frameon=False,
        title="trait group",
        title_fontsize=8.5,
    )

    top_three = [
        row["key"] for row in sorted(significant, key=lambda item: -float(item["pst"]))[:3]
    ]
    names = {row["key"]: row["trait"] for row in significant}
    for plot_row, key in enumerate(top_three):
        axis = fig.add_subplot(grid[plot_row, 1])
        if plot_row == 0:
            axis.set_title(
                "B  Latitudinal cline (top-3 diverged traits)",
                fontsize=11,
                fontweight="bold",
                loc="left",
                pad=8,
            )
        x_values, y_values, colours = [], [], []
        for plant in plants:
            if plant[key] in ("", "nan"):
                continue
            x_values.append(float(plant["lat"]))
            y_values.append(float(plant[key]))
            colours.append(ISLAND_COLOUR[plant["island"]])
        x_values, y_values = np.array(x_values), np.array(y_values)
        axis.scatter(
            x_values,
            y_values,
            s=20,
            c=colours,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_range = np.array([x_values.min(), x_values.max()])
        axis.plot(x_range, slope * x_range + intercept, color=INK, lw=1.3, ls="--", zorder=2)
        rho, _ = stats.spearmanr(x_values, y_values)
        axis.text(
            0.03,
            0.95,
            f"{names[key]}\nrho={rho:+.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.2,
            color=INK,
        )
        axis.set_ylabel(
            "mm" if key.endswith("mm") else ("mm2" if "area" in key else "%"),
            fontsize=8,
        )
        if plot_row == 2:
            axis.set_xlabel("Latitude (degN)   S <-- --> N")
        axis.grid(True, color=GRID, lw=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(labelsize=8)

    fig.suptitle(
        "Among-island divergence of floral traits - Izu-island Campanula microdonta "
        "(125 plants)",
        x=0.16,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.16,
        0.02,
        "Plant means; island test = mixed model with site as random effect. "
        "Pst = Vb/(Vb+2Vw), a phenotypic surrogate, not Qst-Fst.",
        fontsize=7.3,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.16,
        0.005,
        "Guide spot counts, density and dried-specimen colour values are excluded; "
        "area coverage and colour-free spatial structure are retained.",
        fontsize=7.3,
        color=MUTED,
        ha="left",
    )

    out = RESULTS / "island_divergence.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    make_table(statistics)
    make_pairwise(statistics)


if __name__ == "__main__":
    main()
