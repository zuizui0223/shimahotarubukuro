#!/usr/bin/env python3
"""Visualise colour-free nectar-guide spatial structure against an ROI null.

Panel A shows the canonical area-density map. Panel B compares along-length position
with random pixels from the same corollas. Panel C compares distance to the petal
midline in the distal half. These figures establish non-random guide placement; they
do not test a Bombus-presence contrast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from plot_island_traits import INK, MUTED, GRID  # noqa: E402

GUIDE = "#7B2D8E"
RANDOM = "#9AA0A6"


def main() -> None:
    data = np.load(Path("results_shimask_all/guide_spatial.npz"))
    density = data["dens_all"]

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
    colour_map = LinearSegmentedColormap.from_list(
        "guide", ["#f7f4fb", "#c9a5d6", GUIDE, "#2a0a33"]
    )
    colour_map.set_bad("white")

    fig = plt.figure(figsize=(13.6, 5.6))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.9, 1.15, 1.15],
        wspace=0.42,
        left=0.045,
        right=0.985,
        top=0.84,
        bottom=0.15,
    )

    density_axis = fig.add_subplot(grid[0, 0])
    maximum = np.nanpercentile(density, 99)
    image = density_axis.imshow(
        density, cmap=colour_map, vmin=0, vmax=maximum, aspect="auto"
    )
    density_axis.set_title(
        "A  Average guide-area density", fontsize=11, fontweight="bold", loc="left", pad=8
    )
    density_axis.set_xticks([])
    density_axis.set_yticks([0, density.shape[0] - 1])
    density_axis.set_yticklabels(["base", "lobe tips"], fontsize=8)
    density_axis.text(
        0.5,
        -0.06,
        "all guided corollas",
        transform=density_axis.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    colour_bar = fig.colorbar(image, ax=density_axis, fraction=0.045, pad=0.02)
    colour_bar.ax.tick_params(labelsize=7)

    position_axis = fig.add_subplot(grid[0, 1])
    bins = np.linspace(0, 1, 26)
    for values, colour, label in (
        (data["g_pos"], GUIDE, "guide pixels"),
        (data["r_pos"], RANDOM, "random ROI pixels"),
    ):
        histogram, _ = np.histogram(values, bins=bins, density=True)
        centres = (bins[:-1] + bins[1:]) / 2
        position_axis.plot(centres, histogram, color=colour, lw=2.2, label=label)
        position_axis.fill_between(centres, histogram, color=colour, alpha=0.12)
    position_axis.set_xlabel("Position along corolla (0 = base, 1 = lobe tips)")
    position_axis.set_ylabel("Density")
    position_axis.set_title(
        "B  Basal concentration", fontsize=11, fontweight="bold", loc="left", pad=8
    )
    position_axis.axvspan(0, 0.33, color=INK, alpha=0.04)
    position_axis.legend(fontsize=8.5, frameon=False, loc="upper right")
    position_axis.grid(True, color=GRID, lw=0.8)
    position_axis.set_axisbelow(True)
    for spine in ("top", "right"):
        position_axis.spines[spine].set_visible(False)

    midline_axis = fig.add_subplot(grid[0, 2])
    midline_bins = np.linspace(0, 1, 26)
    for values, colour, label in (
        (data["g_mid_distal"], GUIDE, "guide pixels"),
        (data["r_mid_distal"], RANDOM, "random ROI pixels"),
    ):
        histogram, _ = np.histogram(values, bins=midline_bins, density=True)
        centres = (midline_bins[:-1] + midline_bins[1:]) / 2
        midline_axis.plot(centres, histogram, color=colour, lw=2.2, label=label)
        midline_axis.fill_between(centres, histogram, color=colour, alpha=0.12)
    guide_mean = data["g_mid_distal"].mean()
    random_mean = data["r_mid_distal"].mean()
    midline_axis.axvline(guide_mean, color=GUIDE, ls="--", lw=1.2)
    midline_axis.axvline(random_mean, color=RANDOM, ls="--", lw=1.2)
    midline_axis.set_xlabel("Distance to petal midline (0 = midline, 1 = edge)")
    midline_axis.set_ylabel("Density")
    midline_axis.set_title(
        "C  Distal petal-midline concentration",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    midline_axis.text(
        0.42,
        0.60,
        f"guide mean {guide_mean:.2f}\nrandom {random_mean:.2f}",
        transform=midline_axis.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        color=INK,
    )
    midline_axis.legend(fontsize=8.5, frameon=False, loc="upper right")
    midline_axis.grid(True, color=GRID, lw=0.8)
    midline_axis.set_axisbelow(True)
    for spine in ("top", "right"):
        midline_axis.spines[spine].set_visible(False)

    fig.suptitle(
        "Nectar-guide area is non-randomly distributed within the reviewed corolla ROI",
        x=0.045,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.015,
        "Each guided corolla is warped to a common frame and compared with random pixels "
        "from the same ROI. The test concerns spatial placement, not dried-specimen colour.",
        fontsize=7.5,
        color=MUTED,
        ha="left",
    )

    out = Path("results_shimask_all/guide_spatial_structure.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
