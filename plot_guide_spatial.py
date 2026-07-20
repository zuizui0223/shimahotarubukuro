#!/usr/bin/env python3
"""Visualise the non-random spatial structure of the nectar guides.

A  Canonical guide-density map: every corolla warped to a common frame (base at top,
   lobe tips at bottom) and the guide density averaged. Bright = where guides sit on
   the average corolla. The base band and the radiating petal-midline streaks light
   up, not the whole corolla.
B  Basal profile: guide vs random density along the corolla length. Guides pile up
   near the base (the nectar), random pixels are flat.
C  Distal-half midline score (0 = on the petal midline, ->1 = petal edge): guides are
   shifted toward the midlines relative to random, i.e. they track the petal midribs
   where they fan out - the directional cue that would align a bumblebee's approach.

Reads results_shimask_all/guide_spatial.npz (from guide_spatial.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from plot_island_traits import INK, MUTED, GRID  # noqa: E402

GUIDE = "#7B2D8E"   # purple, the guide colour
RANDOM = "#9AA0A6"  # neutral grey for the null


def main() -> None:
    z = np.load(Path("results_shimask_all/guide_spatial.npz"))
    dens = z["dens_all"]
    dens_osh = z["dens_Oshima"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    cmap = LinearSegmentedColormap.from_list("guide", ["#f7f4fb", "#c9a5d6", GUIDE, "#2a0a33"])
    cmap.set_bad("white")

    fig = plt.figure(figsize=(13.6, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 1.15], wspace=0.42,
                          left=0.045, right=0.985, top=0.84, bottom=0.15)

    # -- A: canonical density map (all + Oshima inset-style side by side) --
    axa = fig.add_subplot(gs[0, 0])
    vmax = np.nanpercentile(dens, 99)
    im = axa.imshow(dens, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    axa.set_title("A  Average guide density", fontsize=11, fontweight="bold", loc="left", pad=8)
    axa.set_xticks([]); axa.set_yticks([0, dens.shape[0] - 1])
    axa.set_yticklabels(["base", "lobe tips"], fontsize=8)
    axa.text(0.5, -0.06, "all guided corollas (n=182)", transform=axa.transAxes,
             ha="center", va="top", fontsize=8, color=MUTED)
    cb = fig.colorbar(im, ax=axa, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    # -- B: basal profile --
    axb = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 26)
    for arr, c, lab in ((z["g_pos"], GUIDE, "nectar-guide pixels"),
                        (z["r_pos"], RANDOM, "random ROI pixels (null)")):
        h, _ = np.histogram(arr, bins=bins, density=True)
        axb.plot((bins[:-1] + bins[1:]) / 2, h, color=c, lw=2.2, label=lab)
        axb.fill_between((bins[:-1] + bins[1:]) / 2, h, color=c, alpha=0.12)
    axb.set_xlabel("Position along corolla length  (0 = base, 1 = lobe tips)")
    axb.set_ylabel("Density")
    axb.set_title("B  Basal concentration", fontsize=11, fontweight="bold", loc="left", pad=8)
    axb.axvspan(0, 0.33, color=INK, alpha=0.04)
    axb.text(0.16, axb.get_ylim()[1] * 0.96, "proximal third", ha="center", va="top",
             fontsize=8, color=MUTED)
    axb.legend(fontsize=8.5, frameon=False, loc="upper right")
    axb.grid(True, color=GRID, lw=0.8); axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # -- C: distal midline score --
    axc = fig.add_subplot(gs[0, 2])
    bins2 = np.linspace(0, 1, 26)
    for arr, c, lab in ((z["g_mid_distal"], GUIDE, "nectar-guide pixels"),
                        (z["r_mid_distal"], RANDOM, "random ROI pixels (null)")):
        h, _ = np.histogram(arr, bins=bins2, density=True)
        axc.plot((bins2[:-1] + bins2[1:]) / 2, h, color=c, lw=2.2, label=lab)
        axc.fill_between((bins2[:-1] + bins2[1:]) / 2, h, color=c, alpha=0.12)
    gm, rm_ = z["g_mid_distal"].mean(), z["r_mid_distal"].mean()
    axc.axvline(gm, color=GUIDE, ls="--", lw=1.2)
    axc.axvline(rm_, color=RANDOM, ls="--", lw=1.2)
    axc.set_xlabel("Distance to petal midline  (0 = midrib, 1 = petal edge)")
    axc.set_ylabel("Density")
    axc.set_title("C  Petal-midline concentration (distal half)", fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axc.text(0.42, 0.60, f"guide mean {gm:.2f}\n< random {rm_:.2f}", transform=axc.transAxes,
             fontsize=8.5, va="top", ha="left", color=INK)
    axc.legend(fontsize=8.5, frameon=False, loc="upper right")
    axc.grid(True, color=GRID, lw=0.8); axc.set_axisbelow(True)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    fig.suptitle("Nectar-guide spots are non-random: concentrated at the base and along the petal midlines",
                 x=0.045, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.045, 0.015,
             "Each corolla warped to a common frame; guide pixels compared with random ROI pixels (complete "
             "spatial randomness null). Both patterns pooled p < 1e-300. Structure toward the basal nectar and "
             "along each petal axis is the signature of a functional nectar guide, not scattered pigment.",
             fontsize=7.5, color=MUTED, ha="left")

    out = Path("results_shimask_all/guide_spatial_structure.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
