#!/usr/bin/env python3
"""Per-island average nectar-guide density maps, Bombus present vs absent.

Every guided corolla on an island is warped to the common frame (base at top, lobe
tips at bottom) and its guide density averaged, exactly as in the pooled map. Placed
side by side the islands show whether the guide is not only larger but more sharply
structured where the bumblebee (Bombus ardens, Oshima only) still reads it.

Shikinejima has just one guided corolla, so its panel is a single flower, not an
average - it is drawn but flagged. Reads results_shimask_all/guide_spatial.npz.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from plot_island_traits import ORDER, INK, MUTED  # noqa: E402

GUIDE = "#7B2D8E"


def main() -> None:
    z = np.load(Path("results_shimask_all/guide_spatial.npz"))
    counts = Counter(r["island"] for r in
                     csv.DictReader(Path("results_shimask_all/guide_spatial.csv").open(encoding="utf-8-sig")))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    cmap = LinearSegmentedColormap.from_list("guide", ["#f7f4fb", "#c9a5d6", GUIDE, "#2a0a33"])
    cmap.set_bad("white")

    # Shared scale from the well-sampled islands so panels are comparable.
    maps = {i: z[f"dens_{i}"] for i in ORDER}
    vmax = np.nanpercentile(np.concatenate([maps[i][np.isfinite(maps[i])]
                                            for i in ORDER if counts.get(i, 0) >= 10]), 99)

    fig, axes = plt.subplots(1, len(ORDER), figsize=(14.5, 5.2))
    fig.subplots_adjust(left=0.035, right=0.9, top=0.8, bottom=0.09, wspace=0.18)
    for ax, isl in zip(axes, ORDER):
        im = ax.imshow(maps[isl], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        n = counts.get(isl, 0)
        bombus = " *" if isl == "Oshima" else ""
        ax.set_title(f"{isl}{bombus}", fontsize=12, fontweight="bold", pad=6,
                     color=INK if n >= 10 else MUTED)
        tag = f"n={n}" + ("  (single flower)" if n == 1 else "")
        ax.text(0.5, -0.05, tag, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=MUTED)
        ax.set_xticks([])
        if ax is axes[0]:
            ax.set_yticks([0, maps[isl].shape[0] - 1])
            ax.set_yticklabels(["base", "lobe tips"], fontsize=8)
        else:
            ax.set_yticks([])

    cax = fig.add_axes([0.915, 0.15, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("average guide density", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Nectar-guide density maps across Izu islands  -  Bombus present (Oshima *) vs bumblebee-free",
                 x=0.035, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.035, 0.025, "Every guided corolla warped to the common frame and averaged; shared colour scale "
             "(from islands with n>=10). * = the only island with a bumblebee (Bombus ardens).",
             fontsize=7.5, color=MUTED, ha="left")

    out = Path("results_shimask_all/guide_density_islands.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
