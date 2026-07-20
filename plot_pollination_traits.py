#!/usr/bin/env python3
"""Island comparison of the pollination-relevant corolla morphometrics.

The traits parallel those Nagano et al. (2014) used to link Campanula punctata to
its bumblebee: throat (tube) width and distal mouth width (corolla-entrance proxies),
lobe-incision depth, corolla aspect and tube flare (shape), and the style/corolla
ratio (how far the sexual column reaches toward the bell mouth). Same CVD-safe
five-island palette as the other figures; Oshima (the one island with Bombus) is
ringed so the Bombus contrast is visible at a glance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_island_traits import ORDER, COLOUR, INK, MUTED, GRID, island_of  # noqa: E402

BOMBUS = {"Oshima"}
PANELS = [
    ("throat_width_mm", "Throat (tube) width (mm)", "A  Throat width  (CE proxy, proximal)"),
    ("mouth_width_mm", "Distal mouth width (mm)", "B  Mouth width  (CE proxy, distal rim)"),
    ("lobe_incision_mm", "Lobe-incision depth (mm)", "C  Lobe incision"),
    ("corolla_aspect_L_W", "Corolla length / width", "D  Corolla aspect"),
    ("tube_flare_W_throat", "Width / throat", "E  Tube flare"),
    ("style_corolla_ratio", "Style length / corolla length", "F  Style reach"),
]


def box_strip(ax, data, key, ylabel, title):
    jitter = np.random.RandomState(0)
    for j, isl in enumerate(ORDER):
        v = np.array([float(r[key]) for r in data[isl] if r[key] not in ("", "nan")])
        if v.size == 0:
            continue
        c = COLOUR[isl]
        ax.boxplot(v, positions=[j], widths=0.56, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.0),
                   capprops=dict(color=MUTED, lw=1.0),
                   boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.4))
        ax.scatter(jitter.normal(j, 0.075, len(v)), v, s=13, facecolor=c,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        ax.text(j, v.max() + (v.max() - v.min() + 1e-6) * 0.06, f"{v.mean():.1f}",
                ha="center", va="bottom", fontsize=8, color=c, fontweight="bold")
        if isl in BOMBUS:  # ring the Bombus-present island
            ax.scatter([j], [v.mean()], s=190, facecolor="none", edgecolor=INK,
                       linewidth=1.6, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([i + (" *" if i in BOMBUS else "") for i in ORDER],
                       rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results_shimask_all/pollination_traits.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_shimask_all/island_pollination_traits.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig")))
    data = {i: [] for i in ORDER}
    for r in rows:
        data[island_of(r["sheet"])].append(r)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.09, wspace=0.26, hspace=0.42)
    for ax, (key, ylabel, title) in zip(axes.ravel(), PANELS):
        box_strip(ax, data, key, ylabel, title)

    fig.suptitle("Pollination-relevant corolla morphometrics across Izu-island "
                 "Campanula microdonta  (n = %d corollas)" % len(rows),
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.015, "Traits after Nagano et al. 2014 (Ecol. Evol., doi:10.1002/ece3.1191). "
             "Transverse widths are full-flower-equivalent. * = Oshima, the only island with a "
             "bumblebee (Bombus ardens). Box = median/IQR, whiskers 1.5xIQR.",
             fontsize=7.5, color=MUTED, ha="left")

    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
