#!/usr/bin/env python3
"""Island comparison of nectar-guide traits from the final corolla table.

Draws nectar-guide coverage and spot count per island (box + strip) and the
guide-present fraction per island (bar). Guide-less corollas (mostly Shikinejima)
sit at zero. Same CVD-safe five-island palette as plot_island_traits.py.
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


def _box_strip(ax, data, key, ylabel, title, cast=float):
    jitter = np.random.RandomState(0)
    for j, isl in enumerate(ORDER):
        v = np.array([cast(r[key]) for r in data[isl] if r[key] != ""])
        c = COLOUR[isl]
        ax.boxplot(v, positions=[j], widths=0.56, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.0),
                   capprops=dict(color=MUTED, lw=1.0),
                   boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.4))
        ax.scatter(jitter.normal(j, 0.075, len(v)), v, s=15, facecolor=c,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        ax.text(j, v.max() + (v.max() - v.min() + 1) * 0.05, f"{v.mean():.1f}", ha="center",
                va="bottom", fontsize=8.5, color=c, fontweight="bold")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results_shimask_all/corolla_traits_final.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_shimask_all/island_guide_traits.png"))
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
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.055, right=0.985, top=0.86, bottom=0.17)

    _box_strip(fig.add_subplot(gs[0, 0]), data, "guide_coverage_pct",
               "Nectar-guide coverage (% of corolla)", "A  Guide coverage")
    _box_strip(fig.add_subplot(gs[0, 1]), data, "n_guide_spots",
               "Nectar-guide spots (count)", "B  Guide spot count", cast=lambda x: int(float(x)))

    axc = fig.add_subplot(gs[0, 2])
    fracs = []
    for isl in ORDER:
        present = [r for r in data[isl] if r["has_nectar_guide"] == "1"]
        fracs.append(100.0 * len(present) / len(data[isl]))
    bars = axc.bar(range(len(ORDER)), fracs, color=[COLOUR[i] for i in ORDER],
                   edgecolor="white", linewidth=1.0, width=0.66)
    for j, (b, f) in enumerate(zip(bars, fracs)):
        axc.text(j, f + 1.5, f"{f:.0f}%", ha="center", va="bottom", fontsize=9,
                 color=COLOUR[ORDER[j]], fontweight="bold")
    axc.set_xticks(range(len(ORDER)))
    axc.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=9)
    axc.set_ylabel("Corollas with a nectar guide (%)")
    axc.set_ylim(0, 108)
    axc.set_title("C  Guide-present fraction", fontsize=11, fontweight="bold", loc="left", pad=8)
    axc.yaxis.grid(True, color=GRID, lw=0.8)
    axc.set_axisbelow(True)
    for spine in ("top", "right"):
        axc.spines[spine].set_visible(False)

    fig.suptitle("Nectar-guide traits across Izu-island populations of Campanula microdonta  "
                 f"(n = {len(rows)} corollas)", x=0.055, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.055, 0.035, "Purple guide spots detected inside the corolla ROI on the raw scan. "
             "Guide-present = >=150 guide pixels.", fontsize=7.5, color=MUTED, ha="left")
    fig.text(0.055, 0.01, "Box = median/IQR, whiskers 1.5xIQR, points jittered.",
             fontsize=7.5, color=MUTED, ha="left")

    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
