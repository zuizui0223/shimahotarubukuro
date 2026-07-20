#!/usr/bin/env python3
"""Reproductive-organ length by island and its allometry with corolla length.

Organ length is the reviewer's green line measured end to end. Panel A is organ
length per island; panel B is corolla length vs organ length (allometry), coloured
by island. Same CVD-safe palette as the other figures.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_island_traits import ORDER, COLOUR, INK, MUTED, GRID, island_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results_shimask_all/corolla_traits_final.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_shimask_all/island_organ_traits.png"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.csv.open(encoding="utf-8-sig")) if r["organ_length_mm"]]
    data = {i: [] for i in ORDER}
    for r in rows:
        data[island_of(r["sheet"])].append(r)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.24,
                          left=0.07, right=0.985, top=0.86, bottom=0.16)

    ax = fig.add_subplot(gs[0, 0])
    jitter = np.random.RandomState(0)
    for j, isl in enumerate(ORDER):
        v = np.array([float(r["organ_length_mm"]) for r in data[isl]])
        c = COLOUR[isl]
        ax.boxplot(v, positions=[j], widths=0.56, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.0),
                   capprops=dict(color=MUTED, lw=1.0),
                   boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.4))
        ax.scatter(jitter.normal(j, 0.075, len(v)), v, s=16, facecolor=c,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        ax.text(j, v.max() + 0.8, f"{v.mean():.1f}", ha="center", va="bottom",
                fontsize=8.5, color=c, fontweight="bold")
        ax.text(j, -0.02, f"n={len(v)}", ha="center", va="top", fontsize=8, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Reproductive-organ length (mm)")
    ax.set_title("A  Organ length", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    axb = fig.add_subplot(gs[0, 1])
    allx, ally = [], []
    for isl in ORDER:
        cl = np.array([float(r["corolla_length_mm"]) for r in data[isl]])
        ol = np.array([float(r["organ_length_mm"]) for r in data[isl]])
        allx += list(cl)
        ally += list(ol)
        axb.scatter(cl, ol, s=30, facecolor=COLOUR[isl], edgecolor="white", linewidth=0.5,
                    alpha=0.85, zorder=3)
    allx, ally = np.array(allx), np.array(ally)
    a, b = np.polyfit(allx, ally, 1)
    xs = np.array([allx.min(), allx.max()])
    axb.plot(xs, a * xs + b, color=INK, lw=1.4, ls="--", zorder=2)
    r = np.corrcoef(allx, ally)[0, 1]
    axb.text(0.04, 0.95, f"r = {r:.2f}  (organ = {a:.2f}xcorolla + {b:.1f})", transform=axb.transAxes,
             fontsize=9, va="top", color=INK)
    axb.set_xlabel("Corolla length (mm)")
    axb.set_ylabel("Reproductive-organ length (mm)")
    axb.set_title("B  Organ vs corolla length (allometry)", fontsize=11, fontweight="bold", loc="left", pad=8)
    axb.grid(True, color=GRID, lw=0.8)
    axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOUR[i],
                               markeredgecolor="white", markersize=9, label=i) for i in ORDER],
               title="Island", loc="lower right", fontsize=8, title_fontsize=8.5, frameon=False)

    fig.suptitle(f"Reproductive-organ length by Izu island - Campanula microdonta  "
                 f"(n = {len(rows)} with a marked organ)", x=0.07, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.07, 0.02, "Organ length = reviewer's green line measured end to end. "
             "Box = median/IQR, whiskers 1.5xIQR.", fontsize=7.5, color=MUTED, ha="left")

    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
