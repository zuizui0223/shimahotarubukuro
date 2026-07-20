#!/usr/bin/env python3
"""The bumblebee-absence hypothesis for nectar guides, in three panels.

A  Guide coverage per island, Oshima (Bombus present) ringed - the step at the
   Bombus boundary.
B  Guide coverage vs corolla length: is the difference just because Oshima flowers
   are bigger? The Bombus-free islands stay low at every size while Oshima sits high,
   and the ANCOVA (coverage ~ length + Bombus) shows size carries no signal while
   Bombus presence adds ~+16 coverage points - guide investment is decoupled from
   size, the signature expected under adaptive reduction rather than neutral allometry.
C  Pst-analog per trait: among-island divergence of guide traits vs size traits. A
   deliberately cautious panel - guide divergence is NOT larger than size divergence
   in this variance sense, so the size-controlled test in B, not raw divergence, is
   what carries the hypothesis; a definitive test needs neutral genetic markers.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_island_traits import ORDER, COLOUR, INK, MUTED, GRID, island_of  # noqa: E402
import guide_divergence as gd  # noqa: E402


def main() -> None:
    rows = gd.load()
    data = {i: [] for i in ORDER}
    for r in rows:
        data[island_of(r["sheet"])].append(r)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(14.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.28,
                          left=0.055, right=0.985, top=0.84, bottom=0.16)

    # -- A: coverage by island --
    axa = fig.add_subplot(gs[0, 0])
    jit = np.random.RandomState(0)
    for j, isl in enumerate(ORDER):
        v = np.array([float(r["guide_coverage_pct"]) for r in data[isl]
                      if r["guide_coverage_pct"] not in ("", "nan")])
        c = COLOUR[isl]
        axa.boxplot(v, positions=[j], widths=0.56, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.0),
                    capprops=dict(color=MUTED, lw=1.0),
                    boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.4))
        axa.scatter(jit.normal(j, 0.08, len(v)), v, s=14, facecolor=c,
                    edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        axa.text(j, v.max() + 1.2, f"{v.mean():.1f}", ha="center", va="bottom",
                 fontsize=8.5, color=c, fontweight="bold")
    axa.axvspan(-0.5, 0.5, color=INK, alpha=0.05, zorder=0)
    axa.text(0.02, 0.98, "Bombus\npresent", transform=axa.transAxes, ha="left", va="top",
             fontsize=8, color=INK, fontweight="bold")
    axa.set_xticks(range(len(ORDER)))
    axa.set_xticklabels([i + (" *" if i in gd.BOMBUS_PRESENT else "") for i in ORDER],
                        rotation=25, ha="right", fontsize=8.5)
    axa.set_ylabel("Nectar-guide coverage (% of corolla)")
    axa.set_title("A  Guide investment by island", fontsize=11, fontweight="bold", loc="left", pad=8)
    axa.yaxis.grid(True, color=GRID, lw=0.8)
    axa.set_axisbelow(True)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    # -- B: coverage vs length, Bombus vs free --
    axb = fig.add_subplot(gs[0, 1])
    for isl in ORDER:
        cl = np.array([float(r["corolla_length_mm"]) for r in data[isl]
                       if r["guide_coverage_pct"] not in ("", "nan")])
        gv = np.array([float(r["guide_coverage_pct"]) for r in data[isl]
                       if r["guide_coverage_pct"] not in ("", "nan")])
        axb.scatter(cl, gv, s=28, facecolor=COLOUR[isl], edgecolor="white", linewidth=0.5,
                    alpha=0.85, zorder=3, marker="o" if isl in gd.BOMBUS_PRESENT else "o")
    for grp, ls, lab in ((1, "-", "Bombus present"), (0, "--", "bumblebee-free")):
        sub = [r for r in rows if r["bombus"] == grp and r["guide_coverage_pct"] not in ("", "nan")]
        x = np.array([float(r["corolla_length_mm"]) for r in sub])
        y = np.array([float(r["guide_coverage_pct"]) for r in sub])
        a, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        axb.plot(xs, a * xs + b, color=INK, lw=1.4, ls=ls, zorder=2)
    beta, se, pv, r2, n = gd.ols_ancova(rows)
    axb.text(0.03, 0.97,
             "ANCOVA  coverage ~ length + Bombus\n"
             f"length: coef {beta[1]:+.2f}  p={pv[1]:.2f}  (n.s.)\n"
             f"Bombus: coef {beta[2]:+.1f}  p={pv[2]:.0e}",
             transform=axb.transAxes, fontsize=8.2, va="top", color=INK,
             bbox=dict(boxstyle="round", fc="white", ec=MUTED, lw=0.8, alpha=0.9))
    axb.set_xlabel("Corolla length (mm)")
    axb.set_ylabel("Nectar-guide coverage (%)")
    axb.set_title("B  Guide investment is decoupled from size", fontsize=11,
                  fontweight="bold", loc="left", pad=8)
    axb.grid(True, color=GRID, lw=0.8)
    axb.set_axisbelow(True)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOUR[i],
                      markeredgecolor="white", markersize=8, label=i) for i in ORDER]
    handles += [Line2D([0], [0], color=INK, ls="-", label="Bombus fit"),
                Line2D([0], [0], color=INK, ls="--", label="bumblebee-free fit")]
    axb.legend(handles=handles, fontsize=7.3, loc="upper right", frameon=False, ncol=1)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # -- C: Pst comparison --
    axc = fig.add_subplot(gs[0, 2])
    guide_traits = [("guide_coverage_pct", "guide coverage"), ("n_guide_spots", "guide spots"),
                    ("guide_contrast_dE", "guide contrast"), ("guide_saturation", "guide chroma")]
    size_traits = [("corolla_length_mm", "corolla length"), ("throat_width_mm", "throat width"),
                   ("mouth_width_mm", "mouth width"), ("style_length_mm", "style length")]
    labels, vals, colours = [], [], []
    for key, lab in guide_traits:
        p, *_ = gd.pst(rows, key)
        labels.append(lab); vals.append(p); colours.append(COLOUR["Kozushima"])
    for key, lab in size_traits:
        p, *_ = gd.pst(rows, key)
        labels.append(lab); vals.append(p); colours.append(COLOUR["Oshima"])
    ypos = np.arange(len(labels))[::-1]
    axc.barh(ypos, vals, color=colours, edgecolor="white", height=0.7)
    for y, v in zip(ypos, vals):
        axc.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    axc.set_yticks(ypos)
    axc.set_yticklabels(labels, fontsize=8.5)
    axc.set_xlabel("Pst  (among-island divergence)")
    axc.set_title("C  Guide vs size divergence (Pst)", fontsize=11, fontweight="bold", loc="left", pad=8)
    axc.set_xlim(0, max(vals) * 1.28)
    axc.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOUR["Kozushima"],
                               markeredgecolor="white", markersize=9, label="guide traits"),
                        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOUR["Oshima"],
                               markeredgecolor="white", markersize=9, label="size traits")],
               fontsize=8, loc="upper right", frameon=False)
    axc.xaxis.grid(True, color=GRID, lw=0.8)
    axc.set_axisbelow(True)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    fig.suptitle("Nectar guides track the loss of bumblebees in Izu-island Campanula microdonta",
                 x=0.055, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.055, 0.015,
             "Oshima is the only island with a bumblebee (Bombus ardens). Guide coverage is far higher there and "
             "the size-controlled ANCOVA (B) shows the gap is not explained by flower size. Pst (C) is a phenotypic "
             "surrogate, not a Qst-Fst test - neutral genetic markers would be needed to firmly exclude drift.",
             fontsize=7.5, color=MUTED, ha="left")

    out = Path("results_shimask_all/guide_bombus_hypothesis.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
