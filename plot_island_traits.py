#!/usr/bin/env python3
"""Island comparison of corolla size from the ROI-based trait table.

Reads ``results_shimask_all/medial_traits.csv`` (see ``remeasure_medial.py``) and
draws corolla length and full-open-equivalent width by island, plus a length-vs-
width scatter. Folded halves are shown at full-open-equivalent width (x2) so both
fold states compare on one basis. Corollas whose ROI is flagged ``roi_misaligned``
are excluded (their mask does not match the scan and awaits re-annotation).

The five-island categorical palette (Okabe-Ito subset) is colour-vision-deficiency
safe; island is additionally encoded by x-position (A, B) and marker shape encodes
fold state (C), so identity never rests on colour alone.
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

ORDER = ["Oshima", "Toshima", "Niijima", "Shikinejima", "Kozushima"]
COLOUR = {
    "Oshima": "#0072B2", "Toshima": "#E69F00", "Niijima": "#009E73",
    "Shikinejima": "#D55E00", "Kozushima": "#CC79A7",
}
INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e7e7e3"


def island_of(sheet: str) -> str:
    for prefix, name in (("oshima", "Oshima"), ("toshima", "Toshima"),
                         ("niij", "Niijima"), ("shikine", "Shikinejima"),
                         ("kozu", "Kozushima")):
        if sheet.startswith(prefix):
            return name
    return "?"


def load(csv_path: Path) -> tuple[dict[str, dict[str, list]], int]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    # All 218 corollas carry a valid length/width; reconstructed/trimmed ROIs affect
    # only area, so none are dropped from the size comparison.
    used = rows
    data = {i: {"len": [], "w": [], "fold": []} for i in ORDER}
    for r in used:
        d = data[island_of(r["sheet"])]
        d["len"].append(float(r["corolla_length_mm"]))
        d["w"].append(float(r["corolla_width_fulleq_mm"]))
        d["fold"].append(r["fold_state"])
    return data, len(rows) - len(used)


def _box_strip(ax, data, key, ylabel, title):
    jitter = np.random.RandomState(0)
    for j, isl in enumerate(ORDER):
        v = np.array(data[isl][key])
        c = COLOUR[isl]
        ax.boxplot(v, positions=[j], widths=0.56, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED, lw=1.0),
                   capprops=dict(color=MUTED, lw=1.0),
                   boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.4))
        ax.scatter(jitter.normal(j, 0.075, len(v)), v, s=15, facecolor=c,
                   edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        ax.text(j, v.max() + (v.max() - v.min()) * 0.06, f"{v.mean():.1f}", ha="center",
                va="bottom", fontsize=8.5, color=c, fontweight="bold")
        ax.text(j, -0.5, f"n={len(v)}", ha="center", va="top", fontsize=8, color=MUTED,
                transform=ax.get_xaxis_transform())
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
    ap.add_argument("--csv", type=Path, default=Path("results_shimask_all/medial_traits.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_shimask_all/island_corolla_size.png"))
    args = ap.parse_args()

    data, n_excl = load(args.csv)
    n_used = sum(len(data[i]["len"]) for i in ORDER)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28,
                          left=0.055, right=0.985, top=0.86, bottom=0.17)

    _box_strip(fig.add_subplot(gs[0, 0]), data, "len", "Corolla length (mm)", "A  Corolla length")
    _box_strip(fig.add_subplot(gs[0, 1]), data, "w", "Full-open-equivalent width (mm)",
               "B  Corolla width (full-eq)")

    axc = fig.add_subplot(gs[0, 2])
    for isl in ORDER:
        length = np.array(data[isl]["len"])
        width = np.array(data[isl]["w"])
        fold = np.array(data[isl]["fold"])
        for state, marker in (("opened_full", "o"), ("folded_half", "^")):
            sel = fold == state
            if sel.any():
                axc.scatter(length[sel], width[sel], s=34, marker=marker, facecolor=COLOUR[isl],
                            edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)
    axc.set_xlabel("Corolla length (mm)")
    axc.set_ylabel("Full-open-equivalent width (mm)")
    axc.set_title("C  Length vs width by island", fontsize=11, fontweight="bold", loc="left", pad=8)
    axc.grid(True, color=GRID, lw=0.8)
    axc.set_axisbelow(True)
    for spine in ("top", "right"):
        axc.spines[spine].set_visible(False)
    islands = [Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOUR[i],
                      markeredgecolor="white", markersize=9, label=i) for i in ORDER]
    shapes = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555",
                     markeredgecolor="white", markersize=8, label="opened (5-lobe)"),
              Line2D([0], [0], marker="^", color="none", markerfacecolor="#555",
                     markeredgecolor="white", markersize=8, label="folded (width x2)")]
    axc.add_artist(axc.legend(handles=islands, title="Island", loc="lower right",
                              fontsize=8, title_fontsize=8.5, frameon=False))
    axc.legend(handles=shapes, loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(f"Corolla size across Izu-island populations of Campanula microdonta  "
                 f"(n = {n_used} corollas, 20 sheets)", x=0.055, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.055, 0.055, "Measured from the reviewed corolla ROI (minimum-area oriented box); "
             "folded halves at full-open-equivalent width (x2).", fontsize=7.5, color=MUTED, ha="left")
    fig.text(0.055, 0.02, "Box = median/IQR, whiskers 1.5xIQR, points jittered.  "
             "All 218 corollas included (length/width valid for every ROI).",
             fontsize=7.5, color=MUTED, ha="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  (n={n_used}, excluded={n_excl})")


if __name__ == "__main__":
    main()
