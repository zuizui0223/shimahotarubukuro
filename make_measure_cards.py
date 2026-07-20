#!/usr/bin/env python3
"""Per-flower measurement cards: what is measured on each corolla, and how.

For every corolla each card shows the silhouette turned upright (base at top) with
the measurement constructs drawn on it:
  - corolla length  : the vertical extent, base -> tip (amber)
  - corolla width   : the widest cross-section (blue)
  - throat width    : median ROI width in the proximal tube band (cyan)
  - mouth width     : median ROI width in the distal rim band (teal)
  - nectar guide    : the detected purple spots (magenta)
and a caption lists every trait value including the ones that are ratios or live off
the ROI (lobe incision, aspect, guide reach/contrast, organ length, style ratio) and
the field sexual phase (s = male, p = female). Cards are laid out per sheet, so the
20 sheets document flower-by-flower how each trait is measured.

Writes results_shimask_all/measure_cards/<sheet>.png and a single annotated key
results_shimask_all/measure_cards/_methods_key.png.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import measure_guides as base
import shimask_input
import remeasure_medial as rm
import pollination_traits as pt
from run_all_shimask_confirmed import find_raw

MM = float(base.MM_PX)
OUT = Path("results_shimask_all/measure_cards")
COL = {"len": "#f0a020", "wid": "#2878ff", "throat": "#12b5c9",
       "mouth": "#0d8a6a", "guide": "#c93cc9"}


def master_by_gid() -> dict:
    m = {}
    p = Path("results_shimask_all/corolla_master.csv")
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            m[int(r["collar"])] = r
    return m


def gid_lookup() -> dict:
    g = {}
    for r in csv.DictReader(Path("results_shimask_all/global_index.csv").open(encoding="utf-8-sig")):
        g[(r["sheet"], r["sheet_corolla_id"])] = int(r["global_id"])
    return g


def upright_pair(mask_local, raw_crop, angle):
    """Rotate mask and its raw crop to upright; return cropped (mask, raw, bbox0)."""
    h, w = mask_local.shape
    cx, cy = w / 2.0, h / 2.0
    pad = int(max(h, w))
    M = cv2.getRotationMatrix2D((cx + pad, cy + pad), angle - 90.0, 1.0)
    bm = cv2.copyMakeBorder(mask_local, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    br = cv2.copyMakeBorder(raw_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    rm_ = cv2.warpAffine(bm, M, (bm.shape[1], bm.shape[0]), flags=cv2.INTER_NEAREST)
    rr = cv2.warpAffine(br, M, (br.shape[1], br.shape[0]), flags=cv2.INTER_LINEAR,
                        borderValue=(255, 255, 255))
    ys, xs = np.where(rm_ > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rm_[y0:y1, x0:x1], rr[y0:y1, x0:x1], M, (x0, y0), pad


def row_extent(mask, y):
    xs = np.where(mask[y] > 0)[0]
    return (xs.min(), xs.max()) if xs.size else (None, None)


def draw_card(ax, raw, comps, sheet, cid, gid, meta):
    # locate the piece for this cid (handles a/b split ids)
    base_id = cid.rstrip("ab")
    comp = comps[int(base_id) - 1]
    parts = rm.split_merged_pair(comp["mask"].astype(np.uint8))
    suffixes = [""] if len(parts) == 1 else ["a", "b"]
    piece = parts[suffixes.index(cid[len(base_id):] or "")]
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    raw_crop = raw[y0:y1, x0:x1].copy()
    trimmed = (sheet, cid) in rm.TRIM_TO_PETAL
    mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
    solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
    m = rm.medial_axis(mask_local)
    angle = float(m["angle_deg"])

    umask, uraw, M, off, pad = upright_pair(solid, raw_crop, angle)
    H, W = umask.shape
    prof = np.array([(umask[y] > 0).sum() for y in range(H)], float)

    ax.imshow(cv2.cvtColor(uraw, cv2.COLOR_BGR2RGB))
    cont, _ = cv2.findContours(umask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        ax.plot(c[:, 0, 0], c[:, 0, 1], color="#2ca02c", lw=1.2)

    def band_row(lo, hi):
        seg = range(int(lo * H), max(int(lo * H) + 1, int(hi * H)))
        widths = [(umask[y] > 0).sum() for y in seg]
        yy = list(seg)[int(np.argmin([abs(w - np.median(widths)) for w in widths]))]
        return yy

    # length (base->tip) at the tube-base-centre column
    top_xs = np.where(umask[:max(2, int(0.06 * H))].sum(0) > 0)[0]
    xc = float(top_xs.mean()) if top_xs.size else W / 2
    ax.annotate("", xy=(xc, H - 2), xytext=(xc, 1),
                arrowprops=dict(arrowstyle="<->", color=COL["len"], lw=1.8))
    # width (widest row)
    yw = int(np.argmax(prof))
    a, b = row_extent(umask, yw)
    ax.plot([a, b], [yw, yw], color=COL["wid"], lw=2.2)
    # throat (proximal band 4-16%) and mouth (distal band 72-88%)
    for lo, hi, key in ((0.04, 0.16, "throat"), (0.72, 0.88, "mouth")):
        yy = band_row(lo, hi)
        a, b = row_extent(umask, yy)
        if a is not None:
            ax.plot([a, b], [yy, yy], color=COL[key], lw=2.0)
    # nectar-guide spots
    spot, (bx0, by0), _c, _s = pt.guide_and_color(raw, piece)
    if int(spot.sum()) > 0:
        gloc = np.zeros_like(mask_local)
        gy, gx = np.where(spot)
        yy, xx = gy + (by0 - y0), gx + (bx0 - x0)
        ok = (yy >= 0) & (yy < gloc.shape[0]) & (xx >= 0) & (xx < gloc.shape[1])
        gloc[yy[ok], xx[ok]] = 1
        bg = cv2.copyMakeBorder(gloc, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
        gr = cv2.warpAffine(bg, M, (bg.shape[1], bg.shape[0]), flags=cv2.INTER_NEAREST)
        gr = gr[off[1]:off[1] + H, off[0]:off[0] + W]
        yy, xx = np.where(gr > 0)
        ax.scatter(xx, yy, s=1.2, color=COL["guide"], alpha=0.5, linewidths=0)

    ax.set_xlim(-6, W + 6)
    ax.set_ylim(H + 6, -6)
    ax.axis("off")
    status = {"s": "male", "p": "female", "na": "-"}.get(meta.get("status", ""), "?")
    title = f"#{gid}  ({sheet} C{cid})  [{status}]"
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    cap = (f"L {meta.get('corolla_length_mm','')}  W {meta.get('corolla_width_obs_mm','')} mm\n"
           f"throat {meta.get('throat_width_mm','')}  mouth {meta.get('mouth_width_mm','')}\n"
           f"aspect {meta.get('corolla_aspect_L_W','')}  lobe {meta.get('lobe_incision_mm','')}\n"
           f"guide {meta.get('guide_coverage_pct','')}%  reach {meta.get('guide_reach_frac','')}\n"
           f"organ {meta.get('organ_length_mm','')}  style/cor {meta.get('style_corolla_ratio','')}")
    ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.6, family="DejaVu Sans", color="#222")


def sheet_cards(sheet, master, glookup):
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    ids = []
    for cid0, comp in enumerate(comps):
        parts = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        sufs = [""] if len(parts) == 1 else ["a", "b"]
        ids += [f"{cid0 + 1}{s}" for s in sufs]

    n = len(ids)
    ncol = min(5, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 3.1 + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, cid in enumerate(ids):
        gid = glookup[(sheet, cid)]
        draw_card(axes[k], raw, comps, sheet, cid, gid, master.get(gid, {}))
    lo = min(glookup[(sheet, c)] for c in ids)
    hi = max(glookup[(sheet, c)] for c in ids)
    fig.suptitle(f"{sheet}  -  per-flower measurement cards  (#{lo}-{hi})",
                 fontsize=12, fontweight="bold", y=0.998)
    handles = [plt.Line2D([0], [0], color=COL["len"], lw=2.4, label="corolla length (base->tip)"),
               plt.Line2D([0], [0], color=COL["wid"], lw=2.4, label="corolla width (widest)"),
               plt.Line2D([0], [0], color=COL["throat"], lw=2.4, label="throat width (proximal)"),
               plt.Line2D([0], [0], color=COL["mouth"], lw=2.4, label="mouth width (distal)"),
               plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["guide"],
                          markersize=7, label="nectar-guide spots")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.028, "Widths are full-flower-equivalent (folded x2); caption also lists lobe incision, "
             "aspect, guide reach/coverage, organ length, style/corolla ratio, and [sex phase].",
             ha="center", fontsize=7.2, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sheet}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


DEFS = [
    ("corolla length", COL["len"], "Longer side of the minimum-area box round the ROI "
     "(base->tip; upright-mounted). mm."),
    ("corolla width", COL["wid"], "Shorter box side = widest cross-section. Folded flowers x2 "
     "for a full-flower-equivalent width. mm."),
    ("throat width", COL["throat"], "Median ROI width in the proximal 4-16% band (tube region); "
     "a corolla-entrance proxy. Full-eq. mm."),
    ("mouth width", COL["mouth"], "Median ROI width in the distal 72-88% band (rim below the "
     "lobe tips); the other CE proxy. Full-eq. mm."),
    ("nectar-guide spots", COL["guide"], "Purple pixels inside the ROI ((r-g)>18 etc.). Give "
     "coverage %, spot count, reach along length, CIELab contrast."),
    ("lobe incision", "#a0620a", "Depth of the scalloped distal margin (98th-15th percentile of "
     "the tip-most edge over the distal half). mm."),
    ("corolla aspect", "#555", "length / full-eq width. Tube flare = width / throat."),
    ("reproductive organ", "#d02020", "Reviewer's green stroke measured end to end (chord). Style/"
     "corolla = organ length / corolla length. Field phase s=male, p=female."),
]


def make_methods_key(master, glookup):
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05, left=0.02, right=0.98,
                          top=0.9, bottom=0.05)
    sheet, cid = "oshima1", "1"
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    comps = shimask_input.red_corolla_components(
        raw, ann, strokes=shimask_input.stroke_masks(raw, ann))
    axf = fig.add_subplot(gs[0, 0])
    draw_card(axf, raw, comps, sheet, cid, glookup[(sheet, cid)], master.get(1, {}))

    axt = fig.add_subplot(gs[0, 1])
    axt.axis("off")
    y = 0.98
    for name, col, desc in DEFS:
        axt.plot([0.0, 0.045], [y, y], color=col, lw=3.2, transform=axt.transAxes,
                 clip_on=False)
        axt.text(0.065, y, name, transform=axt.transAxes, fontsize=10.5, fontweight="bold",
                 va="center", color="#111")
        axt.text(0.065, y - 0.045, desc, transform=axt.transAxes, fontsize=8.6, va="top",
                 color="#333", wrap=True)
        y -= 0.125
    fig.suptitle("How each floral trait is measured  -  Campanula microdonta",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold")
    out = OUT / "_methods_key.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=None, help="one sheet only (for testing)")
    ap.add_argument("--key-only", action="store_true")
    args = ap.parse_args()
    master, glookup = master_by_gid(), gid_lookup()
    make_methods_key(master, glookup)
    if args.key_only:
        return
    sheets = ([args.sheet] if args.sheet else
              sorted(p.stem for p in Path("shimask").iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
    for sheet in sheets:
        out = sheet_cards(sheet, master, glookup)
        print(f"[{sheet}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
