#!/usr/bin/env python3
"""Per-flower cards documenting the retained measurements.

Each card shows the reviewed corolla upright with corolla length, full-equivalent
width, proximal throat width, distal mouth width and the area-based nectar-guide mask.
The caption adds the retained ratios, lobe incision, reproductive-organ length and
field sexual phase. No spot count, guide colour, chromatic contrast or guide-reach
value is displayed because those quantities are not part of the publication pipeline.

Writes ``results_shimask_all/measure_cards/<sheet>.png`` and
``measure_cards/_methods_key.png``.
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
import guide_traits as gt
from run_all_shimask_confirmed import find_raw

OUT = Path("results_shimask_all/measure_cards")
COLOUR = {
    "length": "#f0a020",
    "width": "#2878ff",
    "throat": "#12b5c9",
    "mouth": "#0d8a6a",
    "guide": "#c93cc9",
}


def master_by_global_id() -> dict:
    master = {}
    path = Path("results_shimask_all/corolla_master.csv")
    if path.exists():
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            master[int(row["collar"])] = row
    return master


def global_id_lookup() -> dict:
    lookup = {}
    path = Path("results_shimask_all/global_index.csv")
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        lookup[(row["sheet"], row["sheet_corolla_id"])] = int(row["global_id"])
    return lookup


def upright_pair(mask_local, raw_crop, angle):
    """Rotate a mask and matching raw crop to the mounted base-up orientation."""
    height, width = mask_local.shape
    centre_x, centre_y = width / 2.0, height / 2.0
    pad = int(max(height, width))
    transform = cv2.getRotationMatrix2D(
        (centre_x + pad, centre_y + pad), angle - 90.0, 1.0
    )
    mask_big = cv2.copyMakeBorder(
        mask_local, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    raw_big = cv2.copyMakeBorder(
        raw_crop,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    mask_rotated = cv2.warpAffine(
        mask_big,
        transform,
        (mask_big.shape[1], mask_big.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    raw_rotated = cv2.warpAffine(
        raw_big,
        transform,
        (raw_big.shape[1], raw_big.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    ys, xs = np.where(mask_rotated > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (
        mask_rotated[y0:y1, x0:x1],
        raw_rotated[y0:y1, x0:x1],
        transform,
        (x0, y0),
        pad,
    )


def row_extent(mask, y):
    xs = np.where(mask[y] > 0)[0]
    return (xs.min(), xs.max()) if xs.size else (None, None)


def draw_card(axis, raw, components, sheet, corolla_id, global_id, metadata):
    base_id = corolla_id.rstrip("ab")
    component = components[int(base_id) - 1]
    parts = rm.split_merged_pair(component["mask"].astype(np.uint8))
    suffixes = [""] if len(parts) == 1 else ["a", "b"]
    piece = parts[suffixes.index(corolla_id[len(base_id):] or "")]
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    raw_crop = raw[y0:y1, x0:x1].copy()
    trimmed = (sheet, corolla_id) in rm.TRIM_TO_PETAL
    mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
    solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
    measured = rm.medial_axis(mask_local)
    angle = float(measured["angle_deg"])

    upright_mask, upright_raw, transform, offset, pad = upright_pair(
        solid, raw_crop, angle
    )
    height, width = upright_mask.shape
    profile = np.array([(upright_mask[y] > 0).sum() for y in range(height)], float)

    axis.imshow(cv2.cvtColor(upright_raw, cv2.COLOR_BGR2RGB))
    contours, _ = cv2.findContours(
        upright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        axis.plot(contour[:, 0, 0], contour[:, 0, 1], color="#2ca02c", lw=1.2)

    def representative_band_row(lower, upper):
        rows = list(range(int(lower * height), max(int(lower * height) + 1, int(upper * height))))
        widths = [(upright_mask[y] > 0).sum() for y in rows]
        return rows[int(np.argmin([abs(value - np.median(widths)) for value in widths]))]

    top_columns = np.where(
        upright_mask[:max(2, int(0.06 * height))].sum(0) > 0
    )[0]
    centre_x = float(top_columns.mean()) if top_columns.size else width / 2
    axis.annotate(
        "",
        xy=(centre_x, height - 2),
        xytext=(centre_x, 1),
        arrowprops=dict(arrowstyle="<->", color=COLOUR["length"], lw=1.8),
    )

    widest_row = int(np.argmax(profile))
    left, right = row_extent(upright_mask, widest_row)
    axis.plot([left, right], [widest_row, widest_row], color=COLOUR["width"], lw=2.2)

    for lower, upper, key in (
        (0.04, 0.16, "throat"),
        (0.72, 0.88, "mouth"),
    ):
        y = representative_band_row(lower, upper)
        left, right = row_extent(upright_mask, y)
        if left is not None:
            axis.plot([left, right], [y, y], color=COLOUR[key], lw=2.0)

    guide, (guide_x0, guide_y0), _roi_pixels = gt.guide_mask(raw, piece)
    if int(guide.sum()) > 0:
        local_guide = np.zeros_like(mask_local)
        gy, gx = np.where(guide)
        yy, xx = gy + (guide_y0 - y0), gx + (guide_x0 - x0)
        valid = (
            (yy >= 0) & (yy < local_guide.shape[0]) &
            (xx >= 0) & (xx < local_guide.shape[1])
        )
        local_guide[yy[valid], xx[valid]] = 1
        guide_big = cv2.copyMakeBorder(
            local_guide, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
        )
        guide_rotated = cv2.warpAffine(
            guide_big,
            transform,
            (guide_big.shape[1], guide_big.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        guide_rotated = guide_rotated[
            offset[1]:offset[1] + height,
            offset[0]:offset[0] + width,
        ]
        guide_y, guide_x = np.where(guide_rotated > 0)
        axis.scatter(
            guide_x,
            guide_y,
            s=1.2,
            color=COLOUR["guide"],
            alpha=0.5,
            linewidths=0,
        )

    axis.set_xlim(-6, width + 6)
    axis.set_ylim(height + 6, -6)
    axis.axis("off")
    phase = {"s": "male", "p": "female", "na": "-"}.get(
        metadata.get("status", ""), "?"
    )
    axis.set_title(
        f"#{global_id}  ({sheet} C{corolla_id})  [{phase}]",
        fontsize=8,
        fontweight="bold",
        pad=3,
    )
    caption = (
        f"L {metadata.get('corolla_length_mm', '')}  "
        f"W {metadata.get('corolla_width_fulleq_mm', '')} mm\n"
        f"throat {metadata.get('throat_width_mm', '')}  "
        f"mouth {metadata.get('mouth_width_mm', '')}\n"
        f"aspect {metadata.get('corolla_aspect_L_W', '')}  "
        f"lobe {metadata.get('lobe_incision_mm', '')}\n"
        f"guide coverage {metadata.get('guide_coverage_pct', '')}%\n"
        f"organ {metadata.get('organ_length_mm', '')}  "
        f"organ/cor {metadata.get('organ_corolla_ratio', '')}"
    )
    axis.text(
        0.5,
        -0.02,
        caption,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.6,
        family="DejaVu Sans",
        color="#222",
    )


def sheet_cards(sheet, master, global_lookup):
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    components = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    ids = []
    for index, component in enumerate(components):
        parts = rm.split_merged_pair(component["mask"].astype(np.uint8))
        suffixes = [""] if len(parts) == 1 else ["a", "b"]
        ids += [f"{index + 1}{suffix}" for suffix in suffixes]

    n_columns = min(5, len(ids))
    n_rows = math.ceil(len(ids) / n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(n_columns * 2.5, n_rows * 3.1 + 0.5)
    )
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for index, corolla_id in enumerate(ids):
        global_id = global_lookup[(sheet, corolla_id)]
        draw_card(
            axes[index],
            raw,
            components,
            sheet,
            corolla_id,
            global_id,
            master.get(global_id, {}),
        )

    lower = min(global_lookup[(sheet, corolla_id)] for corolla_id in ids)
    upper = max(global_lookup[(sheet, corolla_id)] for corolla_id in ids)
    fig.suptitle(
        f"{sheet}  -  per-flower measurement cards  (#{lower}-{upper})",
        fontsize=12,
        fontweight="bold",
        y=0.998,
    )
    handles = [
        plt.Line2D([0], [0], color=COLOUR["length"], lw=2.4, label="corolla length"),
        plt.Line2D([0], [0], color=COLOUR["width"], lw=2.4, label="full-eq width"),
        plt.Line2D([0], [0], color=COLOUR["throat"], lw=2.4, label="throat width"),
        plt.Line2D([0], [0], color=COLOUR["mouth"], lw=2.4, label="mouth width"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOUR["guide"],
            markersize=7,
            label="guide area",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.text(
        0.5,
        0.028,
        "Transverse widths are full-flower-equivalent for folded flowers (x2). "
        "Captions list only retained publication traits and field phase.",
        ha="center",
        fontsize=7.2,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sheet}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


DEFINITIONS = [
    (
        "corolla length",
        COLOUR["length"],
        "Base-to-tip side of the minimum-area oriented box around the reviewed ROI (mm).",
    ),
    (
        "corolla width",
        COLOUR["width"],
        "Transverse box side; folded flowers are multiplied by two for a full-flower-equivalent width (mm).",
    ),
    (
        "throat width",
        COLOUR["throat"],
        "Median flattened ROI width in the proximal 4-16% band; a 2-D entrance proxy (mm).",
    ),
    (
        "mouth width",
        COLOUR["mouth"],
        "Median flattened ROI width in the distal 72-88% band below the lobe tips (mm).",
    ),
    (
        "nectar-guide coverage",
        COLOUR["guide"],
        "Area fraction of the reviewed ROI classified as purple/magenta guide pixels (%). No spot count or colour value is reported.",
    ),
    (
        "lobe incision",
        "#a0620a",
        "Depth of the scalloped distal margin from the tip-edge distribution (mm).",
    ),
    (
        "shape ratios",
        "#555",
        "Corolla aspect = length/full-equivalent width; tube flare = width/throat.",
    ),
    (
        "reproductive organ",
        "#d02020",
        "Reviewer's green trace measured end to end; organ/corolla = organ length / corolla length. Field phase s=male, p=female.",
    ),
]


def make_methods_key(master, global_lookup):
    fig = plt.figure(figsize=(12.5, 6.2))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1.5],
        wspace=0.05,
        left=0.02,
        right=0.98,
        top=0.9,
        bottom=0.05,
    )
    sheet, corolla_id = "oshima1", "1"
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    components = shimask_input.red_corolla_components(
        raw, ann, strokes=shimask_input.stroke_masks(raw, ann)
    )
    flower_axis = fig.add_subplot(grid[0, 0])
    draw_card(
        flower_axis,
        raw,
        components,
        sheet,
        corolla_id,
        global_lookup[(sheet, corolla_id)],
        master.get(1, {}),
    )

    text_axis = fig.add_subplot(grid[0, 1])
    text_axis.axis("off")
    y = 0.98
    for name, colour, description in DEFINITIONS:
        text_axis.plot(
            [0.0, 0.045],
            [y, y],
            color=colour,
            lw=3.2,
            transform=text_axis.transAxes,
            clip_on=False,
        )
        text_axis.text(
            0.065,
            y,
            name,
            transform=text_axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="center",
            color="#111",
        )
        text_axis.text(
            0.065,
            y - 0.045,
            description,
            transform=text_axis.transAxes,
            fontsize=8.6,
            va="top",
            color="#333",
            wrap=True,
        )
        y -= 0.125
    fig.suptitle(
        "How the retained floral traits are measured - Campanula microdonta",
        x=0.02,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )
    out = OUT / "_methods_key.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default=None, help="one sheet only (for testing)")
    parser.add_argument("--key-only", action="store_true")
    args = parser.parse_args()
    master = master_by_global_id()
    global_lookup = global_id_lookup()
    make_methods_key(master, global_lookup)
    if args.key_only:
        return
    sheets = (
        [args.sheet]
        if args.sheet
        else sorted(
            p.stem for p in Path("shimask").iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    )
    for sheet in sheets:
        out = sheet_cards(sheet, master, global_lookup)
        print(f"[{sheet}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
