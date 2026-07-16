#!/usr/bin/env python3
"""Measure the actual red annotation colour from shimask-minus-raw differences.

The raw scan is resized to the annotation image. Only pixels that changed strongly
in the annotation and moved toward red are inventoried. A second core-pixel subset
removes JPEG edge blending and estimates the original drawing colour.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from run_all_shimask_confirmed import find_raw

SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Could not read {path}')
    return image


def added_red_pixels(annotated: np.ndarray, raw: np.ndarray) -> np.ndarray:
    h, w = annotated.shape[:2]
    raw_small = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB).astype(np.int16)
    raw_rgb = cv2.cvtColor(raw_small, cv2.COLOR_BGR2RGB).astype(np.int16)
    delta = ann_rgb - raw_rgb
    absolute_change = np.linalg.norm(delta, axis=-1)
    r, g, b = ann_rgb[..., 0], ann_rgb[..., 1], ann_rgb[..., 2]
    dr, dg, db = delta[..., 0], delta[..., 1], delta[..., 2]
    keep = (
        (absolute_change >= 55.0)
        & (dr >= 35)
        & (dr - dg >= 45)
        & (dr - db >= 35)
        & (r >= 140)
        & (r - g >= 55)
        & (r - b >= 35)
    )
    return ann_rgb[keep].astype(np.uint8)


def core_pixels(pixels: np.ndarray) -> np.ndarray:
    """Keep high-saturation stroke interiors, excluding pale JPEG edge blends."""
    p = pixels.astype(np.int16)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    keep = (r >= 180) & (g <= 55) & (b <= 65) & (r - g >= 150) & (r - b >= 135)
    core = pixels[keep]
    if len(core) < 100:
        raise RuntimeError(f'Too few core marker pixels: {len(core)}')
    return core


def colour_stats(prefix: str, pixels: np.ndarray) -> dict[str, object]:
    counts = Counter(map(tuple, pixels.tolist()))
    mode_rgb, mode_count = counts.most_common(1)[0]
    median = np.median(pixels, axis=0)
    mean = pixels.mean(axis=0)
    return {
        f'{prefix}_n': len(pixels),
        f'{prefix}_mode_r': mode_rgb[0], f'{prefix}_mode_g': mode_rgb[1], f'{prefix}_mode_b': mode_rgb[2],
        f'{prefix}_mode_count': mode_count,
        f'{prefix}_median_r': round(float(median[0]), 3),
        f'{prefix}_median_g': round(float(median[1]), 3),
        f'{prefix}_median_b': round(float(median[2]), 3),
        f'{prefix}_mean_r': round(float(mean[0]), 3),
        f'{prefix}_mean_g': round(float(mean[1]), 3),
        f'{prefix}_mean_b': round(float(mean[2]), 3),
        f'{prefix}_top20': '|'.join(f'{r},{g},{b}:{n}' for (r,g,b), n in counts.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('shimask'))
    parser.add_argument('--raw-root', type=Path, default=Path('shimahotarubukuro'))
    parser.add_argument('--out-dir', type=Path, default=Path('results_shimask_red_rgb'))
    args = parser.parse_args()

    files = sorted(p for p in args.labels.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(files) != 20:
        raise SystemExit(f'Expected 20 shimask images, found {len(files)}')

    rows = []
    pooled = []
    pooled_core = []
    for label_path in files:
        _, raw_path = find_raw(label_path.stem, args.raw_root)
        pixels = added_red_pixels(read_bgr(label_path), read_bgr(raw_path))
        core = core_pixels(pixels)
        row = {'sheet': label_path.stem, **colour_stats('all', pixels), **colour_stats('core', core)}
        rows.append(row)
        pooled.append(pixels)
        pooled_core.append(core)

    all_pixels = np.concatenate(pooled, axis=0)
    all_core = np.concatenate(pooled_core, axis=0)
    pooled_row = {'sheet': '__ALL_20_SHEETS__', **colour_stats('all', all_pixels), **colour_stats('core', all_core)}
    rows.append(pooled_row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / 'red_rgb_measurements.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    report = args.out_dir / 'RED_RGB_REPORT.txt'
    report.write_text(
        'Red marker measured from annotation-minus-raw pixels\n'
        'Core subset excludes pale JPEG edge blending\n'
        f"core_mode_RGB=({pooled_row['core_mode_r']}, {pooled_row['core_mode_g']}, {pooled_row['core_mode_b']})\n"
        f"core_median_RGB=({pooled_row['core_median_r']}, {pooled_row['core_median_g']}, {pooled_row['core_median_b']})\n"
        f"core_mean_RGB=({pooled_row['core_mean_r']}, {pooled_row['core_mean_g']}, {pooled_row['core_mean_b']})\n"
        f"core_n={pooled_row['core_n']}\n"
        f"core_top20={pooled_row['core_top20']}\n",
        encoding='utf-8',
    )
    print(report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
