#!/usr/bin/env python3
"""Measure the actual red annotation colour from shimask-minus-raw differences.

The raw scan is resized to the annotation image. Only pixels that changed strongly
in the annotation and moved toward red are inventoried. This avoids counting the
flower's natural purple/red pigmentation as marker colour.
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
    """Return annotation pixels that are newly added red marker strokes."""
    h, w = annotated.shape[:2]
    raw_small = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB).astype(np.int16)
    raw_rgb = cv2.cvtColor(raw_small, cv2.COLOR_BGR2RGB).astype(np.int16)

    delta = ann_rgb - raw_rgb
    absolute_change = np.linalg.norm(delta, axis=-1)
    r, g, b = ann_rgb[..., 0], ann_rgb[..., 1], ann_rgb[..., 2]
    dr, dg, db = delta[..., 0], delta[..., 1], delta[..., 2]

    # These are identification conditions for added drawing pixels, not the final
    # fixed-RGB extractor. They require a strong change from the raw scan and a
    # clear movement toward red, so natural purple guide pixels are excluded.
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


def stats_row(sheet: str, pixels: np.ndarray) -> dict[str, object]:
    if pixels.size == 0:
        raise RuntimeError(f'No added red marker pixels in {sheet}')
    counts = Counter(map(tuple, pixels.tolist()))
    mode_rgb, mode_count = counts.most_common(1)[0]
    q = np.percentile(pixels, [1, 5, 25, 50, 75, 95, 99], axis=0)
    row: dict[str, object] = {
        'sheet': sheet,
        'n_marker_pixels': len(pixels),
        'mode_r': mode_rgb[0], 'mode_g': mode_rgb[1], 'mode_b': mode_rgb[2],
        'mode_count': mode_count,
        'mode_fraction': round(mode_count / len(pixels), 6),
        'mean_r': round(float(pixels[:, 0].mean()), 3),
        'mean_g': round(float(pixels[:, 1].mean()), 3),
        'mean_b': round(float(pixels[:, 2].mean()), 3),
    }
    for name, values in zip(('p01','p05','p25','p50','p75','p95','p99'), q):
        row[f'{name}_r'] = round(float(values[0]), 3)
        row[f'{name}_g'] = round(float(values[1]), 3)
        row[f'{name}_b'] = round(float(values[2]), 3)
    row['top20_rgb'] = '|'.join(f'{r},{g},{b}:{n}' for (r,g,b), n in counts.most_common(20))
    return row


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
    for label_path in files:
        _, raw_path = find_raw(label_path.stem, args.raw_root)
        annotated = read_bgr(label_path)
        raw = read_bgr(raw_path)
        pixels = added_red_pixels(annotated, raw)
        rows.append(stats_row(label_path.stem, pixels))
        pooled.append(pixels)

    all_pixels = np.concatenate(pooled, axis=0)
    pooled_row = stats_row('__ALL_20_SHEETS__', all_pixels)
    rows.append(pooled_row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / 'red_rgb_measurements.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    report = args.out_dir / 'RED_RGB_REPORT.txt'
    report.write_text(
        'Red marker RGB measured from annotation-minus-raw changed pixels\n'
        f"pooled_mode_RGB=({pooled_row['mode_r']}, {pooled_row['mode_g']}, {pooled_row['mode_b']})\n"
        f"pooled_mode_fraction={pooled_row['mode_fraction']}\n"
        f"pooled_median_RGB=({pooled_row['p50_r']}, {pooled_row['p50_g']}, {pooled_row['p50_b']})\n"
        f"pooled_p05_RGB=({pooled_row['p05_r']}, {pooled_row['p05_g']}, {pooled_row['p05_b']})\n"
        f"pooled_p95_RGB=({pooled_row['p95_r']}, {pooled_row['p95_g']}, {pooled_row['p95_b']})\n"
        f"pooled_top20={pooled_row['top20_rgb']}\n",
        encoding='utf-8',
    )
    print(report.read_text(encoding='utf-8'))
    for row in rows[:-1]:
        print(row['sheet'], f"mode=({row['mode_r']},{row['mode_g']},{row['mode_b']})", 'median=',
              f"({row['p50_r']},{row['p50_g']},{row['p50_b']})", 'n=', row['n_marker_pixels'])


if __name__ == '__main__':
    main()
