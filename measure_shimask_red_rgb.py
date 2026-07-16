#!/usr/bin/env python3
"""Measure the actual red annotation colour in all shimask images.

This script does not alter extraction. It inventories vivid-red pixels and reports
per-sheet and pooled RGB modes/quantiles so the marker colour can be fixed from
observed data rather than guessed.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def candidate_pixels(image: np.ndarray) -> np.ndarray:
    """Broad red-sector inventory only; intentionally not the final extractor."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Broad enough to retain JPEG variants of a vivid red pen while excluding
    # most purple nectar-guide pixels; results are reported rather than assumed.
    keep = (r >= 150) & (r - g >= 70) & (r - b >= 45)
    return rgb[keep].astype(np.uint8)


def stats_row(sheet: str, pixels: np.ndarray) -> dict[str, object]:
    if pixels.size == 0:
        raise RuntimeError(f'No vivid-red candidate pixels in {sheet}')
    counts = Counter(map(tuple, pixels.tolist()))
    mode_rgb, mode_count = counts.most_common(1)[0]
    q = np.percentile(pixels, [1, 5, 25, 50, 75, 95, 99], axis=0)
    row: dict[str, object] = {
        'sheet': sheet,
        'n_candidate_pixels': len(pixels),
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
    row['top10_rgb'] = '|'.join(f'{r},{g},{b}:{n}' for (r,g,b), n in counts.most_common(10))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('shimask'))
    parser.add_argument('--out-dir', type=Path, default=Path('results_shimask_red_rgb'))
    args = parser.parse_args()

    files = sorted(p for p in args.labels.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(files) != 20:
        raise SystemExit(f'Expected 20 shimask images, found {len(files)}')

    rows = []
    pooled = []
    for path in files:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Could not read {path}')
        pixels = candidate_pixels(image)
        rows.append(stats_row(path.stem, pixels))
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
        'Observed vivid-red pixel inventory (not yet used as extraction threshold)\n'
        f"pooled_mode_RGB=({pooled_row['mode_r']}, {pooled_row['mode_g']}, {pooled_row['mode_b']})\n"
        f"pooled_mode_fraction={pooled_row['mode_fraction']}\n"
        f"pooled_median_RGB=({pooled_row['p50_r']}, {pooled_row['p50_g']}, {pooled_row['p50_b']})\n"
        f"pooled_p05_RGB=({pooled_row['p05_r']}, {pooled_row['p05_g']}, {pooled_row['p05_b']})\n"
        f"pooled_p95_RGB=({pooled_row['p95_r']}, {pooled_row['p95_g']}, {pooled_row['p95_b']})\n"
        f"pooled_top10={pooled_row['top10_rgb']}\n",
        encoding='utf-8',
    )
    print(report.read_text(encoding='utf-8'))
    for row in rows[:-1]:
        print(row['sheet'], f"mode=({row['mode_r']},{row['mode_g']},{row['mode_b']})", 'median=',
              f"({row['p50_r']},{row['p50_g']},{row['p50_b']})", 'n=', row['n_candidate_pixels'])


if __name__ == '__main__':
    main()
