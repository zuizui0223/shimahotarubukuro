#!/usr/bin/env python3
"""Combine the per-sheet shimask-input trait outputs into study-wide tables.

Reads every ``sheets/<sheet>/traits.csv`` produced by
``run_all_shimask_confirmed.py`` and writes:
- ``all_traits.csv``: every corolla row from all sheets, in island/region order
- ``per_island_summary.csv``: one aggregate row per island

No measurement is redone here; this only concatenates and summarises the
already-measured per-sheet trait rows.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

# Islands in geographic (north-to-south) survey order, matching the existing
# results/per_island_summary.csv ordering.
REGION_ORDER = {
    "Oshima": 1,
    "Toshima": 2,
    "Niijima": 3,
    "Shikinejima": 3.5,
    "Kozushima": 4,
}


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_sheet_traits(sheets_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sheet_dir in sorted(sheets_dir.iterdir()):
        traits = sheet_dir / "traits.csv"
        if not traits.is_file():
            continue
        with traits.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def write_all_traits(rows: list[dict[str, str]], out_path: Path) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    def sort_key(row: dict[str, str]) -> tuple[float, str, float, float]:
        island = row.get("island", "")
        region = REGION_ORDER.get(island, 99.0)
        return (region, row.get("sheet", ""), _to_float(row.get("cy", "")) or 0.0, _to_float(row.get("cx", "")) or 0.0)

    rows_sorted = sorted(rows, key=sort_key)
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_sorted)
    return fields


def _mean_round(values: list[float], digits: int) -> str:
    return str(round(mean(values), digits)) if values else ""


def write_island_summary(rows: list[dict[str, str]], out_path: Path) -> list[dict[str, object]]:
    by_island: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_island.setdefault(row.get("island", ""), []).append(row)

    summary: list[dict[str, object]] = []
    for island, island_rows in by_island.items():
        lengths = [v for v in (_to_float(r.get("corolla_len_mm", "")) for r in island_rows) if v is not None]
        areas = [v for v in (_to_float(r.get("corolla_area_mm2", "")) for r in island_rows) if v is not None]
        covs = [v for v in (_to_float(r.get("guide_cov_pct", "")) for r in island_rows) if v is not None]
        present = [v for v in (_to_float(r.get("guide_present", "")) for r in island_rows) if v is not None]
        degraded = [v for v in (_to_float(r.get("degraded_flag", "")) for r in island_rows) if v is not None]
        spots = [v for v in (_to_float(r.get("n_spots", "")) for r in island_rows) if v is not None]
        summary.append(
            {
                "region_order": REGION_ORDER.get(island, 99.0),
                "island": island,
                "n_corolla": len(island_rows),
                "corolla_len_mm_mean": _mean_round(lengths, 1),
                "corolla_area_mm2_mean": _mean_round(areas, 0),
                "guide_cov_pct_mean": _mean_round(covs, 2),
                "guide_present_frac": _mean_round(present, 2),
                "degraded_frac": _mean_round(degraded, 2),
                "n_spots_mean": _mean_round(spots, 1),
            }
        )
    summary.sort(key=lambda r: float(r["region_order"]))
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_shimask_all"))
    args = parser.parse_args()

    sheets_dir = args.results_dir / "sheets"
    if not sheets_dir.is_dir():
        raise SystemExit(f"No per-sheet results under {sheets_dir}")

    rows = read_sheet_traits(sheets_dir)
    if not rows:
        raise SystemExit("No trait rows found; run run_all_shimask_confirmed.py first")

    write_all_traits(rows, args.results_dir / "all_traits.csv")
    summary = write_island_summary(rows, args.results_dir / "per_island_summary.csv")

    total = sum(int(r["n_corolla"]) for r in summary)
    print(f"Aggregated {total} corolla rows across {len(summary)} islands")
    for row in summary:
        print(
            f"  {row['island']}: n={row['n_corolla']} "
            f"len={row['corolla_len_mm_mean']}mm guide_cov={row['guide_cov_pct_mean']}%"
        )


if __name__ == "__main__":
    main()
