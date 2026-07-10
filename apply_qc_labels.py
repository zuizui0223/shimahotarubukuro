#!/usr/bin/env python3
"""Apply manually checked plant labels to traits.csv and styles.csv.

The join key is island + sheet + corolla_id. For styles.csv, nearest_corolla is
used as the corolla reference. By default new *_qc.csv files are written. With
--in-place, originals are backed up once as *.pre_qc.csv and replaced.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import unicodedata
from pathlib import Path


def clean(value: object) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value


def key(island: object, sheet: object, corolla: object) -> tuple[str, str, str]:
    return clean(island).lower(), clean(sheet).lower(), clean(corolla)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_qc(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[str]]:
    _, rows = read_csv(path)
    mapping: dict[tuple[str, str, str], dict[str, str]] = {}
    conflicts: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        plant_id = clean(row.get("plant_id_FILL"))
        exclude = clean(row.get("exclude_FILL"))
        flower_no = clean(row.get("flower_no_FILL"))
        notes = clean(row.get("notes"))
        row_key = key(row.get("island"), row.get("sheet"), row.get("corolla_id"))
        if not all(row_key):
            continue
        payload = {
            "plant_id": plant_id,
            "flower_no": flower_no,
            "exclude": exclude,
            "qc_notes": notes,
        }
        if row_key in mapping and mapping[row_key] != payload:
            conflicts.append(f"line {line_number}: conflicting QC label for {row_key}")
            mapping.pop(row_key, None)
            continue
        mapping[row_key] = payload
    return mapping, conflicts


def output_path(source: Path, in_place: bool) -> Path:
    if in_place:
        backup = source.with_name(f"{source.stem}.pre_qc{source.suffix}")
        if not backup.exists():
            shutil.copy2(source, backup)
        return source
    return source.with_name(f"{source.stem}_qc{source.suffix}")


def apply_traits(path: Path, qc: dict[tuple[str, str, str], dict[str, str]], in_place: bool) -> tuple[int, int, Path]:
    fields, rows = read_csv(path)
    for field in ("plant_id", "flower_no", "exclude", "qc_notes"):
        if field not in fields:
            fields.append(field)
    matched = 0
    for row in rows:
        payload = qc.get(key(row.get("island"), row.get("sheet"), row.get("corolla_id")))
        if payload is None:
            continue
        for field, value in payload.items():
            if value != "" or clean(row.get(field)) == "":
                row[field] = value
        matched += 1
    out = output_path(path, in_place)
    write_csv(out, fields, rows)
    return matched, len(rows), out


def apply_styles(path: Path, qc: dict[tuple[str, str, str], dict[str, str]], in_place: bool) -> tuple[int, int, Path]:
    fields, rows = read_csv(path)
    for field in ("plant_id", "nearest_corolla_exclude", "qc_notes"):
        if field not in fields:
            fields.append(field)
    matched = 0
    for row in rows:
        payload = qc.get(key(row.get("island"), row.get("sheet"), row.get("nearest_corolla")))
        if payload is None:
            continue
        row["plant_id"] = payload["plant_id"]
        row["nearest_corolla_exclude"] = payload["exclude"]
        row["qc_notes"] = payload["qc_notes"]
        matched += 1
    out = output_path(path, in_place)
    write_csv(out, fields, rows)
    return matched, len(rows), out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply qc_plant_labels.csv to result tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace traits.csv/styles.csv after creating *.pre_qc.csv backups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    qc_path = results_dir / "qc_plant_labels.csv"
    if not qc_path.exists():
        raise SystemExit(f"QC file not found: {qc_path}")

    qc, conflicts = load_qc(qc_path)
    if not qc:
        raise SystemExit("No usable plant_id_FILL values found in qc_plant_labels.csv")

    traits_path = results_dir / "traits.csv"
    if not traits_path.exists():
        raise SystemExit(f"Traits file not found: {traits_path}")
    matched, total, out = apply_traits(traits_path, qc, args.in_place)
    print(f"traits: {matched}/{total} rows labeled -> {out}")

    styles_path = results_dir / "styles.csv"
    if styles_path.exists():
        matched, total, out = apply_styles(styles_path, qc, args.in_place)
        print(f"styles: {matched}/{total} rows linked -> {out}")

    if conflicts:
        print("QC conflicts skipped:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
