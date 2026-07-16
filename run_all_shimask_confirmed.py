#!/usr/bin/env python3
"""Run qc_single_sheet_shimask.py for every reviewed shimask image.

Each sheet is isolated in its own subprocess and output directory. A manifest and
per-sheet stdout/stderr are always retained so one failure does not hide results
from the other reviewed sheets.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")


def find_raw(stem: str, raw_root: Path) -> tuple[str, Path]:
    matches: list[tuple[str, Path]] = []
    for folder in ISLAND_FOLDERS:
        island_dir = raw_root / folder
        if not island_dir.is_dir():
            continue
        for path in island_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.lower() == stem.lower():
                matches.append((folder, path))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw scan for {stem!r}, found {len(matches)}: {matches}")
    return matches[0]


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("shimask"))
    parser.add_argument("--raw-root", type=Path, default=Path("shimahotarubukuro"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_shimask_all"))
    args = parser.parse_args()

    labels = sorted(
        p for p in args.labels.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(labels) != 20:
        raise SystemExit(f"Expected exactly 20 shimask images, found {len(labels)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for label in labels:
        sheet = label.stem
        sheet_out = args.out_dir / "sheets" / sheet
        sheet_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, object] = {
            "sheet": sheet,
            "shimask_path": str(label),
            "raw_path": "",
            "island_folder": "",
            "status": "failed",
            "return_code": "",
            "n_traits": 0,
            "n_organs": 0,
            "n_spot_summaries": 0,
            "error": "",
        }
        try:
            folder, raw = find_raw(sheet, args.raw_root)
            row["raw_path"] = str(raw)
            row["island_folder"] = folder
            command = [
                sys.executable,
                "qc_single_sheet_shimask.py",
                "--image", str(raw),
                "--shimask", str(label),
                "--folder", folder,
                "--out-dir", str(sheet_out),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            row["return_code"] = completed.returncode
            (sheet_out / "pipeline_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (sheet_out / "pipeline_stderr.txt").write_text(completed.stderr, encoding="utf-8")
            row["n_traits"] = count_csv_rows(sheet_out / "traits.csv")
            row["n_organs"] = count_csv_rows(sheet_out / "organs.csv")
            row["n_spot_summaries"] = count_csv_rows(sheet_out / "spot_summary.csv")
            required = [
                sheet_out / "traits.csv",
                sheet_out / "organs.csv",
                sheet_out / "spot_summary.csv",
                sheet_out / "visitor_traits.csv",
                sheet_out / "scale_calibration.csv",
            ]
            if completed.returncode != 0:
                raise RuntimeError(f"pipeline exit {completed.returncode}")
            missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise RuntimeError("missing outputs: " + ", ".join(missing))
            if int(row["n_traits"]) <= 0 or int(row["n_spot_summaries"]) <= 0:
                raise RuntimeError("empty trait or spot output")
            row["status"] = "success"
            print(f"[success] {folder}/{sheet}: traits={row['n_traits']} organs={row['n_organs']}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[failed] {sheet}: {exc}")
        manifest.append(row)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    successes = sum(row["status"] == "success" for row in manifest)
    failures = len(manifest) - successes
    (args.out_dir / "SUMMARY.txt").write_text(
        f"total={len(manifest)}\nsuccess={successes}\nfailed={failures}\n",
        encoding="utf-8",
    )
    print(f"Completed {len(manifest)} sheets: success={successes} failed={failures}")
    if failures:
        raise SystemExit(f"{failures} sheet(s) failed; see {manifest_path}")


if __name__ == "__main__":
    main()
