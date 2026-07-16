#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the established reviewed analysis for every confirmed shimask sheet.

Each label image is paired with its raw scan using the repository's existing
``find_raw`` resolver.  The per-sheet analysis remains exactly
``analyze_confirmed_single_sheet.py``: only the red corolla annotations and green
reproductive-organ annotations replace the two unreliable automatic inputs.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from evaluate_v3_against_shimask_v2 import find_raw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("shimask"))
    parser.add_argument("--raw-root", type=Path, default=Path("shimahotarubukuro"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_confirmed_all"))
    args = parser.parse_args()

    label_files = sorted(
        path for path in args.labels.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(label_files) != 20:
        raise SystemExit(f"Expected 20 shimask images, found {len(label_files)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    failures: list[str] = []

    for index, label_path in enumerate(label_files, start=1):
        try:
            raw_path = find_raw(label_path, args.raw_root)
            folder = raw_path.parent.name
            sheet_out = args.out_dir / label_path.stem
            command = [
                sys.executable,
                "analyze_confirmed_single_sheet.py",
                "--raw", str(raw_path),
                "--shimask", str(label_path),
                "--folder", folder,
                "--out-dir", str(sheet_out),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            sheet_out.mkdir(parents=True, exist_ok=True)
            (sheet_out / "pipeline_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (sheet_out / "pipeline_stderr.txt").write_text(completed.stderr, encoding="utf-8")
            status = "success" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                failures.append(f"{label_path.name}: exit={completed.returncode}")
        except Exception as exc:
            raw_path = Path("")
            folder = ""
            status = "failed"
            failures.append(f"{label_path.name}: {type(exc).__name__}: {exc}")

        manifest.append(
            {
                "index": index,
                "shimask": str(label_path),
                "raw": str(raw_path),
                "folder": folder,
                "status": status,
            }
        )
        print(f"[{index:02d}/20] [{status}] {label_path.name}", flush=True)

    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    (args.out_dir / "SUMMARY.txt").write_text(
        f"total={len(manifest)}\nsuccess={sum(row['status'] == 'success' for row in manifest)}\n"
        f"failed={len(failures)}\n",
        encoding="utf-8",
    )
    if failures:
        (args.out_dir / "FAILURES.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} of 20 confirmed sheets failed")

    print(f"Completed all {len(manifest)} confirmed sheets -> {args.out_dir}")


if __name__ == "__main__":
    main()
