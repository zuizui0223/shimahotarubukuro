#!/usr/bin/env python3
"""Locate the raw specimen scan for a reviewed shimask annotation.

Utility shared across the pipeline: ``find_raw`` maps a shimask annotation stem to
its raw scan under ``shimahotarubukuro/<island>/``. The reviewed annotation
filenames were simplified for display and do not always match the original scan
stem, so a small fixed, auditable mapping is applied; no fuzzy image matching is
done.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")

RAW_STEM_BY_SHIMASK_STEM = {
    "niijiama1-2": "niijima1~2",
    "niijiama2-4": "niijima2~4",
    "oshima8-9": "oshima8~9",
    "oshima10-13": "oshima10~13",
    "oshima13-15": "oshima13~15",
    "shikine1": "shikine1~4",
    "toshima1-2": "toshima1~2",
    "toshima2-3": "toshima2~3",
    "toshima3-6": "toshima3~6",
    "toshima6-8": "toshima6~8",
}


def find_raw(shimask_stem: str, raw_root: Path) -> tuple[str, Path]:
    raw_stem = RAW_STEM_BY_SHIMASK_STEM.get(shimask_stem.lower(), shimask_stem)
    matches: list[tuple[str, Path]] = []
    for folder in ISLAND_FOLDERS:
        island_dir = raw_root / folder
        if not island_dir.is_dir():
            continue
        for path in island_dir.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and path.stem.lower() == raw_stem.lower()
            ):
                matches.append((folder, path))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one raw scan for shimask={shimask_stem!r} "
            f"mapped_raw_stem={raw_stem!r}, found {len(matches)}: {matches}"
        )
    return matches[0]
