"""Shared island ordering, labels and publication colours.

The historical module name is retained because the numbered-index and final figure
scripts import these constants. The former flower-level island-comparison plot was
removed: all inferential island comparisons now use plant means, site correction and
Pst in ``island_analysis.py``.
"""
from __future__ import annotations

ORDER = ["Oshima", "Toshima", "Niijima", "Shikinejima", "Kozushima"]
COLOUR = {
    "Oshima": "#0072B2",
    "Toshima": "#E69F00",
    "Niijima": "#009E73",
    "Shikinejima": "#D55E00",
    "Kozushima": "#CC79A7",
}
INK = "#1a1a19"
MUTED = "#6b6b68"
GRID = "#e7e7e3"


def island_of(sheet: str) -> str:
    """Map a scan-sheet stem to its publication island label."""
    for prefix, name in (
        ("oshima", "Oshima"),
        ("toshima", "Toshima"),
        ("niij", "Niijima"),
        ("shikine", "Shikinejima"),
        ("kozu", "Kozushima"),
    ):
        if sheet.startswith(prefix):
            return name
    raise ValueError(f"unrecognised island in sheet name: {sheet}")
