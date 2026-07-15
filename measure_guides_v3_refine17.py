# -*- coding: utf-8 -*-
"""Accepted v3 entry point: refine14 traits with 0.7 mm corolla edge erosion."""
from __future__ import annotations

import measure_guides_v3_refine13 as refine13

# Full 20-sheet benchmark: best boundary F1 with only -1.41% mean area change.
refine13.COROLLA_EDGE_ERODE_MM = 0.7

import measure_guides_v3_refine14 as refine14  # noqa: E402


def main() -> None:
    refine14.main()


if __name__ == "__main__":
    main()
