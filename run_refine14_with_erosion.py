# -*- coding: utf-8 -*-
"""Run refine14 with a selected fixed corolla edge erosion distance."""
from __future__ import annotations

import argparse
import sys

import measure_guides_v3_refine13 as refine13


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--erosion-mm", type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not 0.0 <= args.erosion_mm <= 1.5:
        raise SystemExit("--erosion-mm must be between 0 and 1.5")
    refine13.COROLLA_EDGE_ERODE_MM = float(args.erosion_mm)
    sys.argv = [sys.argv[0], *remaining]
    import measure_guides_v3_refine14 as refine14
    refine14.main()


if __name__ == "__main__":
    main()
