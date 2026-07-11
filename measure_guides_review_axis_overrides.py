# -*- coding: utf-8 -*-
"""Human-reviewed corolla-axis and tiny-mask overrides.

Only add entries after the sheet has been inspected against both the accepted
corolla mask and the raw scan.  These overrides are intentionally sparse: the
automatic polygon-symmetry result remains the default for every unspecified
corolla.
"""
from __future__ import annotations

from typing import Mapping


REVIEWED_AXIS_OVERRIDES: Mapping[tuple[str, str, int], dict[str, float | str]] = {
    # Oshima / oshima1 — accepted after human review.
    # C1/C3 retain the accepted PRE-QC axis geometry except for the reviewed
    # start-point placement recorded below.  C4 keeps its original PRE-QC start
    # point and the human-reviewed end point aligned with the central/third-petal
    # guide stripe.  C2 keeps its accepted end point and only shifts the start
    # slightly rightward toward the centre.
    ("oshima", "oshima1", 1): {
        "base_x": 643.000,
        "base_y": 789.000,
        "tip_x": 662.970,
        "tip_y": 1304.575,
        "review_status": "ACCEPT",
        "review_note": "Accepted after removing only the tiny top-right mask noise spur.",
    },
    ("oshima", "oshima1", 2): {
        "base_x": 1409.000,
        "base_y": 865.000,
        "tip_x": 1474.462,
        "tip_y": 1349.243,
        "review_status": "ACCEPT",
        "review_note": "End point retained; start moved slightly right toward the visual centre.",
    },
    ("oshima", "oshima1", 3): {
        "base_x": 662.366,
        "base_y": 1649.097,
        "tip_x": 688.790,
        "tip_y": 2153.293,
        "review_status": "ACCEPT",
        "review_note": "Accepted unchanged from PRE-QC.",
    },
    ("oshima", "oshima1", 4): {
        "base_x": 1635.580,
        "base_y": 1698.427,
        "tip_x": 1607.659,
        "tip_y": 2186.048,
        "review_status": "ACCEPT",
        "review_note": "Original PRE-QC start retained; reviewed end follows the central/third-petal guide direction.",
    },
}


# Tiny mask correction only; coordinates are in the canonical ruler-at-top
# orientation used by QC.  Do not replace the full corolla mask with a new
# segmentation.  Subtract only this small polygon from the already accepted C1
# mask.
REVIEWED_MASK_SUBTRACT_POLYGONS: Mapping[tuple[str, str, int], tuple[tuple[float, float], ...]] = {
    ("oshima", "oshima1", 1): (
        (758.0, 676.0),
        (872.0, 676.0),
        (872.0, 739.0),
        (815.0, 739.0),
        (778.0, 722.0),
        (758.0, 704.0),
    ),
}


def get_axis_override(folder: str, sheet: str, corolla_id: int):
    return REVIEWED_AXIS_OVERRIDES.get((folder.lower(), sheet.lower(), int(corolla_id)))


def get_mask_subtract_polygon(folder: str, sheet: str, corolla_id: int):
    return REVIEWED_MASK_SUBTRACT_POLYGONS.get((folder.lower(), sheet.lower(), int(corolla_id)))
