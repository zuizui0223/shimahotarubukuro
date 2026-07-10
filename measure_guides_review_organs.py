# -*- coding: utf-8 -*-
"""Reviewed organ detector.

For sheets that have been inspected visually, use the reviewed centre-lines and do
not mix them with generic Hough candidates. Unreviewed sheets retain the generic
fast detector.
"""
from __future__ import annotations

import numpy as np

import measure_guides_review as review
import measure_guides_review_fast as fast


def organs_reviewed(
    img: np.ndarray,
    corolla_mask: np.ndarray,
    top: int,
) -> list[dict]:
    manual = review.manual_organ_rows(img.shape)
    if manual is not None:
        return manual
    return fast.organs_fast(img, corolla_mask, top)
