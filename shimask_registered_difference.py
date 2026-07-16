# -*- coding: utf-8 -*-
"""Fallback registration for raw-vs-shimask annotation differencing.

Used only when direct resized differencing recovers no closed red outline. The raw
scan is aligned to the reviewed image with ECC before classifying changed pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

import shimask_input


def _ecc_aligned_raw(raw: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    raw_small = shimask_input._raw_at_annotation_resolution(raw, annotated)
    template = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    moving = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # An affine transform handles the small translation/scale/rotation differences
    # introduced when the review preview was exported.
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        300,
        1e-6,
    )
    try:
        cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC raw/shimask registration failed: {exc}") from exc

    return cv2.warpAffine(
        raw_small,
        warp,
        (annotated.shape[1], annotated.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def registered_stroke_masks(raw: np.ndarray, annotated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover red/green strokes after affine alignment of raw to shimask."""
    aligned = _ecc_aligned_raw(raw, annotated).astype(np.int16)
    review = annotated.astype(np.int16)
    delta = review - aligned
    absolute = np.abs(delta)
    changed = (
        (absolute.max(axis=2) >= shimask_input.DIFF_CHANNEL_MIN)
        & (absolute.sum(axis=2) >= shimask_input.DIFF_TOTAL_MIN)
    )

    b, g, r = cv2.split(review)
    red_dominance = r - np.maximum(g, b)
    green_dominance = g - np.maximum(r, b)
    red_delta = delta[:, :, 2] - np.maximum(delta[:, :, 1], delta[:, :, 0])
    green_delta = delta[:, :, 1] - np.maximum(delta[:, :, 2], delta[:, :, 0])

    red = changed & (red_dominance >= shimask_input.ANNOTATION_DOMINANCE_MIN) & (red_delta >= 12)
    green = changed & (green_dominance >= shimask_input.ANNOTATION_DOMINANCE_MIN) & (green_delta >= 12)
    return (
        shimask_input._remove_tiny_components(red.astype(np.uint8)),
        shimask_input._remove_tiny_components(green.astype(np.uint8)),
    )
