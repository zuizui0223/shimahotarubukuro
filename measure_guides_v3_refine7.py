# -*- coding: utf-8 -*-
"""Remove low-chroma paper tails and suppress tiny organ fragments.

The guide-supported pass removes side loops, but a broad paper fold can remain in
the largest opened core.  On the upright sheets this appears as a contiguous,
guide-free, low-chroma band at the image-bottom end of a corolla.  The rule is
conservative: it requires at least 1.5 mm of low-chroma tail and refuses to remove
more than 15% of the current mask.
"""
from __future__ import annotations

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine2 as refine2
import measure_guides_v3_refine5 as refine5
import measure_guides_v3_refine6 as refine6  # noqa: F401  (installs guide pruning)

_BASE_GUIDE_DETACH = refine.detach_thin_appendages
_ORIGINAL_CLASSIFY_STYLE = refine.classify_style_candidate

_TAIL_SEARCH_MM = 9.0
_TAIL_MIN_HEIGHT_MM = 1.5
_TAIL_CHROMA_MAX = 8.0


def classify_style_candidate(features: dict) -> tuple[str, float, str]:
    """Downgrade ambiguous cleanup pieces and neutral paper to noise."""
    label, confidence, reason = _ORIGINAL_CLASSIFY_STYLE(features)
    length = float(features.get("rect_length_mm", 0.0))
    chroma = float(features.get("median_chroma", 0.0))
    if label == "reproductive_organ_unknown" and chroma < 7.0:
        return "fragment_or_paper", 0.95, "low_chroma_paper_cleanup"
    if label == "reproductive_organ_unknown" and length < 6.5:
        return "fragment_or_paper", 0.92, "short_ambiguous_cleanup_fragment"
    if label == "style_or_pistil_candidate" and length < 5.0 and chroma < 8.0:
        return "fragment_or_paper", 0.90, "short_low_chroma_cleanup_fragment"
    return label, confidence, reason


def _bottom_tail(mask: np.ndarray) -> np.ndarray | None:
    channels = refine2._CURRENT_CHANNELS
    if channels is None:
        return None
    q = (np.asarray(mask) > 0).astype(np.uint8)
    if not q.any():
        return None

    _, a, b, _, _, _, chroma = channels
    spots = base.spot_segment(a, b, q.astype(bool))
    ys, xs = np.where(q > 0)
    lower = int(ys.max())
    upper_limit = max(int(ys.min()), lower - int(round(_TAIL_SEARCH_MM / float(base.MM_PX))))

    low_rows: list[int] = []
    high_run = 0
    started = False
    for y in range(lower, upper_limit - 1, -1):
        selected = q[y] > 0
        if int(selected.sum()) < 3:
            if started:
                high_run += 1
            continue
        row_chroma = float(np.median(chroma[y][selected]))
        row_spots = float(spots[y][selected].mean())
        is_low = row_chroma <= _TAIL_CHROMA_MAX and row_spots <= 0.01
        if is_low:
            low_rows.append(y)
            started = True
            high_run = 0
        elif started:
            high_run += 1
            if high_run >= 5:
                break

    if not low_rows:
        return None
    cut_y = min(low_rows)
    height_mm = (lower - cut_y + 1) * float(base.MM_PX)
    if height_mm < _TAIL_MIN_HEIGHT_MM:
        return None

    tail = np.zeros_like(q)
    tail[cut_y : lower + 1] = q[cut_y : lower + 1]
    selected = tail > 0
    if not selected.any():
        return None
    if float(np.median(chroma[selected])) > 7.5:
        return None
    if float(spots[selected].mean()) > 0.005:
        return None
    if int(tail.sum()) * float(base.MM2_PX) < 5.0:
        return None
    return tail


def detach_with_paper_tail(mask: np.ndarray):
    cleaned, removed = _BASE_GUIDE_DETACH(mask)
    tail = _bottom_tail(cleaned)
    if tail is None:
        return cleaned, removed
    if int(tail.sum()) > int(np.asarray(cleaned).astype(bool).sum()) * 0.15:
        return cleaned, removed

    candidate = (np.asarray(cleaned) > 0).astype(np.uint8) & (1 - tail)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n <= 1:
        return cleaned, removed
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    candidate = (labels == keep).astype(np.uint8)
    merged_removed = refine5.merge_removed_masks([*removed, tail])
    return candidate, merged_removed


refine.classify_style_candidate = classify_style_candidate
refine.detach_thin_appendages = detach_with_paper_tail


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
