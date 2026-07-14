# -*- coding: utf-8 -*-
"""Safe entrypoint for the multiscale v3 floral-mask refinement."""
from __future__ import annotations

import numpy as np

import measure_guides_v3_refine as refine

# Save the conservative implementation before importing refine2, because refine2
# intentionally patches the refine module globals used by process_sheet.
_CONSERVATIVE_DETACH = refine.detach_thin_appendages

import measure_guides_v3_refine2 as refine2  # noqa: E402

_MULTISCALE_DETACH = refine2.detach_thin_appendages


def safe_multiscale_detach(mask: np.ndarray):
    """Run multiscale cleanup with a non-recursive conservative fallback."""
    previous = refine.detach_thin_appendages
    # refine2's safety branch calls refine.detach_thin_appendages. Point that
    # name at the original implementation only for the duration of this call.
    refine.detach_thin_appendages = _CONSERVATIVE_DETACH
    try:
        return _MULTISCALE_DETACH(mask)
    finally:
        refine.detach_thin_appendages = previous


refine.detach_thin_appendages = safe_multiscale_detach


def main() -> None:
    refine.main()


if __name__ == "__main__":
    main()
