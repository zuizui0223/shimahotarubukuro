"""Campanula-specific interpretations of generic ShimaFlora measurements."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from shimaflora.morphology import measure_shape, width_profile


@dataclass(frozen=True)
class CampanulaGuidePreset:
    """Identify likely guide/oxidation/petal GMM components in Lab a*,b* space."""

    def assign_roles(self, component_centres: np.ndarray) -> dict[str, int]:
        means = np.asarray(component_centres, dtype=float)
        if means.ndim != 2 or means.shape[1] != 2 or len(means) < 3:
            raise ValueError("CampanulaGuidePreset expects >=3 component centres in (a*, b*)")
        magenta_score = means[:, 0] - means[:, 1]
        guide = int(np.argmax(magenta_score))
        remaining = [i for i in range(len(means)) if i != guide]
        yellow_red_score = means[:, 0] + means[:, 1]
        oxidation = int(max(remaining, key=lambda i: yellow_red_score[i]))
        petal = int(min((i for i in remaining if i != oxidation), key=lambda i: np.linalg.norm(means[i])))
        return {"petal": petal, "guide": guide, "oxidation": oxidation}


def _band_median(profile: np.ndarray, lo: float, hi: float) -> float:
    n = len(profile)
    start = max(0, min(n - 1, int(lo * n)))
    stop = max(start + 1, min(n, int(hi * n)))
    return float(np.median(profile[start:stop]))


def campanula_tubular_traits(
    mask: np.ndarray,
    scale_mm_per_px: float = 1.0,
    *,
    width_factor: float = 1.0,
) -> dict[str, float]:
    """Campanula-style flattened tubular-flower traits.

    This preset deliberately keeps Campanula assumptions outside the generic core.
    `width_factor=2` can reproduce full-flower-equivalent transverse measurements
    for reviewed half-folded specimens.
    """
    shape = measure_shape(mask, scale_mm_per_px=scale_mm_per_px)
    prof = width_profile(mask, axis="major")
    throat = _band_median(prof, 0.04, 0.16) * scale_mm_per_px * width_factor
    mouth = _band_median(prof, 0.72, 0.88) * scale_mm_per_px * width_factor
    length = shape["major_axis_mm"]
    width = shape["minor_axis_mm"] * width_factor
    return {
        "corolla_length_mm": length,
        "corolla_width_mm": width,
        "throat_width_mm": throat,
        "mouth_width_mm": mouth,
        "corolla_aspect_L_W": length / max(width, 1e-12),
        "tube_flare_W_throat": width / max(throat, 1e-12),
    }
