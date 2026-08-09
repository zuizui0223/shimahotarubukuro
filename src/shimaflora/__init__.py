"""ShimaFlora: image-based floral phenotyping."""

from .morphology import measure_shape, width_profile, width_at
from .pattern import ColorPatternModel, measure_pattern
from .qc import overlay_roi, overlay_morphology, overlay_pattern
from .examples import synthetic_flower

__all__ = [
    "measure_shape",
    "width_profile",
    "width_at",
    "ColorPatternModel",
    "measure_pattern",
    "overlay_roi",
    "overlay_morphology",
    "overlay_pattern",
    "synthetic_flower",
]

__version__ = "0.2.0"
