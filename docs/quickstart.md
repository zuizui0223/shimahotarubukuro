# ShimaFlora quickstart

This tutorial uses a deterministic synthetic flower, so it can be copied, tested,
and redistributed without specimen-image licensing constraints.

```python
from shimaflora import (
    measure_shape,
    measure_pattern,
    overlay_morphology,
    overlay_pattern,
    synthetic_flower,
)

image, roi, pattern = synthetic_flower(size=256)

shape = measure_shape(roi, scale_mm_per_px=0.10)
pattern_traits = measure_pattern(pattern, roi)

shape_qc = overlay_morphology(image, roi)
pattern_qc = overlay_pattern(image, roi, pattern)
```

`shape` contains calibrated area, major/minor axes, aspect ratio, perimeter,
solidity, circularity, and geometric orientation. `pattern_traits` contains pattern
coverage, connected-component count, relative centroid, and spatial dispersion.

The two QC arrays are ordinary RGB `numpy.uint8` arrays. Save them with your image
library of choice, display them in a notebook, or return them from an application.
ShimaFlora deliberately does not choose filenames or write them automatically.

## Using your own images

ShimaFlora starts after a floral ROI has been defined. The ROI may come from manual
annotation, ImageJ, Photoshop, SAM, YOLO, or another segmentation system.

```python
import cv2
from shimaflora import measure_shape, overlay_morphology

image_bgr = cv2.imread("flower.jpg")
roi = cv2.imread("flower_mask.png", cv2.IMREAD_GRAYSCALE) > 0

traits = measure_shape(roi, scale_mm_per_px=0.0847)
qc_rgb = overlay_morphology(image_bgr, roi, input_order="bgr")
```

For colour-pattern discovery, fit `ColorPatternModel` across a biologically sensible
set of floral ROIs, inspect its component centres and QC overlays, and only then
assign biological meanings such as nectar guide, bullseye, oxidation, or background
tissue. Generic ShimaFlora components are intentionally unlabeled.
