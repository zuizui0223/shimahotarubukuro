# ShimaFlora

ShimaFlora is a Python toolkit for reproducible image-based phenotyping of floral morphology and intrafloral colour patterns from calibrated images and binary floral masks.

The package separates generic measurements from taxon-specific biological interpretation. The original *Campanula microdonta* workflow remains a case study and validation pipeline; reusable algorithms live in the public `shimaflora` API.

## Install from GitHub

```bash
pip install "git+https://github.com/zuizui0223/shimahotarubukuro.git"
```

To install this development branch before merge:

```bash
pip install "git+https://github.com/zuizui0223/shimahotarubukuro.git@agent/package-shimaflora-v0.1"
```

## Generic morphology

```python
import cv2
from shimaflora import measure_shape, width_profile, width_at

mask = cv2.imread("flower_mask.png", cv2.IMREAD_GRAYSCALE) > 0
traits = measure_shape(mask, scale_mm_per_px=0.0847)
profile = width_profile(mask)
mid_width_px = width_at(mask, 0.5)
```

`measure_shape` returns area, major/minor axes, aspect ratio, perimeter, solidity, circularity and orientation without assuming a taxon or biological proximal/distal direction.

## Generic intrafloral colour patterns

```python
import cv2
from shimaflora import ColorPatternModel, measure_pattern

images = [cv2.imread(p) for p in image_paths]
masks = [cv2.imread(p, 0) > 0 for p in mask_paths]

model = ColorPatternModel(n_components=3, channels=("a", "b"), random_state=1)
model.fit(images, masks)

# Components are intentionally unlabeled in the generic API.
pattern_mask = model.segment(images[0], masks[0], component=1, min_posterior=0.5)
pattern_traits = measure_pattern(pattern_mask, masks[0])
```

The default colour representation uses CIELAB chromatic channels `(a*, b*)`, excluding lightness so that illumination, folds and shading do not automatically define colour classes. Equal per-ROI sampling prevents large or strongly pigmented specimens from dominating a global model.

## Campanula preset

```python
from shimaflora.presets import CampanulaGuidePreset, campanula_tubular_traits

roles = CampanulaGuidePreset().assign_roles(model.component_centres_)
guide = model.segment(image, mask, component=roles["guide"], min_posterior=0.5)
traits = campanula_tubular_traits(mask, scale_mm_per_px=0.0847)
```

The preset contains the biological assumptions used by the current flattened bellflower workflow, while the generic core remains reusable for other taxa and other intrafloral patterns.

## Design principles

- Accept externally defined masks rather than forcing one segmentation method.
- Keep measurement separate from biological interpretation.
- Prefer continuous first-order quantities such as pattern coverage over fragile spot counts.
- Retain optional connected-component and spatial descriptors for taxa where discrete marks are meaningful.
- Make assumptions explicit through presets rather than hidden constants in generic functions.
- Support visual and numerical QC in publication pipelines.

## Current scope (v0.1)

Included: calibrated 2-D morphology, oriented width profiles, unsupervised Lab colour-component modelling, posterior-threshold segmentation, pattern coverage/component count/centroid/dispersion, and a Campanula preset.

Not yet included: automatic flower detection, SAM/YOLO segmentation, multispectral/UV calibration, 3-D reconstruction, or taxon-independent automatic biological labeling of colour components.

## Research pipeline

The full *Campanula microdonta* specimen workflow, including reviewed annotations, reproductive-organ measurements, island analyses, Pst estimates and audit figures, remains documented in the repository root `README.md` and run through `run_pipeline.sh`.
