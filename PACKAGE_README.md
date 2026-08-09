# ShimaFlora

ShimaFlora is a Python toolkit for reproducible image-based phenotyping of floral morphology and intrafloral colour patterns from calibrated images and binary floral masks.

The package separates generic measurements from taxon-specific biological interpretation. The original *Campanula microdonta* workflow remains a case study and validation pipeline; reusable algorithms live in the public `shimaflora` API.

## Install from GitHub

Versioned release:

```bash
pip install "git+https://github.com/zuizui0223/shimahotarubukuro.git@v0.2.0"
```

Development branch before the v0.2.0 merge/release:

```bash
pip install "git+https://github.com/zuizui0223/shimahotarubukuro.git@agent/shimaflora-v0.2"
```

## Five-line quick start

```python
from shimaflora import synthetic_flower, measure_shape, measure_pattern, overlay_pattern

image, roi, pattern = synthetic_flower()
shape = measure_shape(roi, scale_mm_per_px=0.1)
pattern_traits = measure_pattern(pattern, roi)
qc_rgb = overlay_pattern(image, roi, pattern)
```

The synthetic example is generated deterministically and is freely redistributable, so tutorials and tests do not depend on a specimen image license. See `docs/quickstart.md` for an end-to-end example with real image-loading conventions.

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

## Visual QC

```python
from shimaflora import overlay_roi, overlay_morphology, overlay_pattern

roi_qc = overlay_roi(image, mask)
shape_qc = overlay_morphology(image, mask)
pattern_qc = overlay_pattern(image, mask, pattern_mask)
```

QC helpers return RGB `numpy.uint8` arrays and never write files. `overlay_morphology` shows the same orientation-free minimum-area geometry used by the generic morphology API; it does not silently assign a biological base or tip. OpenCV BGR images are supported with `input_order="bgr"`.

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
- Return QC images as arrays so notebooks, apps, and publication pipelines can choose their own output format.

## v0.2 scope

In addition to v0.1 morphology and colour-pattern measurement, v0.2 adds reusable QC overlays, a redistributable synthetic flower example, an end-to-end tutorial, software citation metadata, and Zenodo-ready archival metadata.

Still out of scope: automatic flower detection, SAM/YOLO segmentation, multispectral/UV calibration, 3-D reconstruction, or taxon-independent automatic biological labeling of colour components.

## Citation and archiving

Use the versioned software release used in your analysis. `CITATION.cff` records the software citation independently from the biological *Campanula* manuscript. `.zenodo.json` is provided so tagged GitHub releases can be archived with consistent metadata once the repository is connected to Zenodo.

For reproducible work, record the exact release tag (for example `v0.2.0`) or commit SHA in the Methods or software availability statement.

## Release safety

GitHub Releases are created only after the `reproduce-pipeline` workflow succeeds on `main`. That workflow first checks the reusable package against all 218 reviewed corollas and then verifies the tracked publication tables, so a package release cannot silently precede the biological regression gate.

## Research pipeline

The full *Campanula microdonta* specimen workflow, including reviewed annotations, reproductive-organ measurements, island analyses, Pst estimates and audit figures, remains documented in the repository root `README.md` and run through `run_pipeline.sh`.
