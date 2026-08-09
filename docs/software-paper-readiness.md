# Software-paper readiness

ShimaFlora should be evaluated as reusable floral image-phenotyping software, while
the *Campanula microdonta* biological analysis remains a validation case study.

## Required before submission

- [x] Versioned installable package
- [x] Open-source license
- [x] Public API separated from taxon-specific presets
- [x] Automated tests on multiple Python versions
- [x] Numerical regression against the 218-corolla case study
- [x] Redistributable end-to-end example without specimen-image licensing ambiguity
- [x] Visual QC API
- [x] CITATION.cff
- [x] Zenodo-ready metadata
- [ ] Archive a tagged release on Zenodo and record the DOI
- [ ] Add the DOI badge to package documentation
- [ ] Add API reference pages generated from public docstrings
- [ ] Add at least one independent floral morphology/pattern example beyond Campanula
- [ ] Freeze a v0.2.x API for manuscript review

## JOSS framing

The software contribution is a reproducible, mask-first framework for jointly
quantifying calibrated 2-D floral morphology and spatial intrafloral colour patterns.
The paper should emphasize: (1) segmentation-method independence, (2) separation of
measurement from biological interpretation, (3) reproducible colour-component
modelling, (4) spatial/coverage pattern metrics, and (5) numerical validation against
a reviewed biological dataset.

A JOSS submission will additionally need a short `paper.md` and bibliography in the
required format, a clear statement of need, installation/example instructions, tests,
and an archived release DOI.

## Applications in Plant Sciences framing

For a methods/software-oriented manuscript, expand validation beyond the software
engineering checks: demonstrate transfer to at least one contrasting floral form or
pattern type, report repeatability/sensitivity to mask perturbation or image
calibration, and discuss the biological interpretation limits of 2-D flattened or
photographed floral material.

## Separation of citations

The software release and the biological Campanula manuscript should remain separately
citable. Users applying ShimaFlora to other taxa should not be required to cite the
Campanula biological conclusions merely to identify the software they used.
