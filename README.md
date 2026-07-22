# *Campanula microdonta* floral-trait pipeline

Publication-ready, reproducible measurement and island-divergence pipeline for
flattened, pressed *Campanula microdonta* corollas from five Izu Islands: Oshima,
Toshima, Niijima, Shikinejima and Kozushima (20 scan sheets, 218 corollas, 125
individual plants).

The repository follows one final route from reviewed image annotations to plant-level,
site-corrected global and pairwise Pst. Trial automatic segmentation, flower-level
island tests, dried-specimen colour measurements and the confounded Oshima-versus-
other-islands Bombus contrast are not part of the publication pipeline.

## Repository layout

```text
src/shimaflora/
├── core/          shared scan, annotation, island and file-discovery utilities
├── measurement/   corolla, guide and reproductive-organ measurements
├── metadata/      authoritative field-metadata integration
├── analysis/      guide spatial tests and global/pairwise Pst
├── figures/       manuscript figures and tables
└── audit/         numbered indexes, overlays and measurement cards
```

`run_pipeline.sh` remains the single stable entry point. It configures the source
search path and executes the modules in publication order, so the directory cleanup
does not change the measurements or output schemas. See `src/shimaflora/README.md`
for the responsibility of each package.

## Authoritative inputs

| Path | Role |
| --- | --- |
| `shimahotarubukuro/<island>/<sheet>.jpg` | 300-DPI, ruler-calibrated specimen scans |
| `shimask/<sheet>.jpg` | reviewed red corolla outlines and green reproductive-organ traces |
| `mask.zip` | clean iPhone-lifted corolla silhouettes used only to refine the size ROI |
| `withlocation.csv` | field provenance, island/site, coordinates, plant identity, corolla number and sexual phase |

The reviewed annotations define which corolla and organ are measured. The iPhone
silhouette replaces the hand boundary for corolla size only when it registers to the
reviewed ROI; the scan ruler remains the metric scale. Unmatched objects use the hand
ROI.

## Retained measurements

### Corolla size and shape

- corolla length
- observed and full-flower-equivalent width
- observed and full-flower-equivalent area
- flattened proximal throat width
- flattened distal mouth width
- corolla aspect ratio
- tube-flare ratio
- lobe-incision depth and ratio

Half-folded corollas retain their observed length; transverse width and area are
multiplied by two for full-flower-equivalent comparisons. Throat and mouth widths are
2-D proxies from equally flattened specimens, not reconstructed 3-D entrance
diameters.

### Reproductive organ

The reviewed green trace is measured end to end and reported as
`organ_length_mm`. Image analysis does not reclassify it as style or stamen. Sexual
phase (`s`/`p`) comes from the field table. `organ_corolla_ratio` is organ length
divided by corolla length.

### Nectar guide

The retained amount metric is `guide_coverage_pct`, the area percentage of the
reviewed corolla ROI classified as purple/magenta guide pixels. Colour-free spatial
metrics test basal and petal-midline concentration against random pixels from the same
ROI.

The following are intentionally **not measured or analysed**:

- discrete guide spot count
- spot density derived from that count
- dried-specimen guide colour, saturation or chromatic contrast
- guide reach or colour-based conspicuousness
- a causal Bombus-present versus Bombus-absent island effect

These quantities are unstable under spot merging/splitting, scan threshold and
specimen fading, or are inseparable from island geography in this dataset.

## Reproduce

```bash
pip install -r requirements.txt
bash run_pipeline.sh
```

The same pipeline runs in GitHub Actions and uploads tables and figures as artifacts.

## Final analysis design

1. Measure 218 reviewed corollas.
2. Join the authoritative field metadata without renumbering.
3. Average the 1-2 flowers from each plant to one plant mean (`n = 125`).
4. Test the island effect with island as a fixed effect and site as a random intercept.
5. Correct trait-wise p-values with Benjamini-Hochberg.
6. Estimate global Pst with a plant bootstrap 95% CI.
7. Estimate Pst separately for every island pair.

Pst is used as a phenotypic divergence surrogate for ranking traits. It is not Qst and
is not, by itself, evidence of selection.

## Key outputs (`results_shimask_all/`)

| File | Contents |
| --- | --- |
| **`corolla_master.csv`** | final field metadata × retained measurements, one row per corolla |
| `corolla_traits_final.csv` | final corolla size, guide coverage and organ length |
| `pollination_traits.csv` | supported flattened 2-D morphometrics and ratios |
| `guide_spatial.csv` | colour-free guide spatial metrics per sufficiently guided corolla |
| **`plant_means.csv`** | one row per individual plant, used for inference |
| **`island_analysis_stats.csv`** | global Pst, bootstrap CI, site-corrected test, KW comparison and latitude correlation |
| **`island_pst_pairwise.csv`** | pairwise Pst for every retained trait and island pair |
| `island_divergence_table.csv` | paper table of site-corrected significant traits |
| `island_divergence.png` | significant-trait Pst forest and top latitude plots |
| `island_pst_pairwise.png` | pairwise-Pst heatmaps for significant traits |
| `guide_spatial_structure.png` | guide placement versus within-ROI random null |
| `measure_cards/<sheet>.png` | flower-by-flower visual measurement audit |
| `numbered_index/<sheet>.png` | continuous 1-218 corolla/organ index |

## Pipeline order

`measurement/remeasure_medial` -> `measurement/register_iphone_masks` ->
`measurement/guide_traits` -> `measurement/organ_traits` ->
`measurement/merge_traits` -> `measurement/pollination_traits` ->
`audit/make_numbered_index` -> `metadata/integrate_metadata` ->
`analysis/guide_spatial` -> `analysis/island_analysis` -> publication figures and
audit cards.

The morphometric rationale follows Nagano et al. (2014, *Ecology and Evolution*
4:3819; doi:10.1002/ece3.1191), restricted here to measurements supported by the
flattened specimens.
