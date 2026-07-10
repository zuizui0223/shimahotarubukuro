# shimahotarubukuro — nectar-guide & floral-trait extraction

Image-analysis pipeline that measures **nectar-guide** and **floral-size** traits
from flat-bed scans of *Campanula microdonta* (シマホタルブクロ) corollas across
five Izu islands (Oshima, Toshima, Niijima, Shikinejima, Kozushima).

It is the fully-measured *Campanula* "calibration seed" for the Izu island-rule
study (companion repo: `campanula-channel-identification`), and provides the
**phenotypic (Pst) side** of a planned Fst–Pst comparison.

## What it measures

Each scan sheet holds several opened/flattened corollas (1–2 flowers per plant)
plus a printed cm ruler. **All scans are 300 DPI**, verified from JPEG metadata
and cross-checked against the ruler, so the scale is fixed:

```
px/cm = 300 / 2.54 = 118.11      (1 px = 0.08467 mm ; 1 px² = 0.007168 mm²)
```

**Robust traits** (per corolla) — validated: island-mean corolla length
reproduces Inoue's independent common-garden values for Oshima (38.8 vs 39.31 mm)
and Toshima (34.9 vs 35.27 mm):

| trait | meaning |
|---|---|
| `corolla_len_mm`, `corolla_width_mm`, `corolla_area_mm2` | flower size / "display" axis |
| `guide_area_mm2`, `guide_cov_pct` | nectar-guide (purple spot) amount |
| `n_spots`, `spot_density_cm2` | guide pattern (dotted vs solid) |
| `guide_extent_rel` | how far the guide reaches up from the base |
| `guide_present` | guide coverage ≥ 0.5 % |
| `brown_frac`, `degraded_flag` | preservation control (browning fades pigment) |

Spots are found with the **pigment index `a* − b*`** (CIELAB): purple guide is
strongly positive, orange-brown degradation strongly negative, cream tissue
negative — a clean separator. Colour values are **not** reported; colour is used
only to locate the spots. Individual dots are isolated with a top-hat + adaptive
threshold, so faint/absent guides read as `cov ≈ 0` correctly.

**Provisional traits (UNRELIABLE — do not use as-is):** `prov_mouth_diam_mm`,
`prov_tube_depth_mm`, `prov_n_lobes`. Flattening removes the 3-D funnel geometry,
folded-vs-open preparation changes width→girth, and torn/curled edges defeat lobe
& sinus detection (detected lobe count scatters 3–7 vs the true 5). For **mouth
diameter and inner ("length-to-nectar") depth**, measure fresh flowers with
calipers or a lateral calibrated photo instead (`caliper` / `calibrated_photo`
methods in the companion repo's `field_flower_geometry` schema).

**Reproductive organs** (`styles.csv`): thin sticks laid beside the corollas
(likely styles/pistils) are detected and length-measured, but **require manual
QC** (link to plant; drop tape/filament artefacts). Stamen length and herkogamy
are **not** recoverable from these detached, flattened specimens.

## Sampling unit

**Plant (株)** is the unit; 1–2 flowers (corollas) per plant. Corollas are measured
individually; the handwritten circled numbers on each sheet are the ground-truth
plant IDs. Fill `qc_plant_labels.csv` (`plant_id_FILL`, `exclude_FILL`) against
the overlays before aggregating. For Pst, average flowers within a plant, then
compute among-island vs among-plant-within-island variance.

## Usage

```bash
pip install -r requirements.txt
python measure_guides.py \
    --data-root "path/to/shimahotarubukuro" \
    --out-dir results
```

Data-root layout (island sub-folders, English names):

```
oshima/  toshima/  niijima/  shikinejima/  kozushima/     # *.jpg scan sheets
```

## Plant IDs and coordinates

After manually filling `results/qc_plant_labels.csv`, propagate the checked plant
IDs to the flower and style tables. `--in-place` makes one-time backups named
`traits.pre_qc.csv` and `styles.pre_qc.csv` before replacing the originals.

```bash
python apply_qc_labels.py --results-dir results --in-place
```

Then attach GPS data from `location.xlsx`:

```powershell
python attach_locations.py `
  --locations "C:\Users\zuizui\OneDrive - Kyoto University\デスクトップ\location.xlsx" `
  --results-dir results
```

The workbook may use Japanese or English headers and decimal-degree or DMS
coordinates. The preferred columns are:

```
island, sheet, plant_id, site_id, latitude, longitude,
elevation_m, coord_datum, coord_accuracy_m, coordinate_notes
```

Only `latitude` and `longitude` are mandatory, but reliable plant-level matching
normally requires `island + plant_id` or `island + sheet + plant_id`. A header-only
example is provided at `data/location_template.csv`.

Matching is conservative, in this order:

1. `island + sheet + plant_id`
2. `island + plant_id`
3. `island + sheet`
4. island alone, only when that island has one unique coordinate

Ambiguous coordinates are never guessed. They remain unmatched and are listed in
`results/with_coordinates/location_join_report.csv`.

## Outputs (`results/`)

- `traits.csv` — one row per corolla (all traits above)
- `styles.csv` — detected organ sticks (needs QC)
- `qc_plant_labels.csv` — template to lock plant IDs / exclusions
- `per_island_summary.csv` — island means
- `overlays/` — per-sheet QC images (corolla = green, spots = cyan, sticks =
  orange). *Not committed by default* — they embed the specimen scans; regenerate
  locally.

Coordinate-enriched outputs are written without altering the source workbook:

- `with_coordinates/locations_normalized.csv` — cleaned location rows
- `with_coordinates/traits_with_coordinates.csv`
- `with_coordinates/qc_plant_labels_with_coordinates.csv`
- `with_coordinates/styles_with_coordinates.csv`
- `with_coordinates/per_location_summary.csv`
- `with_coordinates/location_join_report.csv` — unmatched or ambiguous rows

## Caveats before analysis

1. **Preservation confound** — browning fades pigment, so `guide_cov` is a lower
   bound on degraded specimens; screen with `degraded_flag`.
2. **Uneven / small n** — Shikinejima especially; wide Pst confidence intervals.
3. **Corolla length** absolute values may differ from Inoue's definition (pressing
   splays lobes), but island means and ranks agree.
4. Raw scans are kept out of the repo (`.gitignore`); document their location
   separately.
5. Exact occurrence coordinates can be sensitive. Keep the repository private or
   publish a coarsened coordinate table when releasing the data publicly.

## Notes

Not comparable across preparation modes for width-derived traits; folded vs fully
opened corollas must be handled consistently. See module docstring in
`measure_guides.py` for method detail.
