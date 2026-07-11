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
| `n_oxidized_recovered_spots`, `guide_area_incl_oxidized_mm2`, `guide_cov_incl_oxidized_pct` | *oxidised-inclusive* guide (see below) — reported **separately** |
| `brown_frac`, `degraded_flag` | preservation control (browning fades pigment) |

Spots are found with the **pigment index `a* − b*`** (CIELAB): purple guide is
strongly positive, orange-brown degradation strongly negative, cream tissue
negative — a clean separator. Colour values are **not** reported; colour is used
only to locate the spots. Individual dots are isolated with a top-hat + adaptive
threshold, so faint/absent guides read as `cov ≈ 0` correctly. The primary
`guide_cov_pct` / `guide_present` therefore stay a strict **purple-pigment**
measurement.

**Oxidised-inclusive guide (separate columns).** On aged specimens many of the
fine guide stipples have oxidised toward brown, so they read as `a* − b* < 0` and
are spectrally indistinguishable from degradation — the purple index cannot
recover them by colour. They are measured separately (never folded into the
primary trait): dark, locally reddish pinpoints are recovered, but **only inside
the field of a confirmed purple guide** (a corolla whose strong-purple coverage
clears `OXIDIZED_SEED_MIN_COV_PCT`, default 1 %). Guide-absent / degraded
corollas build no field, so their `guide_cov_incl_oxidized_pct` equals
`guide_cov_pct` and `guide_present` is unaffected. Use the primary purple columns
for a conservative, colour-defined guide; use the `*_incl_oxidized*` columns when
aged/oxidised stipples should count as guide (e.g. Shikinejima `⑤`, where purple
coverage ≈ 11 % but oxidised-inclusive coverage ≈ 23 %). In the QC overlays these
recovered pinpoints are drawn **magenta**, distinct from cyan strong-purple and
blue weak-recovered guide.

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

## Sampling hierarchy

```
island  >  site (地点)  >  individual / plant (個体)  >  flower (corolla)
             |                    |                          |
   plain number in filename;  handwritten CIRCLED number    1–2 per individual
   coordinates in location.xlsx (per site)   on the sheet
```

- **Site** = the plain (non-circled) number; its lat/lon is in `location.xlsx`
  (`island, no, lat, lon`, where `no` = site). One sheet may hold several sites
  (e.g. `toshima3~6` = sites 3–6); a single-site island keeps one site for the
  whole sheet (**Shikinejima = 1 site, 5 individuals**).
- **Individual (株/個体)** = the CIRCLED number on the sheet, 1–2 flowers each.

`measure_guides.py --locations location.xlsx` auto-attaches `site_no`/`site_lat`/
`site_lon` for **single-site sheets** (79/209 corollas). Multi-site sheets can't
be split by site automatically — the corolla→(site, individual) link needs the
circled numbers, filled in `qc_plant_labels.csv` (`site_no_FILL`,
`individual_FILL`, `flower_no_FILL`) against `overlays/`. For Pst: average flowers
within individual, then among-island vs among-individual-within-island.

### Preparation & QC flags

- **`fold_check`** — corollas are scanned folded-in-half *or* fully opened.
  Folding is longitudinal, so **corolla length and every ratio trait
  (`guide_cov_pct`, `spot_density_cm2`, `guide_extent_rel`) are fold-robust**,
  but absolute `corolla_area/width/guide_area` are ~halved when folded.
  `fold_check=check` (width/length < 0.55) marks likely-folded corollas; confirm
  in `fold_state_FILL`. Both folded and expanded specimens ARE measured.
- **`merge_check`** — two touching corollas can be segmented as one blob
  (over-long/over-wide, e.g. length > 55 mm — a single corolla is ≤ ~55 mm).
  `merge_check=check` marks these for a manual split or exclusion
  (`split_or_exclude_FILL`).

## Usage

```bash
pip install -r requirements.txt
python measure_guides.py \
    --data-root "path/to/shimahotarubukuro" \
    --locations "path/to/location.xlsx" \
    --out-dir results
```

`--locations` reads the real site table (`island, no, lat, lon`; island names
`shikine`/`kozu` are aliased to the folder names) and is optional — without it the
`site_*` columns stay blank.

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
