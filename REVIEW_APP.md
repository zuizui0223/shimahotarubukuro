# Review app — image-grounded floral trait correction

Interactive per-sheet QC for the flattened *Campanula microdonta* corollas. Fix the
things automation gets wrong: the **corolla mask**, **central axis**, measurement
cross-sections, pigment regions, and reproductive-organ contamination. Derived
trait values are recalculated from the reviewed image evidence.

Everything is shown in the **canonical ruler-at-top** frame that the pipeline uses.

## Install & run

```bash
pip install -r requirements.txt -r requirements-app.txt
streamlit run review_app.py
```

## Prerequisite: PRE-QC per sheet

The app reads the accepted per-corolla outlines from a **canonical reviewed overlay**
and the committed corolla ids from `results/reviewed/<sheet>/traits.csv`. Extract a
sheet first if it has no overlay yet:

```bash
python qc_single_sheet.py --image "shimahotarubukuro/oshima/oshima10~13.jpg" \
    --folder oshima --out-dir results/review_cache/oshima10-13
```

(The app auto-finds overlays under `results/review_cache/…`, `results_single/…`, or
`results_all_review/…`. If none exists it prints the exact command to run.)

## Using it

1. **Sidebar** — pick the sheet, then a corolla (C1…CN). Per-corolla flags live here
   (Exclude +reason, Fold state open/folded_half, Visible pistil).
2. **Axis workspace (click)** — pick *BASE* or *TIP* and click the point on the flower.
   Put the base at the throat/top-centre and the tip on the true central-lobe tip.
   *Reset axis to PRE-QC* restores the automatic axis.
3. **Mask workspace** — drag white circular handles to move polygon vertices, drag
   diamond handles to move an edge, or grab the green outline to insert a new
   vertex. Brush mode remains available for local add/subtract corrections.
4. **Measurements workspace** — review maximum span, throat/lobe boundary,
   mid-tube width, basal-tube width, and visible lobe count directly on the flower.
   Tube length, lobe length, mouth proxy, entrance area, and shape ratios update
   from these reviewed guides.
5. **Pigment workspace** — review purple guide, oxidized guide, and brown/degraded
   regions with add/subtract brushes. Coverage, count, density, placement, and area
   statistics are recalculated from the corrected regions.
6. **Organs workspace** — adjust the seeded organ centre-line and set its width.
   The seed comes from the detector or from a thin appendage in the flower mask. Use
   *Outline + shape* for an organ incorrectly joined to the flower silhouette, or
   *Pigment only* when it lies over a petal and should not create a hole in area.
7. **All traits workspace** — audit every per-flower trait with its original value,
   reviewed value, and source. Identifiers, coordinates, and sheet-level ruler
   calibration stay protected; manual overrides remain available for traits that
   cannot be reconstructed from the scan.
8. **Save state** persists to `results/review_state/<sheet>.json` (resume later).
9. **Export** writes `results/reviewed/<sheet>/app_review/` including:
   `reviewed_axes.csv`, `human_review.csv`, `reviewed_exclusions.csv`,
   `reviewed_mask_corrections.csv`, `reviewed_measurement_guides.csv`,
   `reviewed_organ_exclusions.csv`, `reviewed_region_corrections.csv`,
   `reviewed_traits.csv`, `reviewed_trait_overrides.csv`, and
   `axis_overrides_snippet.py`.

Reviewed decisions are per corolla and never touch the others — the automatic PRE-QC
result stays the default for anything you don't change.

## Deploying to Streamlit Cloud

`requirements.txt` (repo root, the only file Cloud reads) already pins the app deps
and uses **opencv-python-headless** (plain `opencv-python` fails on Cloud with a
`libGL` ImportError).

Two caveats on Cloud:

- **Masks need a committed overlay per sheet.** The app reads the canonical reviewed
  overlay from `results/review_overlays/<sheet>/`. Only `oshima10-13` is committed
  there; commit the others (copy each `results_single/<sheet>/overlays/*.png` into
  `results/review_overlays/<sheet>/`) to review them on Cloud.
- **State is ephemeral on Cloud** — *Save state* / *Export* write files that are lost
  on restart/redeploy. For real reviewing, **run locally** (`streamlit run
  review_app.py`) so your `results/review_state/` persists.

## Notes

- Orientation is derived from the overlay + committed centroids (robust to the
  per-sheet `SHEET_ROTATION`), so masks/axes always line up with the raw scan.
- Mask and pigment edits store polygons (not raster screenshots). Measurement and
  organ corrections store calibrated image coordinates, so every reviewed number
  remains traceable to visible evidence.
