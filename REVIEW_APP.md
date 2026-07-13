# Review app — image-grounded floral trait correction

Interactive per-sheet QC for the flattened *Campanula microdonta* corollas. Fix the
things automation gets wrong: the **corolla mask**, **central axis**, measurement
cross-sections, pigment regions, and reproductive organs mounted on the sheet. The focused
output contains pollination-related traits that can be traced back to reviewed
image evidence.

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

1. **Sidebar** — pick the sheet, then a corolla (C1…CN).
2. **Mask workspace** — drag white circular handles to move polygon vertices, drag
   diamond handles to move an edge, or grab the green outline to insert a new
   vertex. Brush mode remains available for local add/subtract corrections.
3. **Shape workspace** — drag both endpoints of the corolla axis, maximum span,
   throat span, and basal-tube span. Put the axis base at the basal end and the tip
   on the central-lobe tip. Put the throat line at the lobe-sinus height. Measure
   axis-to-edge on an open corolla and the full visible width on a folded half.
4. **Pigment workspace** — review purple guide, oxidized guide, and brown/degraded
   regions. Final guide area, coverage, and presence use the union of purple and
   oxidized guide regions.
5. **Stamen/pistil workspace** — auto-detect sheet-level organ candidates, assign
   independent O numbers, then review each centre-line length, organ type, and
   identity confidence. C-to-O correspondence remains unconfirmed at this stage;
   undetected organs can be added with a new O number.
6. **Confirmation workspace** — inspect the focused pollination-trait table, set
   the fold state, exclusion, note, and review-complete flag.
7. **Save state** persists to `results/review_state/<sheet>.json` (resume later).
8. **Export** writes `results/reviewed/<sheet>/app_review/` including:
   `reviewed_axes.csv`, `human_review.csv`, `reviewed_exclusions.csv`,
   `reviewed_mask_corrections.csv`, `reviewed_measurement_guides.csv`,
   `reviewed_region_corrections.csv`, `reviewed_reproductive_organs.csv`,
   `reviewed_pollination_traits.csv`, and `axis_overrides_snippet.py`.

## Full-open and half-folded standardization

The four linear traits use the same half-corolla unit in both states. For `open`,
draw maximum, throat, and basal widths from the axis to one outer edge. For
`folded_half`, draw each width across the visible half-corolla. Corolla length is
measured directly in both states. No circumference or diameter conversion is used.

Area traits use a full-corolla unit: `open` mask and guide areas are used as
observed, while `folded_half` areas are doubled. Coverage, relative opening, and
tube expansion ratios are not multiplied. For `unknown`, comparison values stay
blank. `fold_state_reviewed`, `area_correction_factor`, and `area_scope` preserve
the area correction in the export.

Corolla decisions are stored by C number. Organ decisions are stored separately by
O number. `nearest_corolla_hint` is only a spatial hint; exported `corolla_id` stays
blank with `association_status=unconfirmed` until correspondence is reviewed later.

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
