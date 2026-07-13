# Review app — manual mask & axis correction

Interactive per-sheet QC for the flattened *Campanula microdonta* corollas. Fix the
things automation gets wrong — the **corolla mask** (e.g. paper shadow that leaked
in, as on oshima10~13 C17/C18) and the **central axis** (midline) — plus record
exclude / fold-state / visible-pistil per corolla.

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
2. **Axis tab (click)** — pick *BASE* or *TIP* and click the point on the flower.
   Put the base at the throat/top-centre and the tip on the true central-lobe tip.
   *Reset axis to PRE-QC* restores the automatic axis.
3. **Mask tab (drag to paint)** — set a brush size, choose *SUBTRACT* (erase
   shadow/noise) or *ADD* (recover tissue), then **drag over the region** on the
   flower and press **Apply paint**. The stroke is converted to a mask polygon;
   *Undo last mask edit* reverts one. (Drag painting uses `streamlit-drawable-canvas`.)
4. Live **length / width / area (mm)** update as you edit (300 DPI scale).
6. **Save state** persists to `results/review_state/<sheet>.json` (resume later).
7. **Export** writes `results/reviewed/<sheet>/app_review/`:
   `reviewed_axes.csv`, `human_review.csv`, `reviewed_exclusions.csv`,
   `reviewed_mask_corrections.csv`, and `axis_overrides_snippet.py` (paste into
   `REVIEWED_AXIS_OVERRIDES`).

Reviewed decisions are per corolla and never touch the others — the automatic PRE-QC
result stays the default for anything you don't change.

## Notes

- Orientation is derived from the overlay + committed centroids (robust to the
  per-sheet `SHEET_ROTATION`), so masks/axes always line up with the raw scan.
- Mask edits store polygons (not new segmentations); the accepted outline stays the
  base and your polygons are subtracted/added on top.
