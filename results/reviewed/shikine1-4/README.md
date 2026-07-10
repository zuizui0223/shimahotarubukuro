# Shikinejima `shikine1~4` reviewed QC

This directory contains numeric reviewed outputs only. The raw scan and QC overlay images remain private and are not committed.

## Confirmed specimen interpretation

- Five handwritten circled individuals are present.
- Circled individual ③ contains two corollas: one folded and one fully opened.
- Total measured corollas: **6**.
- The transparent-tape branch attached near the open flower of individual ③ is removed using a raw-image local ROI correction.
- No reproductive organ is invented for a corolla when the organ is missing, damaged, hidden, or absent from the mount.
- Four visible reproductive-organ candidates are retained; organ identity and biological association remain manual QC fields.

## Nectar-guide extraction

Spots are extracted from the raw scan after reviewed corolla correction.

- Components smaller than **0.02 mm²** are treated as scan noise.
- Guide coverage uses all accepted guide pixels.
- Large connected guide clusters of at least **3.0 mm²** are retained for coverage but flagged because touching spots make the component count conservative.
- Brown/degraded tissue is represented separately and is not silently counted as a guide.

Reviewed result:

- C1–C5: guide coverage below 0.5%; `guide_present = 0`.
- C6: guide coverage **9.647%**, 366 accepted connected components, 7 large connected-cluster flags; `guide_present = 1`.

## Files

- `traits.csv`: reviewed corolla traits and guide summary.
- `spot_summary.csv`: one row per corolla.
- `organs.csv`: visible reproductive-organ candidates only.
- The reproducible per-component `spots.csv` and private spot-overlay images are produced by `qc_single_sheet.py` in the GitHub Actions artifact.

## Provenance

- Baseline: `8cfae03`.
- Reviewed corolla correction accepted from run 65.
- Spot extraction and final quality gate: workflow run 79.
- Source branch head used by run 79: `895f2e37ac53b38855090cb89a128934ea3fb32e`.
- PR merge-test commit used by the workflow: `81fdb0dd687d52ebdc5abcd265b92b3cd74342fb`.
