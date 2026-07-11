# shimahotarubukuro — nectar-guide & floral-trait extraction

Image-analysis pipeline for flattened *Campanula microdonta* corollas.

## Reviewed symmetry-axis method

For floral-shape measurements, use the accepted per-corolla polygon/mask first.
Do not derive the axis from raw-image texture and do not force it vertical or horizontal.

1. Estimate the longitudinal axis by reflecting the polygon across candidate lines.
2. Select the axis that maximizes polygon-vs-mirror overlap (IoU).
3. Keep the ruler/top side as the corolla-base side in normalized scans.
4. Project the selected axis back onto the raw scan for visual QC.
5. Measure longitudinal traits along the axis and width traits perpendicular to it.
6. Fully opened five-lobed corollas use this as the working flattened-display symmetry axis.
7. Half-folded corollas stay explicitly flagged: their axis describes the observed folded polygon, not the complete five-lobed flower.

Implementation: `measure_guides_symmetry_axis.py`.

Reviewed Shikinejima `shikine1~4` state:
- C1, C4, C5, C6: fully opened, five lobes.
- C2, C3: half-folded, about 2.5 lobes visible.

Other safeguards remain unchanged: ruler-based metric calibration, separate strict-purple and oxidised-inclusive guide traits, visible-only reproductive-organ measurement, and one-sheet-at-a-time raw-scan QC.
