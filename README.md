# shimahotarubukuro — nectar-guide & floral-trait extraction

Image-analysis pipeline that measures **nectar-guide** and **floral-size** traits
from flat-bed scans of *Campanula microdonta* (シマホタルブクロ) corollas across
five Izu islands (Oshima, Toshima, Niijima, Shikinejima, Kozushima).

## Locked reviewed axis workflow

For reviewed floral-shape measurements, the longitudinal axis is **not** taken
from raw-image PCA and is not forced vertical/horizontal.

1. Start from the accepted per-corolla polygon/mask.
2. Search candidate axes and reflect the polygon across each candidate line.
3. Choose the axis that maximizes mask-vs-mirror overlap (IoU).
4. Constrain direction so the ruler/top side is the corolla base in the normalized scan layout.
5. Project the polygon-derived axis back onto the raw scan for visual QC.
6. Measure longitudinal traits along that axis and width traits perpendicular to it.
7. Fully opened 5-lobed corollas use this as the working flattened-display symmetry axis.
8. Half-folded corollas remain explicitly flagged: their axis describes the observed folded polygon, not the complete five-lobed flower.

The implementation is in `measure_guides_symmetry_axis.py`.

For the reviewed Shikinejima sheet `shikine1~4`, C1/C4/C5/C6 are fully opened
five-lobed corollas and C2/C3 are half-folded with about 2.5 lobes visible.

## Measurement principles

- Use the real ruler in each scan for metric calibration when reliable.
- Keep strict purple-pigment guide coverage separate from oxidised-inclusive recovery.
- Measure visible reproductive-organ candidates only when actually mounted; missing organs are allowed and never fabricated.
- Review one sheet at a time against the raw scan before accepting outputs.
