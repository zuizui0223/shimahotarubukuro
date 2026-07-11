# shimahotarubukuro — nectar-guide & floral-trait extraction

Image-analysis pipeline that measures **nectar-guide** and **floral-size** traits
from flat-bed scans of *Campanula microdonta* (シマホタルブクロ) corollas across
five Izu islands (Oshima, Toshima, Niijima, Shikinejima, Kozushima).

It is the fully-measured *Campanula* "calibration seed" for the Izu island-rule
study (companion repo: `campanula-channel-identification`), and provides the
**phenotypic (Pst) side** of a planned Fst–Pst comparison.

## Reviewed symmetry-axis workflow

For reviewed floral-shape measurements, the longitudinal axis is **not** taken
from raw-image PCA and is not forced vertical/horizontal. Start from the accepted
per-corolla polygon/mask, search candidate axes, reflect the polygon across each
candidate line, and choose the axis that maximizes mask-vs-mirror overlap (IoU).
The direction is constrained so the ruler/top side is the corolla base in the
normalized scan layout. The polygon-derived axis is then projected back onto the
raw scan for visual QC. Longitudinal traits are measured along this axis and
width traits perpendicular to it.

Fully opened five-lobed corollas use this as the working symmetry axis of the
flattened display. Half-folded corollas remain explicitly flagged: their axis
describes the observed folded polygon, not the complete five-lobed flower.
Implementation: `measure_guides_symmetry_axis.py`.

For reviewed Shikinejima `shikine1~4`, C1/C4/C5/C6 are fully opened five-lobed
corollas and C2/C3 are half-folded with about 2.5 lobes visible.

## Measurement safeguards

- Use the real ruler in each scan for metric calibration when reliable.
- Keep strict purple-pigment guide coverage separate from oxidised-inclusive recovery.
- Measure visible reproductive-organ candidates only when actually mounted; missing organs are allowed and never fabricated.
- Review one sheet at a time against the raw scan before accepting outputs.

See the repository history for the full legacy workflow documentation; this branch
adds the reviewed symmetry-axis method without changing the raw-data privacy policy.
