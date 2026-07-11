# Reviewed corolla symmetry-axis QC

This workflow is locked for sheet-by-sheet floral-shape measurement.

## Principle

Use the accepted corolla polygon/mask first, then validate the resulting axis on the raw scan.

The axis is **not** taken from raw-image texture or a global PCA, and it is **not** forced horizontal or vertical.

## Algorithm

For each accepted corolla mask:

1. Generate candidate longitudinal axes within the biologically valid base-up family.
2. Reflect the mask across each candidate axis.
3. Compute overlap between the original mask and the reflected mask using IoU.
4. Select the axis with maximum IoU.
5. Orient the selected axis from the ruler/top side (corolla base) toward the lobe/tip side.
6. Project that same polygon-derived axis back onto the raw image for visual QC.
7. Measure longitudinal traits along the selected axis.
8. Measure width traits perpendicular to that axis.

## Fold handling

- Fully opened 5-lobed corollas: the selected axis is the working symmetry axis of the flattened display.
- Half-folded corollas: the selected axis describes only the observed folded polygon and must remain flagged `folded_half`; it is not interpreted as the full five-lobed floral symmetry axis.

## Shikinejima reviewed state

- C1: fully opened, 5 lobes
- C2: half-folded, about 2.5 lobes visible
- C3: half-folded, about 2.5 lobes visible
- C4: fully opened, 5 lobes
- C5: fully opened, 5 lobes
- C6: fully opened, 5 lobes

## QC acceptance rule

A result is accepted only after both views agree:

- mask view: the axis follows the polygon's symmetry;
- raw view: the projected axis is biologically plausible for the actual specimen.

No manual freehand axis is used as the primary measurement axis. Manual review is used to accept/reject the algorithmic result and to flag folded or damaged specimens.
