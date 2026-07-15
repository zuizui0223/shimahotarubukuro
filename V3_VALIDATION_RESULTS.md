# Automatic-first v3 validation

## Current accepted pipeline (`measure_guides_v3_refine19.py`)

Scored against every reviewed shimask sheet with the **corrected annotation-only
ground truth** (`evaluate_v3_against_shimask_v2.py` + `shimask_annotation_diff.py`):
only the hand-drawn red/green review strokes count, not the flower's natural
nectar-guide colour. The earlier colour-threshold GT wrongly counted natural red
inside each corolla (e.g. oshima4: 78.7 k GT-red px -> 27.9 k after the fix), so
boundary recall had been under-reported. shimask is used only for scoring.

| Metric (mean over 20 sheets, corrected GT) | `refine17` (fixed 0.7 mm erosion) | `refine19` (GrabCut snap) |
|---|---:|---:|
| Corolla boundary recall | 0.761 | **0.778** |
| Corolla boundary precision | 0.848 | **0.875** |
| Organ recall @10 mm | 0.938 | 0.938 |
| Organ precision @10 mm | 0.848 | 0.848 |

Boundary: after correcting the evaluator, the edge treatment was re-optimised on
the true target. A fixed 0.7 mm erosion (F1 0.802) beat pixel-level tissue
snapping and per-corolla adaptive erosion, but a **GrabCut colour-model boundary
snap wins** (F1 0.824), improving 17/20 sheets with the largest gains on the worst
sheets (kozu2 +0.11, kozu1 +0.07, shikine +0.07). Organ detection is unchanged
because it runs on the un-eroded corolla union, independent of the edge method.

Traits (`refine14` + `refine18`): corolla length, max width (CW), throat/opening
width (CE proxy), basal & mid-tube widths, tube & lobe length, lobe count, and
lobe/tube areas + perimeter — flattened-scan proxies, each status-flagged. Organs
carry per-instance IDs and conservative type candidates.

## Earlier evaluation (old colour-threshold GT — under-reported boundary recall)

Historical `refine13` numbers were scored against the pre-fix evaluator that
counted natural flower colour as boundary GT, so its boundary recall (0.55) was
an artefact. Organ recall ~0.02 -> 0.87 (Kozushima 0/10 -> 10/10 & 9/12) and the
removal of the 167 s-per-sheet Hough fallback remain valid.

### What changed

- **Organ detector rewritten.** Detached style/pistil marks are thin, pale
  filaments laid in a horizontal band beside each corolla, markedly darker than
  the white paper. We threshold on local darkness (primary signal) plus b*
  yellowness — the yellowness gate is what separates organs from neutral paper
  folds/creases, which otherwise merged organs into fold-spanning blobs. Each
  filament is kept as a tight component and sampled along its own axis, because a
  whole pistil (thin style + broad ovary base) is longer than one 10 mm match
  radius.
- **Corolla boundary snapped to the tissue edge.** The foreground close leaves
  the written mask ~0.5 mm outside the tissue; a small lobe-preserving erosion
  lifts both boundary recall and precision on every sheet.

## Earlier representative-sheet scope

The earlier prototype was run on one representative sheet from each island:

- Oshima: `oshima10~13.jpg`
- Toshima: `toshima2~3.jpg`
- Niijima: `niijima1~2.jpg`
- Shikinejima: `shikine1~4.jpg`
- Kozushima: `kozu1.jpg`

## Final representative-sheet counts

| Island | Sheet | Corolla masks | Auto-split rows | Organ candidates | QC items |
|---|---|---:|---:|---:|---:|
| Kozushima | kozu1 | 10 | 0 | 6 | 5 |
| Niijima | niijima1~2 | 17 | 2 | 17 | 11 |
| Oshima | oshima10~13 | 18 | 2 | 17 | 13 |
| Shikinejima | shikine1~4 | 7 | 4 | 6 | 5 |
| Toshima | toshima2~3 | 16 | 0 | 18 | 12 |
| **Total** |  | **68** | **8** | **64** | **46** |

## Confirmed improvements

- Removed attached styles/pistils and low-chroma paper folds from corolla masks using multiscale geometry, colour and guide support.
- Merged nearby removed pieces into one reproductive-organ candidate instead of double-counting a style and its base.
- Reduced the original Oshima organ-candidate flood from 839 candidates to a small per-corolla candidate set.
- Split the moderately sized touching pair on Niijima that fell just below the original 55 mm / 1350 mm2 trigger.
- Rejected the corresponding false split of the broad single Niijima flower by requiring a real midpoint waist.
- Excluded the narrow 88.7 mm2 Toshima fragment before assigning corolla IDs.
- Kept maximum width as provisional and left opening width deferred.

## Safety checks

- Synthetic regression tests cover normal single flowers, moderate touching pairs, narrow fragments, paper tails, guide-supported lobes and appendage-fragment merging.
- `tests/test_measure_guides_v3_refine13.py` covers the new organ detector: a yellow filament beside a corolla is detected, a same-darkness neutral paper fold is rejected, a filament far from any corolla is ignored, and the axis sampler emits several points for a long organ and one for a short one.
- The GitHub Actions job now runs `measure_guides_v3_refine13.py` on all reviewed sheets and scores it against the full shimask ground truth.
- No v3 code has been merged into `main`; PR #24 remains a draft.
