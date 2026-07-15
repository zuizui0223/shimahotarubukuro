# Automatic-first v3 validation

## Full shimask evaluation (`measure_guides_v3_refine13.py`)

Scored against every reviewed shimask sheet (red = corolla boundary GT, green =
reproductive-organ GT). shimask is used only for scoring, never at runtime. The
detector runs on the raw scans in about six seconds per sheet, so the CI job
now evaluates all 20 reviewed sheets rather than a four-sheet focus subset.

| Metric (mean over 20 sheets) | Previous (`refine12`, colour + Hough) | `refine13` |
|---|---:|---:|
| Corolla boundary recall | 0.41 | **0.55** |
| Corolla boundary precision | 0.72 | **0.86** |
| Organ recall @10 mm | ~0.02 | **0.87** |
| Organ precision @10 mm | ~0.50 | **0.85** |

Organ recall is 0.71-1.00 on every sheet. Kozushima, previously 0/10, is now
kozu1 10/10 and kozu2 9/12. The 167 s-per-sheet full-resolution Hough fallback
was removed; it matched almost no reviewed organs.

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
