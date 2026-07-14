# Automatic-first v3 validation

## Scope

The current prototype was run on one representative sheet from each island:

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
- The five-island GitHub Actions validation completed successfully.
- No v3 code has been merged into `main`; PR #24 remains a draft.
