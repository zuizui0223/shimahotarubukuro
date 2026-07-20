# *Campanula microdonta* floral-trait pipeline

Reproducible image-analysis pipeline that measures pollination-relevant floral traits
from flattened, pressed *Campanula microdonta* corollas scanned across five Izu
Islands (Oshima, Toshima, Niijima, Shikinejima, Kozushima; 20 sheets, 218 corollas).

Everything needed to reproduce the results is committed to the repository, so the
whole analysis runs from a clean checkout with no local dependencies (see
[Reproduce](#reproduce)).

## Inputs

| Path | What it is |
| --- | --- |
| `shimahotarubukuro/<island>/<sheet>.jpg` | raw specimen scans (ruler-calibrated) |
| `shimask/<sheet>.jpg` | reviewer's hand annotations — corolla outlines in red, reproductive-organ strokes in green |
| `mask.zip` | iPhone-lifted clean per-corolla silhouettes (subject masks) |
| `withlocation.csv` | field table: date, island, site, lat/lon, individual plant, corolla number (`collar` = the continuous 1–218 number), organ number, and sexual phase (`s` = staminate/male, `p` = pistillate/female) |

The hand annotations are the backbone (corolla identification and numbering, organ
strokes, nectar-guide detection, and the registration target). The iPhone silhouettes
refine only the corolla-size ROI; the pipeline runs without them by falling back to
the hand ROI.

## Reproduce

```bash
pip install -r requirements.txt
bash run_pipeline.sh
```

This regenerates `results_shimask_all/` — the measured tables and the figures /
per-flower measurement cards. The same run happens automatically on GitHub
(`.github/workflows/pipeline.yml`), which uploads the tables and figures as
artifacts.

## Key outputs (`results_shimask_all/`)

| File | Contents |
| --- | --- |
| **`corolla_master.csv`** | **final integrated table** — field metadata (locality, individual, sexual phase) joined to every measured trait, one row per corolla |
| `corolla_traits_final.csv` | corolla length / width / area (iPhone-registered ROI, hand fallback) + nectar-guide + organ |
| `pollination_traits.csv` | throat & mouth width (corolla-entrance proxies), lobe incision, aspect, tube flare, style/corolla ratio, guide conspicuousness |
| `organ_traits.csv` | reproductive-organ length from the reviewer's green strokes |
| `guide_traits.csv` | nectar-guide coverage, spot count, density |
| `guide_divergence_stats.csv` | bumblebee-absence test (Oshima has *Bombus ardens*; others do not) |
| `guide_spatial.csv` | non-random basal + petal-midline concentration of the guide spots |
| `global_index.csv` | the continuous 1–218 corolla numbering ↔ sheet / corolla id |
| `measure_cards/<sheet>.png` | per-flower cards showing how each trait is measured |
| `numbered_index/<sheet>.png` | each sheet with corollas and organs numbered 1–218 |

Traits follow Nagano et al. 2014 (*Ecol. Evol.* 4:3819,
[doi:10.1002/ece3.1191](https://doi.org/10.1002/ece3.1191)) for the bumblebee-fit
morphometrics, extended with nectar-guide and reproductive-phase traits.

## Pipeline stages

`remeasure_medial` → `register_iphone_masks` → `guide_traits` → `organ_traits` →
`merge_traits` → `pollination_traits` → `make_numbered_index` → `integrate_metadata`
→ `guide_divergence` → `guide_spatial` → figures (`plot_*`, `make_overlays`,
`make_measure_cards`). `run_pipeline.sh` runs them in this order.

Figures embed the specimen scans and are git-ignored; regenerate them locally or from
the CI artifacts.
