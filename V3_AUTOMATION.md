# Automatic-first v3 extraction

`measure_guides_v3.py` is a non-Streamlit prototype that turns manual review
into exception handling.

## What it produces

- `traits_v3.csv`: accepted corolla measurements plus mask confidence
- `organs_v3.csv`: detached reproductive-organ candidates and associations
- `qc_required.csv`: only low/medium-confidence records that need a person
- `masks/<island>/<sheet>/C*.png`: one binary mask per detected corolla
- `overlays/*.png`: confidence-coloured full-sheet previews

Green corollas are high confidence, orange are medium confidence and red are
low confidence. The script never treats opening width as measured. It writes
`opening_width_status=deferred`. `auto_max_width_mm` is retained only as a
provisional value and is marked `provisional_not_for_primary_analysis`.

## Run

```bash
python measure_guides_v3.py \
  --data-root shimahotarubukuro \
  --out-dir results_v3 \
  --locations locations.csv
```

The `--locations` argument is optional. To disable the inherited v2 touching-
corolla split during comparison:

```bash
python measure_guides_v3.py \
  --data-root shimahotarubukuro \
  --out-dir results_v3_no_split \
  --no-auto-split
```

## Intended QC workflow

1. Run the command on all sheets.
2. Inspect `overlays/` and `qc_required.csv`.
3. Do not redraw high-confidence masks.
4. For corollas in `qc_required.csv`, choose only `accept`, `exclude`, or
   `split` in a later lightweight correction step.
5. For organs in `qc_required.csv`, choose only `pistil`, `stamen_bundle`,
   `unknown`, or `noise`.

The current organ labels are deliberately conservative:

- `pistil_candidate`
- `stamen_bundle_candidate`
- `reproductive_organ_unknown`
- `fragment_or_noise`

They are based on skeleton length, width profile, terminal swelling, branching,
and association with the nearest corolla. The first real-data run should be
used to tune thresholds before these labels are used in biological analysis.

## Validation

Synthetic tests cover:

- automatic acceptance of a clean corolla mask
- QC routing for an unresolved split and border contact
- skeleton length measurement
- swollen-tip pistil-like geometry
- branched stamen-bundle-like geometry
- endpoint-distance and axis-based corolla association

Run them with:

```bash
python -m unittest -v tests/test_measure_guides_v3.py
```
