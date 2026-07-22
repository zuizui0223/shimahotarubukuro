# Source layout

The publication pipeline is grouped by responsibility:

- `core/` — scan calibration, reviewed-annotation extraction, island metadata and file lookup
- `measurement/` — corolla, guide and reproductive-organ measurements and final trait assembly
- `metadata/` — authoritative field metadata integration
- `analysis/` — colour-free guide spatial tests and global/pairwise Pst
- `figures/` — manuscript figures and tables
- `audit/` — numbered indexes, overlays and per-flower measurement cards

`run_pipeline.sh` is the stable public entry point. It configures the module search path and runs these stages in publication order. Individual stages may also be run after sourcing `scripts/env.sh`.
