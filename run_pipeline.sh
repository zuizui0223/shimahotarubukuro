#!/usr/bin/env bash
# Reproduce the publication floral-trait pipeline through pairwise Pst.
set -euo pipefail
cd "$(dirname "$0")"

SRC="$PWD/src/shimaflora"
export PYTHONPATH="$SRC/core:$SRC/measurement:$SRC/metadata:$SRC/analysis:$SRC/figures:$SRC/audit${PYTHONPATH:+:$PYTHONPATH}"
run_py() { python3 "$SRC/$1"; }

echo ">> unpack iPhone silhouettes"
rm -rf mask __MACOSX
unzip -q -o mask.zip -x '__MACOSX/*' -d .
test -d mask

mkdir -p results_shimask_all
rm -f \
  results_shimask_all/guide_divergence_stats.csv \
  results_shimask_all/guide_bombus_hypothesis.png \
  results_shimask_all/guide_density_islands.png \
  results_shimask_all/island_guide_traits.png \
  results_shimask_all/island_pollination_traits.png \
  results_shimask_all/island_corolla_size.png \
  results_shimask_all/island_organ_length.png

echo ">> 1/12 reviewed hand-ROI corolla size     -> medial_traits.csv"
run_py measurement/remeasure_medial.py
echo ">> 2/12 iPhone-registered size ROI         -> iphone_traits.csv"
run_py measurement/register_iphone_masks.py
echo ">> 3/12 area-based nectar-guide coverage   -> guide_traits.csv"
run_py measurement/guide_traits.py
echo ">> 4/12 reviewed reproductive-organ length -> organ_traits.csv"
run_py measurement/organ_traits.py
echo ">> 5/12 retained per-corolla measurements  -> corolla_traits_final.csv"
run_py measurement/merge_traits.py
echo ">> 6/12 supported 2-D morphometrics        -> pollination_traits.csv"
run_py measurement/pollination_traits.py
echo ">> 7/12 continuous 1-218 numbering         -> global_index.csv"
run_py audit/make_numbered_index.py
echo ">> 8/12 integrate authoritative metadata   -> corolla_master.csv"
run_py metadata/integrate_metadata.py
echo ">> 9/12 colour-free guide spatial tests     -> guide_spatial.csv"
run_py analysis/guide_spatial.py
echo ">> 10/12 plant/site-corrected global + pairwise Pst"
run_py analysis/island_analysis.py

echo ">> 11/12 publication figures and tables"
run_py figures/plot_guide_spatial.py
run_py figures/plot_island_analysis.py

echo ">> 12/12 measurement overlays and per-flower cards"
run_py audit/make_overlays.py
run_py audit/make_measure_cards.py

echo ">> DONE"
echo "Final corolla table: results_shimask_all/corolla_master.csv"
echo "Final global analysis: results_shimask_all/island_analysis_stats.csv"
echo "Final pairwise Pst: results_shimask_all/island_pst_pairwise.csv"
ls -1 results_shimask_all/*.csv
