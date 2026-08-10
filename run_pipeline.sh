#!/usr/bin/env bash
# Reproduce the publication floral-trait pipeline through multivariate Pst analogue.
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

echo ">> 1/13 reviewed hand-ROI corolla size     -> medial_traits.csv"
run_py measurement/remeasure_medial.py
echo ">> 2/13 iPhone-registered size ROI         -> iphone_traits.csv"
run_py measurement/register_iphone_masks.py
echo ">> 3/13 area-based nectar-guide coverage   -> guide_traits.csv"
run_py measurement/guide_traits.py
echo ">> 4/13 reviewed reproductive-organ length -> organ_traits.csv"
run_py measurement/organ_traits.py
echo ">> 5/13 retained per-corolla measurements  -> corolla_traits_final.csv"
run_py measurement/merge_traits.py
echo ">> 6/13 supported 2-D morphometrics        -> pollination_traits.csv"
run_py measurement/pollination_traits.py
echo ">> 7/13 continuous 1-218 numbering         -> global_index.csv"
run_py audit/make_numbered_index.py
echo ">> 8/13 integrate authoritative metadata   -> corolla_master.csv"
run_py metadata/integrate_metadata.py
echo ">> 9/13 colour-free guide spatial tests     -> guide_spatial.csv"
run_py analysis/guide_spatial.py
echo ">> 10/13 plant/site-corrected global + pairwise Pst"
run_py analysis/island_analysis.py
echo ">> 11/13 multivariate phenotypic divergence"
run_py analysis/multivariate_phenotype.py

echo ">> 12/13 publication figures and tables"
run_py figures/plot_guide_spatial.py
run_py figures/plot_island_analysis.py

echo ">> 13/13 measurement overlays and per-flower cards"
run_py audit/make_overlays.py
run_py audit/make_measure_cards.py

echo ">> DONE"
echo "Final corolla table: results_shimask_all/corolla_master.csv"
echo "Final global analysis: results_shimask_all/island_analysis_stats.csv"
echo "Final pairwise Pst: results_shimask_all/island_pst_pairwise.csv"
echo "Multivariate axes: results_shimask_all/multivariate_pst_axes.csv"
ls -1 results_shimask_all/*.csv
