#!/usr/bin/env bash
# Reproduce the publication floral-trait pipeline through complex syndrome diagnostics.
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
  results_shimask_all/island_organ_length.png \
  results_shimask_all/multivariate_*.csv \
  results_shimask_all/syndrome_*.csv

echo ">> 1/18 reviewed hand-ROI corolla size     -> medial_traits.csv"
run_py measurement/remeasure_medial.py
echo ">> 2/18 iPhone-registered size ROI         -> iphone_traits.csv"
run_py measurement/register_iphone_masks.py
echo ">> 3/18 area-based nectar-guide coverage   -> guide_traits.csv"
run_py measurement/guide_traits.py
echo ">> 4/18 reviewed reproductive-organ length -> organ_traits.csv"
run_py measurement/organ_traits.py
echo ">> 5/18 retained per-corolla measurements  -> corolla_traits_final.csv"
run_py measurement/merge_traits.py
echo ">> 6/18 supported 2-D morphometrics        -> pollination_traits.csv"
run_py measurement/pollination_traits.py
echo ">> 7/18 continuous 1-218 numbering         -> global_index.csv"
run_py audit/make_numbered_index.py
echo ">> 8/18 integrate authoritative metadata   -> corolla_master.csv"
run_py metadata/integrate_metadata.py
echo ">> 9/18 colour-free guide spatial tests     -> guide_spatial.csv"
run_py analysis/guide_spatial.py
echo ">> 10/18 plant/site-corrected global + pairwise Pst"
run_py analysis/island_analysis.py
echo ">> 11/18 site-aware multivariate phenotypic divergence"
run_py analysis/multivariate_phenotype.py
echo ">> 12/18 pollination-syndrome module heterogeneity"
run_py analysis/syndrome_modules.py
echo ">> 13/18 extended complex-adaptation diagnostics"
run_py analysis/syndrome_extended.py
echo ">> 14/18 trait lability and selective signal disassembly"
run_py analysis/syndrome_lability.py
echo ">> 15/18 physical matching vs receiver-dependent signalling"
run_py analysis/signal_mechanical_contrast.py
echo ">> 16/18 divergence vs within-island phenotypic covariance"
run_py analysis/syndrome_covariance.py

echo ">> 17/18 publication figures and tables"
run_py figures/plot_guide_spatial.py
run_py figures/plot_island_analysis.py

echo ">> 18/18 measurement overlays and per-flower cards"
run_py audit/make_overlays.py
run_py audit/make_measure_cards.py

echo ">> DONE"
echo "Final corolla table: results_shimask_all/corolla_master.csv"
echo "Final global analysis: results_shimask_all/island_analysis_stats.csv"
echo "Final pairwise Pst: results_shimask_all/island_pst_pairwise.csv"
echo "Multivariate axes: results_shimask_all/multivariate_pst_axes.csv"
echo "Syndrome heterogeneity: results_shimask_all/syndrome_module_heterogeneity.csv"
echo "Syndrome integration: results_shimask_all/syndrome_integration_levels.csv"
echo "Selective signal disassembly: results_shimask_all/syndrome_attraction_disassembly.csv"
echo "Signal-mechanical hypotheses: results_shimask_all/syndrome_signal_mechanical_hypotheses.csv"
echo "Covariance alignment: results_shimask_all/syndrome_divergence_covariance_alignment.csv"
ls -1 results_shimask_all/*.csv
