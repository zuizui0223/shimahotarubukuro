#!/usr/bin/env bash
# Reproduce the Campanula microdonta floral-trait results from the committed inputs.
#
# Inputs (all in the repo):
#   shimahotarubukuro/<island>/<sheet>.jpg  raw specimen scans
#   shimask/<sheet>.jpg                     reviewer's hand annotations (corolla
#                                           outlines in red, organ strokes in green)
#   mask.zip                                iPhone-lifted per-corolla silhouettes
#   withlocation.csv                        field table (locality, individual, sexual
#                                           phase) keyed by the 1-218 corolla number
#
# Output: results_shimask_all/ with the measured CSVs (corolla_master.csv is the
# final integrated table) and the figures / per-flower measurement cards.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> unzip iPhone masks"
rm -rf mask && unzip -q -o mask.zip -d . && test -d mask

mkdir -p results_shimask_all

echo ">> 1/12 corolla size from hand ROI      -> medial_traits.csv"
python3 remeasure_medial.py
echo ">> 2/12 iPhone-registered size ROI      -> iphone_traits.csv"
python3 register_iphone_masks.py
echo ">> 3/12 nectar-guide traits             -> guide_traits.csv"
python3 guide_traits.py
echo ">> 4/12 reproductive-organ length       -> organ_traits.csv"
python3 organ_traits.py
echo ">> 5/12 combined final table            -> corolla_traits_final.csv"
python3 merge_traits.py
echo ">> 6/12 pollination morphometrics       -> pollination_traits.csv"
python3 pollination_traits.py
echo ">> 7/12 continuous 1-218 numbering       -> global_index.csv (+ numbered_index/)"
python3 make_numbered_index.py
echo ">> 8/12 integrate field metadata        -> corolla_master.csv  (FINAL)"
python3 integrate_metadata.py
echo ">> 9/12 guide divergence stats          -> guide_divergence_stats.csv"
python3 guide_divergence.py
echo ">> 10/12 guide spatial structure        -> guide_spatial.csv"
python3 guide_spatial.py

echo ">> 11/12 figures"
for p in plot_island_traits plot_guide_traits plot_organ_traits plot_pollination_traits \
         plot_guide_hypothesis plot_guide_spatial plot_guide_density_islands; do
  python3 "$p.py" || true
done

echo ">> 12/12 per-flower measurement cards + trait overlays"
python3 make_overlays.py || true
python3 make_measure_cards.py

echo ">> DONE. Final table: results_shimask_all/corolla_master.csv"
ls -1 results_shimask_all/*.csv
