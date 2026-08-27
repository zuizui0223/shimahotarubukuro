import json
from pathlib import Path

from shimaflora.analysis.chapter3_axis_contrast import summarize


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "results_shimask_all" / "island_analysis_stats.csv"
FROZEN = ROOT / "results_shimask_all" / "chapter3_axis_contrast_frozen_20260827.json"


def test_prespecified_axis_contrast_matches_frozen_result():
    observed = summarize(STATS)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    for group in ("absolute_investment", "proportional_spatial"):
        assert observed[group] == frozen[group]
    assert observed["contrast"] == frozen["contrast"]
    assert observed["excluded_trait"] == frozen["excluded_trait"]


def test_absolute_investment_axes_are_completely_separated_in_pst():
    result = summarize(STATS)
    absolute = result["absolute_investment"]
    proportional = result["proportional_spatial"]

    assert absolute["min_pst"] > proportional["max_pst"]
    assert result["contrast"]["cliffs_delta"] == 1.0
    assert absolute["site_corrected_significant_count"] == 7
    assert proportional["site_corrected_significant_count"] == 0
