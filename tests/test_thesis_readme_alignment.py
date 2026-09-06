from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_matches_final_chapter2_to_chapter3_boundary():
    text = README.read_text(encoding="utf-8")
    lower = text.lower()

    assert "direct phenotypic-realization" in lower
    assert "measurement continuity" in lower
    assert "not a missing validation panel for chapter 2" in lower
    assert "chapter 2 is already complete at the continuity-system boundary" in lower
    assert "strong coordinated size/investment trajectory plus selected departures" in lower
    assert "does not identify historical *bombus* loss" in lower
