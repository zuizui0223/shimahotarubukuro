"""Regression checks linking the reusable package API to the published case study.

These tests deliberately exercise all 218 reviewed Campanula corollas. They ensure
that generalisation of the measurement code does not silently change the numerical
traits already used by the publication pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "shimaflora" / "core"))
sys.path.insert(0, str(ROOT / "src" / "shimaflora" / "measurement"))

from shimaflora import measure_pattern, measure_shape  # noqa: E402
from shimaflora.presets.campanula import campanula_tubular_traits  # noqa: E402
import guide_colour_model as gcm  # noqa: E402
import measure_guides as base  # noqa: E402
import remeasure_medial as rm  # noqa: E402


def _rows(path: str) -> dict[tuple[str, str], dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig") as handle:
        return {(r["sheet"], r["corolla_id"]): r for r in csv.DictReader(handle)}


def test_all_218_corollas_match_published_mounted_axis_geometry():
    legacy = _rows("results_shimask_all/medial_traits.csv")
    seen = 0
    for sheet in gcm.all_sheets():
        for corolla_id, raw, piece in gcm.iter_corollas(sheet):
            key = (sheet, corolla_id)
            row = legacy[key]
            local = (
                rm.crop_to_petal(raw, piece)
                if key in rm.TRIM_TO_PETAL
                else rm.crop_to_mask(piece)
            )
            solid, _repair = rm._solid_roi((local > 0).astype(np.uint8))
            width_factor = 1.0 if row["fold_state"] == "opened_full" else 2.0
            current = campanula_tubular_traits(
                solid,
                scale_mm_per_px=float(base.MM_PX),
                width_factor=width_factor,
            )
            generic = measure_shape(solid, scale_mm_per_px=float(base.MM_PX))

            # Published dimensions are stored to 0.01 mm. For folded corollas the
            # legacy table rounded observed width first and then multiplied by two,
            # so an otherwise identical raw measurement can differ by exactly 0.01 mm.
            assert np.isclose(
                current["corolla_length_mm"], float(row["corolla_length_mm"]), atol=0.011
            ), key
            assert np.isclose(
                current["corolla_width_mm"], float(row["corolla_width_fulleq_mm"]), atol=0.011
            ), key
            assert np.isclose(
                generic["area_mm2"], float(row["corolla_area_obs_mm2"]), atol=0.051
            ), key
            seen += 1

    assert seen == len(legacy) == 218


def test_all_218_guide_coverages_use_generic_pattern_metric_without_drift():
    legacy = _rows("results_shimask_all/guide_traits.csv")
    params = gcm.load_model()
    seen = 0
    for sheet in gcm.all_sheets():
        for corolla_id, raw, piece in gcm.iter_corollas(sheet):
            key = (sheet, corolla_id)
            guide, _origin, roi_pixels = gcm.segment_piece(raw, piece, params)
            _sub, roi, _origin2 = gcm.crop_piece(raw, piece)
            current = measure_pattern(guide, roi)

            assert int(roi.sum()) == roi_pixels, key
            assert round(current["coverage_pct"], 2) == float(legacy[key]["guide_coverage_pct"]), key
            seen += 1

    assert seen == len(legacy) == 218
