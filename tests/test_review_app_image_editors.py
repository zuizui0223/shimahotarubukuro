from __future__ import annotations

import cv2
import numpy as np

import trait_review
from mask_editor_component import (
    buffered_line_polygon,
    display_line_to_raw,
    display_polygons_to_raw,
    mask_to_display_polygons,
    raw_line_to_display,
    stroke_to_raw_polygons,
)


def synthetic_corolla(shape=(640, 520)):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, (165, 230), (355, 585), 1, -1)
    for x in (180, 220, 260, 300, 340):
        cv2.ellipse(mask, (x, 220), (42, 115), 0, 180, 360, 1, -1)
    return mask


def test_mask_polygon_round_trip_preserves_shape():
    mask = synthetic_corolla()
    box = (100, 70, 430, 620)
    polygons = mask_to_display_polygons(mask, box, 720, 600)
    raw = display_polygons_to_raw(polygons, box, 720, 600, mask.shape)
    rebuilt = np.zeros_like(mask)
    for polygon in raw:
        cv2.fillPoly(rebuilt, [np.asarray(polygon, np.int32)], 1)
    intersection = int(((mask > 0) & (rebuilt > 0)).sum())
    union = int(((mask > 0) | (rebuilt > 0)).sum())
    assert polygons
    assert intersection / union > 0.97


def test_line_coordinate_round_trip():
    box = (100, 200, 500, 700)
    line = [[160.0, 250.0], [440.0, 650.0]]
    display = raw_line_to_display(line, box, 720, 600)
    raw = display_line_to_raw(display, box, 720, 600, (900, 800, 3))
    assert np.allclose(raw, line, atol=1.0)


def test_measurement_guides_drive_derived_traits():
    mask = synthetic_corolla()
    state = {
        "axis_base": [260.0, 585.0],
        "axis_tip": [260.0, 105.0],
        "fold_state": "open",
        "measurement_lines": {},
        "measurement_lines_changed": [],
        "flat_n_lobes": 5,
    }
    lines = trait_review.ensure_measurement_lines(
        state, mask, {"flat_lobe_length_mm": "10", "corolla_length_ruler_mm": "40"}
    )
    values = trait_review.shape_trait_values(mask, state, 0.085)
    assert set(lines) == set(trait_review.MEASUREMENT_GUIDES)
    assert values["corolla_length_ruler_mm"] > 35
    assert values["corolla_max_span_ruler_mm"] > 10
    assert values["flat_tube_length_mm"] > values["flat_lobe_length_mm"]
    assert values["prov_mouth_diameter_ruler_mm"] > 0
    assert values["flat_n_lobes"] == 5


def test_stroke_and_buffered_organ_create_polygons():
    box = (100, 100, 500, 500)
    stroke = [[100, 100], [180, 180], [260, 210]]
    paint = stroke_to_raw_polygons(stroke, 18, box, 720, 600)
    organ = buffered_line_polygon(
        [[200.0, 150.0], [240.0, 430.0]], 16.0, (640, 520, 3)
    )
    assert paint
    assert organ


def test_best_organ_candidate_uses_mask_overlap():
    mask = synthetic_corolla()
    cv2.rectangle(mask, (405, 120), (420, 520), 1, -1)
    rows = [
        {
            "organ_id": "near",
            "cx": "412",
            "cy": "320",
            "length_mm": "34",
            "width_mm": "1.4",
            "angle_deg": "90",
        },
        {
            "organ_id": "far",
            "cx": "40",
            "cy": "40",
            "length_mm": "30",
            "width_mm": "2.0",
            "angle_deg": "0",
        },
    ]
    candidate = trait_review.best_organ_candidate(rows, mask, 0.085)
    assert candidate["candidate_id"] == "near"
    assert np.isclose(trait_review.line_length(candidate["line"]) * 0.085, 34)
    assert candidate["width_mm"] == 1.4


def test_organ_candidate_line_prefers_reviewed_endpoints():
    row = {
        "x1": "10.5",
        "y1": "20.5",
        "x2": "40.5",
        "y2": "80.5",
        "cx": "999",
        "cy": "999",
        "length_mm": "1",
        "angle_deg": "0",
    }
    assert trait_review.organ_candidate_line(row, 0.1) == [
        [10.5, 20.5],
        [40.5, 80.5],
    ]


def test_organ_candidate_line_reconstructs_legacy_detector_row():
    row = {
        "cx": "100",
        "cy": "200",
        "length_mm": "20",
        "angle_deg": "90",
    }
    line = trait_review.organ_candidate_line(row, 0.1)
    assert np.allclose(trait_review.line_midpoint(line), [100, 200])
    assert np.isclose(trait_review.line_length(line) * 0.1, 20)


def test_nearest_detached_organ_skips_assigned_candidate():
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 25, 1, -1)
    rows = [
        {
            "organ_id": "1",
            "x1": "110",
            "y1": "100",
            "x2": "150",
            "y2": "100",
            "width_mm": "1.1",
        },
        {
            "organ_id": "2",
            "x1": "180",
            "y1": "100",
            "x2": "220",
            "y2": "100",
            "width_mm": "1.4",
        },
    ]
    candidate = trait_review.nearest_detached_organ_candidate(
        rows, mask, 0.1, used_ids=["1"]
    )
    assert candidate["candidate_id"] == "2"
    assert candidate["width_mm"] == 1.4


def test_core_pollination_traits_stay_focused():
    fields = {field for field, _, _, _ in trait_review.CORE_POLLINATION_TRAITS}
    assert fields == {
        "corolla_length_ruler_mm",
        "corolla_area_ruler_mm2",
        "corolla_max_span_ruler_mm",
        "flat_throat_span_mm",
        "flat_throat_openness",
        "flat_tube_taper_ratio",
        "guide_area_mm2",
        "guide_cov_pct",
        "guide_present",
    }


def test_manual_guide_presence_produces_consistent_analysis_values():
    automatic = {
        "guide_area_mm2": 12.3,
        "guide_cov_pct": 4.5,
        "guide_present": 1,
        "n_spots": 7,
        "guide_area_incl_oxidized_mm2": 15.0,
        "guide_cov_incl_oxidized_pct": 5.5,
        "brown_frac": 0.1,
    }
    absent = trait_review.reviewed_guide_trait_values(automatic, "absent")
    uncertain = trait_review.reviewed_guide_trait_values(automatic, "uncertain")
    present = trait_review.reviewed_guide_trait_values(
        {**automatic, "guide_present": 0}, "present"
    )

    assert all(absent[field] == 0 for field in trait_review.GUIDE_ANALYSIS_FIELDS)
    assert all(
        uncertain[field] == "" for field in trait_review.GUIDE_ANALYSIS_FIELDS
    )
    assert present["guide_present"] == 1
    assert present["guide_area_mm2"] == automatic["guide_area_mm2"]
    assert absent["brown_frac"] == automatic["brown_frac"]


def test_thin_appendage_seeds_organ_when_detector_misses_it():
    mask = np.zeros((640, 520), dtype=np.uint8)
    cv2.rectangle(mask, (120, 260), (360, 590), 1, -1)
    cv2.rectangle(mask, (330, 75), (350, 300), 1, -1)
    candidate = trait_review.best_organ_candidate([], mask, 0.085)
    assert candidate["candidate_id"] == "mask-thin-appendage"
    assert trait_review.line_length(candidate["line"]) * 0.085 > 12
    assert 0.5 <= candidate["width_mm"] <= 5.0
    assert candidate["polygons"]

    shifted_line = np.asarray(candidate["line"]) + np.array([20.0, 10.0])
    shifted = trait_review.transform_seed_polygons(
        candidate["polygons"], candidate["line"], shifted_line, image_shape=mask.shape
    )
    assert shifted
    assert np.asarray(shifted).reshape(-1, 2)[:, 0].mean() > (
        np.asarray(candidate["polygons"]).reshape(-1, 2)[:, 0].mean() + 15
    )


def test_colour_traits_return_reviewable_measurements():
    mask = synthetic_corolla()
    raw = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    raw[mask > 0] = (205, 220, 230)
    for centre in ((230, 310), (285, 370), (260, 470)):
        cv2.circle(raw, centre, 9, (150, 45, 180), -1)
    state = {
        "axis_base": [260.0, 585.0],
        "axis_tip": [260.0, 105.0],
        "fold_state": "open",
        "measurement_lines": {},
        "measurement_lines_changed": [],
        "region_edits": {
            key: {"add": [], "subtract": []}
            for key in trait_review.REGION_TARGETS
        },
    }
    trait_review.ensure_measurement_lines(state, mask, {})
    values, regions = trait_review.colour_trait_values(
        raw, mask, (100, 70, 430, 620), state, 0.085
    )
    assert set(trait_review.REGION_TARGETS).issubset(regions)
    assert values["guide_area_mm2"] >= 0
    assert values["n_spots"] >= 0
    assert 0 <= values["guide_centroid_rel"] <= 1
