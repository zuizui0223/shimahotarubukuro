import numpy as np

from shimaflora import measure_shape, measure_pattern


def test_measure_shape_rectangle():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 8:28] = 1
    out = measure_shape(mask, scale_mm_per_px=0.5)
    assert out["area_mm2"] == 50.0
    assert out["major_axis_mm"] >= out["minor_axis_mm"]
    assert out["aspect_ratio"] >= 1.0


def test_measure_pattern_coverage_and_components():
    roi = np.zeros((10, 10), dtype=np.uint8)
    roi[1:9, 1:9] = 1
    pattern = np.zeros_like(roi)
    pattern[2:4, 2:4] = 1
    pattern[6:8, 6:8] = 1
    out = measure_pattern(pattern, roi)
    assert out["coverage_pct"] == 12.5
    assert out["component_count"] == 2.0
    assert 0.0 <= out["centroid_x_rel"] <= 1.0
    assert 0.0 <= out["centroid_y_rel"] <= 1.0
