import numpy as np

from shimaflora import (
    measure_pattern,
    measure_shape,
    overlay_morphology,
    overlay_pattern,
    overlay_roi,
    synthetic_flower,
)


def test_synthetic_example_is_deterministic_and_measurable():
    image1, roi1, pattern1 = synthetic_flower(160)
    image2, roi2, pattern2 = synthetic_flower(160)
    assert np.array_equal(image1, image2)
    assert np.array_equal(roi1, roi2)
    assert np.array_equal(pattern1, pattern2)
    assert image1.shape == (160, 160, 3)
    assert image1.dtype == np.uint8
    assert roi1.dtype == np.uint8
    assert pattern1.dtype == np.uint8
    assert np.all(pattern1 <= roi1)

    shape = measure_shape(roi1, scale_mm_per_px=0.1)
    pattern = measure_pattern(pattern1, roi1)
    assert shape["area_mm2"] > 0
    assert shape["major_axis_mm"] > shape["minor_axis_mm"]
    assert 0 < pattern["coverage_pct"] < 20
    assert pattern["component_count"] >= 4


def test_qc_overlays_return_rgb_uint8_without_mutating_input():
    image, roi, pattern = synthetic_flower(128)
    original = image.copy()
    for out in (
        overlay_roi(image, roi),
        overlay_morphology(image, roi),
        overlay_pattern(image, roi, pattern),
    ):
        assert out.shape == image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, image)
    assert np.array_equal(image, original)


def test_bgr_input_is_converted_to_rgb():
    image, roi, _ = synthetic_flower(128)
    bgr = image[:, :, ::-1]
    rgb_out = overlay_roi(image, roi, input_order="rgb")
    bgr_out = overlay_roi(bgr, roi, input_order="bgr")
    assert np.array_equal(rgb_out, bgr_out)


def test_pattern_outside_roi_is_ignored():
    image, roi, pattern = synthetic_flower(128)
    outside = pattern.copy()
    outside[0:10, 0:10] = 1
    clean = overlay_pattern(image, roi, pattern, alpha=0.5)
    spill = overlay_pattern(image, roi, outside, alpha=0.5)
    assert np.array_equal(clean, spill)


def test_grayscale_input_supported():
    _, roi, pattern = synthetic_flower(128)
    gray = np.full(roi.shape, 180, dtype=np.uint8)
    out = overlay_pattern(gray, roi, pattern)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8
