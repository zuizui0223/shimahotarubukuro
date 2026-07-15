from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13


def _scene():
    """White sheet with a corolla, a yellow filament beside it (a laid-out organ),
    and a neutral grey crease at the same level (a paper fold)."""
    img = np.full((760, 1000, 3), 245, np.uint8)
    corolla = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(corolla, (300, 380), 170, 1, -1)
    img[corolla > 0] = (170, 205, 225)          # pale tan tissue
    # detached organ: thin, darker, yellowish filament ~14 mm long, beside the corolla
    cv2.rectangle(img, (588, 300), (612, 465), (120, 150, 180), -1)
    # paper fold: same darkness but neutral grey (no yellowness)
    cv2.rectangle(img, (708, 300), (732, 465), (150, 150, 150), -1)
    return img, corolla


class OrganDetectorTests(unittest.TestCase):
    def test_yellow_filament_detected_grey_fold_rejected(self) -> None:
        img, corolla = _scene()
        channels = refine._lab_channels(img)
        rows = refine13.detect_organs(corolla, [{"mask": corolla.astype(bool)}], 0, channels)
        self.assertTrue(rows, "expected the yellow organ filament to be detected")
        # a detection lands on the yellow filament (x ~ 600)
        self.assertTrue(any(abs(float(r["cx"]) - 600) <= 40 for r in rows))
        # no detection lands on the neutral grey fold (x ~ 720)
        self.assertFalse(any(abs(float(r["cx"]) - 720) <= 40 for r in rows))

    def test_detection_is_associated_to_the_corolla(self) -> None:
        img, corolla = _scene()
        channels = refine._lab_channels(img)
        rows = refine13.detect_organs(corolla, [{"mask": corolla.astype(bool)}], 0, channels)
        self.assertTrue(all(r["nearest_corolla"] == 1 for r in rows))
        self.assertTrue(all(float(r["association_distance_mm"]) <= refine13.ASSOC_MAX_MM for r in rows))

    def test_filament_far_from_any_corolla_is_ignored(self) -> None:
        # Wide scene: a corolla on the left, its organ beside it, and an identical
        # yellow filament far to the right (beyond the search band).
        img = np.full((760, 1900, 3), 245, np.uint8)
        corolla = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(corolla, (300, 380), 170, 1, -1)
        img[corolla > 0] = (170, 205, 225)
        cv2.rectangle(img, (588, 300), (612, 465), (120, 150, 180), -1)   # near organ
        cv2.rectangle(img, (1788, 300), (1812, 465), (120, 150, 180), -1)  # distant filament
        channels = refine._lab_channels(img)
        rows = refine13.detect_organs(corolla, [{"mask": corolla.astype(bool)}], 0, channels)
        self.assertTrue(any(abs(float(r["cx"]) - 600) <= 40 for r in rows))
        self.assertFalse(any(abs(float(r["cx"]) - 1800) <= 40 for r in rows))


class AxisSamplingTests(unittest.TestCase):
    def test_long_component_yields_several_points(self) -> None:
        step_px = refine13.SAMPLE_STEP_MM / refine13.MM_PX
        length = int(round(step_px * 3))
        ys = np.arange(0, length)
        xs = np.zeros_like(ys)
        pts = refine13._axis_points(ys, xs)
        self.assertGreaterEqual(len(pts), 2)

    def test_short_component_yields_one_point(self) -> None:
        ys = np.arange(0, 6)
        xs = np.zeros_like(ys)
        pts = refine13._axis_points(ys, xs)
        self.assertEqual(len(pts), 1)


if __name__ == "__main__":
    unittest.main()
