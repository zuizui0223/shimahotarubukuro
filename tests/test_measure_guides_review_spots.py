from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_review_spots as spots


class ReviewedSpotTests(unittest.TestCase):
    def test_tiny_scan_noise_is_removed(self) -> None:
        mask = np.zeros((300, 300), dtype=np.uint8)
        # One accepted guide spot and one single-pixel scan artifact.
        cv2.circle(mask, (100, 100), 8, 1, -1)
        mask[200, 200] = 1

        accepted, labels, rows = spots._accepted_spots(mask)

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(accepted[200, 200]), 0)
        self.assertGreater(int(accepted[100, 100]), 0)
        self.assertGreater(int(labels[100, 100]), 0)

    def test_all_accepted_components_meet_area_threshold(self) -> None:
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (80, 80), 5, 1, -1)
        cv2.circle(mask, (180, 180), 10, 1, -1)
        cv2.circle(mask, (300, 300), 15, 1, -1)

        accepted, labels, rows = spots._accepted_spots(mask)

        self.assertEqual(len(rows), 3)
        self.assertTrue(
            all(row["area_mm2"] >= spots.MIN_SPOT_AREA_MM2 for row in rows)
        )
        self.assertGreater(int(accepted.sum()), 0)
        self.assertEqual(accepted.shape, labels.shape)

    def test_threshold_is_consistent_with_300_dpi_scale(self) -> None:
        required_pixels = spots.MIN_SPOT_AREA_MM2 / base.MM2_PX
        self.assertGreater(required_pixels, 1.0)
        self.assertLess(required_pixels, 10.0)


if __name__ == "__main__":
    unittest.main()
