from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_review_spots as spots


class ReviewedSpotTests(unittest.TestCase):
    def test_tiny_scan_noise_is_removed(self) -> None:
        combined = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(combined, (100, 100), 8, 1, -1)
        combined[200, 200] = 1
        strong = combined.copy()
        weak = np.zeros_like(combined)

        accepted, labels, rows = spots._accepted_spots(
            combined, strong, weak, base.MM_PX
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(accepted[200, 200]), 0)
        self.assertGreater(int(accepted[100, 100]), 0)
        self.assertGreater(int(labels[100, 100]), 0)

    def test_all_accepted_components_meet_area_threshold(self) -> None:
        combined = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(combined, (80, 80), 5, 1, -1)
        cv2.circle(combined, (180, 180), 10, 1, -1)
        cv2.circle(combined, (300, 300), 15, 1, -1)
        strong = combined.copy()
        weak = np.zeros_like(combined)

        accepted, labels, rows = spots._accepted_spots(
            combined, strong, weak, base.MM_PX
        )

        self.assertEqual(len(rows), 3)
        self.assertTrue(
            all(float(row["area_mm2"]) >= spots.MIN_SPOT_AREA_MM2 for row in rows)
        )
        self.assertGreater(int(accepted.sum()), 0)
        self.assertEqual(accepted.shape, labels.shape)

    def test_faint_magenta_spot_is_recovered_separately(self) -> None:
        shape = (220, 220)
        corolla = np.zeros(shape, dtype=np.uint8)
        cv2.circle(corolla, (110, 110), 80, 1, -1)
        a = np.zeros(shape, dtype=np.float32)
        b = np.zeros(shape, dtype=np.float32)

        cv2.circle(a, (80, 110), 5, 8.0, -1)    # strong magenta
        cv2.circle(a, (145, 110), 5, 1.35, -1)  # faint magenta
        cv2.circle(b, (145, 110), 5, 0.85, -1)  # a*-b* = 0.50

        strong, weak, combined = spots.spot_candidate_masks(a, b, corolla)

        self.assertGreater(int(strong[110, 80]), 0)
        self.assertEqual(int(strong[110, 145]), 0)
        self.assertGreater(int(weak[110, 145]), 0)
        self.assertGreater(int(combined[110, 145]), 0)

    def test_threshold_is_consistent_with_300_dpi_scale(self) -> None:
        required_pixels = spots.MIN_SPOT_AREA_MM2 / base.MM2_PX
        self.assertGreater(required_pixels, 1.0)
        self.assertLess(required_pixels, 4.0)


if __name__ == "__main__":
    unittest.main()
