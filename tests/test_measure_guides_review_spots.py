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

    def _guide_scene(self, strong_cov_frac: float):
        """A corolla with a dense purple guide plus one dark, brown-spectrum dot."""
        shape = (240, 240)
        corolla = np.zeros(shape, dtype=np.uint8)
        cv2.circle(corolla, (120, 120), 90, 1, -1)
        a = np.zeros(shape, dtype=np.float32)
        b = np.zeros(shape, dtype=np.float32)
        L = np.full(shape, 235.0, dtype=np.float32)  # cream tissue is bright

        # Dense field of strong purple guide (high a*, a*-b* > 0), sized to hit
        # the requested fraction of the corolla area.
        rng = np.random.default_rng(0)
        cor_px = int(corolla.sum())
        target = int(strong_cov_frac * cor_px)
        ys, xs = np.where(corolla.astype(bool))
        placed = 0
        idx = rng.permutation(len(ys))
        for k in idx:
            if placed >= target:
                break
            cv2.circle(a, (int(xs[k]), int(ys[k])), 3, 25.0, -1)
            placed += (2 * 3 + 1) ** 2

        # A small oxidised (brown-spectrum) dot inside the guide, in a cleared
        # patch so it reads as an isolated local minimum: dark + reddish but
        # a*-b* < 0, so the purple detector cannot see it.
        cv2.circle(a, (120, 120), 9, 0.0, -1)    # clear purple around the dot
        cv2.circle(a, (120, 120), 2, 12.0, -1)
        cv2.circle(b, (120, 120), 2, 20.0, -1)   # a*-b* = -8 (brown)
        cv2.circle(L, (120, 120), 2, 150.0, -1)  # locally dark
        return a, b, L, corolla

    def test_oxidized_dot_recovered_inside_confirmed_guide(self) -> None:
        a, b, L, corolla = self._guide_scene(strong_cov_frac=0.08)
        strong, weak, combined = spots.spot_candidate_masks(a, b, corolla)
        # The brown-spectrum dot is not part of the purple guide.
        self.assertEqual(int(combined[120, 120]), 0)
        oxidized = spots.oxidized_guide_mask(a, L, corolla, strong, combined)
        self.assertGreater(int(oxidized[120, 120]), 0)

    def test_oxidized_recovery_suppressed_without_a_real_guide(self) -> None:
        # Same brown-spectrum dot, but almost no purple guide: the seed guard
        # must refuse to promote the dark dot (degradation stays degradation).
        a, b, L, corolla = self._guide_scene(strong_cov_frac=0.0)
        strong, weak, combined = spots.spot_candidate_masks(a, b, corolla)
        oxidized = spots.oxidized_guide_mask(a, L, corolla, strong, combined)
        self.assertEqual(int(oxidized.sum()), 0)


if __name__ == "__main__":
    unittest.main()
