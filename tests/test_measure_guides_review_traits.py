from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_review_traits as traits


class ReviewedVisitorTraitTests(unittest.TestCase):
    def test_ruler_centimetre_ticks_define_scale(self) -> None:
        image = np.full((900, 1800, 3), 255, dtype=np.uint8)
        baseline = 520
        cv2.line(image, (120, baseline), (1680, baseline), (0, 0, 0), 5)
        # This unit test isolates the long centimetre marks. The real Shikine scan
        # additionally contains half- and millimetre marks and is tested by the
        # workflow's raw-image scale output.
        for x in range(200, 1601, 118):
            cv2.line(image, (x, baseline - 130), (x, baseline), (0, 0, 0), 3)

        scale = traits.calibrate_ruler(image, specimen_top=560)

        self.assertEqual(scale["scale_source"], "ruler_1cm_ticks")
        self.assertEqual(scale["scale_qc"], "ok")
        self.assertAlmostEqual(float(scale["px_per_cm"]), 118.0, delta=1.2)
        self.assertAlmostEqual(float(scale["mm_per_px"]), 10.0 / 118.0, delta=0.001)

    def test_planar_flower_traits_are_positive(self) -> None:
        mask = np.zeros((600, 500), dtype=np.uint8)
        # Basal tube.
        cv2.rectangle(mask, (155, 230), (345, 560), 1, -1)
        # Five upper lobes.
        for x in (170, 210, 250, 290, 330):
            cv2.ellipse(mask, (x, 210), (42, 115), 0, 180, 360, 1, -1)
        spots = np.zeros_like(mask)
        cv2.circle(spots, (230, 390), 8, 1, -1)
        cv2.circle(spots, (280, 430), 8, 1, -1)

        result, oriented, oriented_spots, guides = traits.measure_flat_traits(
            mask.astype(bool), spots.astype(bool), mm_per_px=0.085
        )

        self.assertGreater(result["corolla_length_ruler_mm"], 20)
        self.assertGreater(result["flat_tube_length_mm"], 5)
        self.assertGreater(result["flat_throat_span_mm"], 5)
        self.assertGreater(result["prov_mouth_diameter_ruler_mm"], 1)
        self.assertGreater(result["prov_entrance_area_mm2"], 1)
        self.assertEqual(oriented.shape, oriented_spots.shape)
        self.assertLess(guides["sinus_y"], oriented.shape[0])

    def test_folded_specimen_uses_doubled_circumference_proxy(self) -> None:
        mask = np.zeros((700, 260), dtype=np.uint8)
        cv2.rectangle(mask, (80, 160), (180, 650), 1, -1)
        cv2.ellipse(mask, (130, 150), (100, 120), 0, 180, 360, 1, -1)
        spots = np.zeros_like(mask)

        result, _, _, _ = traits.measure_flat_traits(
            mask.astype(bool), spots.astype(bool), mm_per_px=0.085
        )

        self.assertEqual(result["fold_state_auto"], "folded_half")
        self.assertEqual(result["mouth_proxy_assumption"], "2x_flat_span_over_pi")


if __name__ == "__main__":
    unittest.main()
