from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine as refine
import measure_guides_v3_refine18 as refine18


class CorollaAreaTraitTests(unittest.TestCase):
    def test_lobe_and_tube_areas_partition_the_corolla(self) -> None:
        # A lobed corolla: broad top with three lobe tips, narrowing to the base.
        mask = np.zeros((900, 640), np.uint8)
        pts = np.array([
            [120, 300], [200, 90], [300, 260], [340, 90], [440, 260],
            [520, 90], [560, 300], [420, 820], [220, 820],
        ], np.int32)
        cv2.fillPoly(mask, [pts], 1)
        traits = refine18.corolla_area_traits(mask)
        self.assertIn(traits["area_partition_status"], {"automatic_provisional", "review_required"})
        # When measurable, the lobe/tube split is a genuine partition of the mask.
        if traits["lobe_area_mm2"] != "":
            lobe = float(traits["lobe_area_mm2"]); tube = float(traits["tube_area_mm2"])
            oriented_total = lobe + tube
            self.assertGreater(lobe, 0.0)
            self.assertGreater(tube, 0.0)
            self.assertTrue(0.0 < float(traits["lobe_area_frac"]) < 1.0)
            # partition ~= mask area (nearest-neighbour reorientation is near area-preserving)
            total = float(int(mask.sum()) * float(base.MM2_PX))
            self.assertAlmostEqual(oriented_total, total, delta=0.05 * total)
        self.assertNotEqual(traits["corolla_perimeter_mm"], "")
        self.assertGreater(float(traits["corolla_perimeter_mm"]), 0.0)

    def test_tiny_mask_is_not_measurable(self) -> None:
        mask = np.zeros((50, 50), np.uint8)
        mask[10:13, 10:13] = 1
        traits = refine18.corolla_area_traits(mask)
        self.assertEqual(traits["area_partition_status"], "not_measurable")
        self.assertEqual(traits["lobe_area_mm2"], "")

    def test_recompute_is_chained_onto_refine14(self) -> None:
        # refine18 must extend, not replace, the refine14 trait recompute.
        self.assertEqual(refine._recompute_traits.__name__, "_recompute_with_areas")


if __name__ == "__main__":
    unittest.main()
