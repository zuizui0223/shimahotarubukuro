from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine as refine
import measure_guides_v3_refine13 as refine13
import measure_guides_v3_refine19 as refine19


class GrabcutRefineTests(unittest.TestCase):
    def _scene(self):
        # White paper with a tan corolla blob; the cleaned mask over-extends into a
        # bright halo ring, which GrabCut should trim back toward the tan tissue.
        img = np.full((600, 600, 3), 246, np.uint8)
        cv2.circle(img, (300, 300), 150, (168, 202, 224), -1)  # pale tan tissue
        mask = np.zeros((600, 600), np.uint8)
        cv2.circle(mask, (300, 300), 168, 1, -1)               # mask 18 px too big
        return img, mask

    def test_snap_returns_sane_mask_within_guard(self) -> None:
        img, mask = self._scene()
        out = refine19.grabcut_refine(img, mask)
        self.assertGreater(int(out.sum()), 0)
        ratio = int(out.sum()) / int(mask.sum())
        self.assertTrue(refine19._AREA_LO <= ratio <= refine19._AREA_HI)
        # a single connected component
        n, _, _, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
        self.assertEqual(n, 2)

    def test_degenerate_mask_falls_back(self) -> None:
        img = np.full((100, 100, 3), 246, np.uint8)
        tiny = np.zeros((100, 100), np.uint8)
        tiny[10:14, 10:14] = 1
        out = refine19.grabcut_refine(img, tiny)
        self.assertTrue(np.array_equal(out > 0, tiny > 0))

    def test_pipeline_wiring(self) -> None:
        self.assertEqual(float(refine13.COROLLA_EDGE_ERODE_MM), 0.0)
        self.assertEqual(refine13.process_sheet.__name__, "process_sheet")
        # trait + organ extensions from refine14/refine18 remain installed
        self.assertEqual(refine._recompute_traits.__name__, "_recompute_with_areas")
        self.assertEqual(refine13.detect_organs.__name__, "_instance_aware_organs")


if __name__ == "__main__":
    unittest.main()
