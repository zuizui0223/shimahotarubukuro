from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine21 as refine21


class GuardedLocalGrowthTests(unittest.TestCase):
    def test_recovers_missing_tissue_ring(self) -> None:
        image = np.full((500, 500, 3), 246, np.uint8)
        cv2.circle(image, (250, 250), 120, (170, 202, 224), -1)
        mask = np.zeros((500, 500), np.uint8)
        cv2.circle(mask, (250, 250), 110, 1, -1)
        grown, evidence, status = refine21.guarded_expand(image, mask)
        self.assertEqual(status, "accepted")
        self.assertGreater(int(grown.sum()), int(mask.sum()))
        self.assertGreater(evidence, refine21._MIN_EVIDENCE)

    def test_does_not_expand_into_uniform_paper(self) -> None:
        image = np.full((400, 400, 3), 246, np.uint8)
        mask = np.zeros((400, 400), np.uint8)
        cv2.circle(mask, (200, 200), 90, 1, -1)
        grown, _, status = refine21.guarded_expand(image, mask)
        self.assertNotEqual(status, "accepted")
        self.assertTrue(np.array_equal(grown, mask))

    def test_area_guard_blocks_large_leak(self) -> None:
        image = np.full((500, 500, 3), 246, np.uint8)
        cv2.circle(image, (250, 250), 170, (170, 202, 224), -1)
        mask = np.zeros((500, 500), np.uint8)
        cv2.circle(mask, (250, 250), 90, 1, -1)
        grown, _, status = refine21.guarded_expand(image, mask)
        self.assertNotEqual(status, "accepted")
        self.assertTrue(np.array_equal(grown, mask))


if __name__ == "__main__":
    unittest.main()
