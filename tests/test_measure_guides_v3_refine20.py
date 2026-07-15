from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine20 as refine20


class MultiscaleBoundaryTests(unittest.TestCase):
    def test_boundary_evidence_prefers_true_tissue_edge(self) -> None:
        image = np.full((500, 500, 3), 248, np.uint8)
        cv2.circle(image, (250, 250), 120, (165, 198, 220), -1)
        true_mask = np.zeros((500, 500), np.uint8)
        cv2.circle(true_mask, (250, 250), 120, 1, -1)
        oversized = np.zeros_like(true_mask)
        cv2.circle(oversized, (250, 250), 145, 1, -1)
        self.assertGreater(
            refine20.boundary_evidence(image, true_mask, oversized),
            refine20.boundary_evidence(image, oversized, oversized),
        )

    def test_multiscale_result_is_connected_and_guarded(self) -> None:
        image = np.full((500, 500, 3), 248, np.uint8)
        cv2.ellipse(image, (250, 250), (130, 90), 15, 0, 360, (170, 200, 225), -1)
        mask = np.zeros((500, 500), np.uint8)
        cv2.ellipse(mask, (250, 250), (148, 108), 15, 0, 360, 1, -1)
        out, score, label = refine20.multiscale_refine(image, mask)
        self.assertTrue(np.isfinite(score))
        self.assertIsInstance(label, str)
        ratio = int(out.sum()) / int(mask.sum())
        self.assertTrue(refine20._AREA_LO <= ratio <= refine20._AREA_HI)
        n, _, _, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
        self.assertEqual(n, 2)

    def test_tiny_mask_falls_back(self) -> None:
        image = np.full((80, 80, 3), 248, np.uint8)
        mask = np.zeros((80, 80), np.uint8)
        mask[10:14, 10:14] = 1
        out, _, _ = refine20.multiscale_refine(image, mask)
        self.assertTrue(np.array_equal(out, mask))


if __name__ == "__main__":
    unittest.main()
