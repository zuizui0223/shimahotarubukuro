from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine10 as refine10


class FaintOrganRecoveryTests(unittest.TestCase):
    def test_faint_linear_object_is_recovered(self) -> None:
        image = np.full((500, 500, 3), 245, np.uint8)
        cv2.line(image, (300, 180), (315, 330), (225, 225, 225), 5)
        excluded = np.zeros((500, 500), np.uint8)
        mask = refine10._line_candidates(image, excluded)
        self.assertGreater(int(mask[:, 280:340].sum()), 100)

    def test_uniform_paper_is_not_recovered(self) -> None:
        image = np.full((400, 400, 3), 245, np.uint8)
        excluded = np.zeros((400, 400), np.uint8)
        mask = refine10._line_candidates(image, excluded)
        self.assertEqual(int(mask.sum()), 0)

    def test_corolla_region_is_excluded(self) -> None:
        image = np.full((400, 400, 3), 245, np.uint8)
        cv2.line(image, (200, 100), (200, 300), (220, 220, 220), 5)
        excluded = np.zeros((400, 400), np.uint8)
        excluded[:, 180:220] = 1
        mask = refine10._line_candidates(image, excluded)
        self.assertEqual(int(mask[:, 180:220].sum()), 0)


if __name__ == "__main__":
    unittest.main()
