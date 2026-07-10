from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_review as review


class ReviewedPipelineTests(unittest.TestCase):
    def test_large_open_single_corolla_is_not_split_by_area_alone(self) -> None:
        mask = np.zeros((900, 900), dtype=np.uint8)
        cv2.ellipse(mask, (450, 450), (285, 225), 0, 0, 360, 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "not_triggered")
        self.assertEqual(len(pieces), 1)

    def test_long_thin_attached_spur_is_pruned(self) -> None:
        mask = np.zeros((700, 700), dtype=np.uint8)
        cv2.ellipse(mask, (350, 420), (170, 190), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (338, 90), (362, 250), 1, -1)
        cv2.rectangle(mask, (330, 240), (370, 300), 1, -1)
        before = int(mask.sum())

        cleaned, changed = review.prune_thin_spurs(mask)

        self.assertTrue(changed)
        self.assertLess(int(cleaned.sum()), before)
        self.assertEqual(int(cleaned[110, 350]), 0)
        self.assertEqual(int(cleaned[420, 350]), 1)

    def test_short_paper_noise_is_not_an_organ(self) -> None:
        image = np.full((800, 800, 3), 255, dtype=np.uint8)
        cv2.line(image, (250, 400), (300, 400), (150, 170, 190), 5)
        corolla = np.zeros((800, 800), dtype=np.uint8)

        organs = review.organs_reviewed(image, corolla, 100)

        self.assertEqual(organs, [])


if __name__ == "__main__":
    unittest.main()
