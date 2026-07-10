from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_review as review
import measure_guides_review_fast as fast


class ReviewedPipelineTests(unittest.TestCase):
    def test_large_open_single_corolla_is_not_split_by_area_alone(self) -> None:
        mask = np.zeros((900, 900), dtype=np.uint8)
        cv2.ellipse(mask, (450, 450), (285, 225), 0, 0, 360, 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "not_triggered")
        self.assertEqual(len(pieces), 1)

    def test_broad_single_body_rejects_internal_kmeans_split(self) -> None:
        mask = np.zeros((1200, 1200), dtype=np.uint8)
        cv2.ellipse(mask, (600, 600), (380, 300), 0, 0, 360, 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "not_triggered")
        self.assertEqual(len(pieces), 1)

    def test_two_touching_corollas_remain_two_bodies(self) -> None:
        # Based on Shikine circled individual ③: two distinct flower bodies joined
        # by a narrow tissue bridge, rather than two arbitrary halves of one body.
        mask = np.zeros((1200, 1500), dtype=np.uint8)
        cv2.ellipse(mask, (420, 650), (165, 275), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (850, 650), (185, 275), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (580, 620), (670, 680), 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "auto_split")
        self.assertEqual(len(pieces), 2)

    def test_short_paper_noise_is_not_an_organ(self) -> None:
        image = np.full((800, 800, 3), 255, dtype=np.uint8)
        cv2.line(image, (250, 400), (300, 400), (150, 170, 190), 5)
        corolla = np.zeros((800, 800), dtype=np.uint8)

        organs = fast.organs_fast(image, corolla, 100)

        self.assertEqual(organs, [])


if __name__ == "__main__":
    unittest.main()
