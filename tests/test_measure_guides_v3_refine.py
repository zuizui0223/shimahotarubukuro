from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine as refine


class RefinedV3Tests(unittest.TestCase):
    def test_long_thin_appendage_is_detached(self) -> None:
        mask = np.zeros((700, 700), np.uint8)
        cv2.ellipse(mask, (300, 380), (150, 210), 0, 0, 360, 1, -1)
        cv2.line(mask, (435, 300), (575, 115), 1, 16)
        cv2.circle(mask, (575, 115), 13, 1, -1)

        cleaned, removed = refine.detach_thin_appendages(mask)

        self.assertEqual(len(removed), 1)
        self.assertLess(int(cleaned.sum()), int(mask.sum()))
        self.assertGreater(int(removed[0].sum()), 500)
        self.assertEqual(int(cleaned[110:140, 555:595].sum()), 0)
        self.assertGreater(int(cleaned[300:500, 200:400].sum()), 10000)

    def test_broad_corolla_without_appendage_is_retained(self) -> None:
        mask = np.zeros((700, 700), np.uint8)
        cv2.ellipse(mask, (350, 350), (165, 225), 0, 0, 360, 1, -1)

        cleaned, removed = refine.detach_thin_appendages(mask)

        self.assertEqual(removed, [])
        self.assertGreater(int(cleaned.sum()), int(mask.sum() * 0.99))

    def test_low_chroma_crease_is_not_called_pistil(self) -> None:
        features = {
            "median_chroma": 4.4,
            "mean_b": 4.0,
            "rect_length_mm": 17.0,
            "aspect": 10.0,
            "median_width_mm": 1.2,
        }
        label, confidence, reason = refine.classify_style_candidate(features)
        self.assertEqual(label, "fragment_or_paper")
        self.assertGreaterEqual(confidence, 0.85)
        self.assertEqual(reason, "low_plant_chroma")

    def test_coloured_elongated_tissue_is_style_candidate(self) -> None:
        features = {
            "median_chroma": 11.5,
            "mean_b": 9.0,
            "rect_length_mm": 17.0,
            "aspect": 6.5,
            "median_width_mm": 1.8,
        }
        label, confidence, reason = refine.classify_style_candidate(features)
        self.assertEqual(label, "style_or_pistil_candidate")
        self.assertGreaterEqual(confidence, 0.70)
        self.assertEqual(reason, "elongated_coloured_plant_tissue")


if __name__ == "__main__":
    unittest.main()
