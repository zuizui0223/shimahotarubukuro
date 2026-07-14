from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine as refine
import measure_guides_v3_refine2 as refine2
import measure_guides_v3_refine7 as refine7


class PaperTailRefinementTests(unittest.TestCase):
    def test_low_chroma_bottom_tail_is_detected(self) -> None:
        image = np.full((420, 320, 3), 250, np.uint8)
        mask = np.zeros((420, 320), np.uint8)
        cv2.ellipse(mask, (160, 190), (95, 145), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (80, 315), (240, 390), 1, -1)
        image[mask > 0] = (160, 175, 220)
        # Neutral paper-like tail.
        image[315:391, 80:241] = (205, 205, 205)
        refine2._CURRENT_CHANNELS = refine._lab_channels(image)
        self.addCleanup(setattr, refine2, "_CURRENT_CHANNELS", None)

        tail = refine7._bottom_tail(mask)

        self.assertIsNotNone(tail)
        self.assertGreater(int(tail[340:].sum()), 0)
        self.assertEqual(int(tail[:280].sum()), 0)

    def test_short_unknown_cleanup_piece_is_noise(self) -> None:
        label, confidence, reason = refine7.classify_style_candidate(
            {"rect_length_mm": 5.7, "median_chroma": 9.5, "aspect": 1.8,
             "median_width_mm": 3.0}
        )
        self.assertEqual(label, "fragment_or_paper")
        self.assertGreaterEqual(confidence, 0.9)
        self.assertIn("short_ambiguous", reason)

    def test_long_style_candidate_is_preserved(self) -> None:
        label, _, _ = refine7.classify_style_candidate(
            {"rect_length_mm": 18.0, "median_chroma": 10.0, "aspect": 5.0,
             "median_width_mm": 1.2}
        )
        self.assertEqual(label, "style_or_pistil_candidate")


if __name__ == "__main__":
    unittest.main()
