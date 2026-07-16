from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine23 as refine23


class RidgeRecoveryTests(unittest.TestCase):
    def test_sato_recovers_dark_yellow_filament(self) -> None:
        image = np.full((420, 620, 3), 246, np.uint8)
        union = np.zeros((420, 620), np.uint8)
        cv2.circle(union, (180, 210), 70, 1, -1)
        cv2.line(image, (285, 205), (430, 215), (150, 195, 220), 7)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        light, a, b = cv2.split(lab)
        blur = cv2.GaussianBlur(light, (0, 0), 15)
        local_dark = blur - light
        zeros = np.zeros_like(light)
        channels = (light, a - 128, b - 128, zeros, zeros, local_dark, zeros)
        mask, response = refine23._candidate_components(union, 0, channels)
        self.assertGreater(float(response.max()), 0.0)
        self.assertGreater(int(mask.sum()), 0)

    def test_duplicate_candidate_is_not_added(self) -> None:
        original = refine23._ORIGINAL_DETECT
        ridge = refine23._ridge_rows
        try:
            refine23._ORIGINAL_DETECT = lambda *args: [{
                'cx': 100.0, 'cy': 100.0, 'organ_instance_id': 1,
                'nearest_corolla': 1,
            }]
            refine23._ridge_rows = lambda *args: [{
                'cx': 101.0, 'cy': 101.0, 'nearest_corolla': 1,
                'organ_len_mm': 10.0, 'organ_width_mm': 1.0,
                'ridge_score': 0.5,
            }]
            rows = refine23.detect_organs(np.zeros((10, 10), np.uint8), [], 0, ())
            self.assertEqual(len(rows), 1)
        finally:
            refine23._ORIGINAL_DETECT = original
            refine23._ridge_rows = ridge


if __name__ == '__main__':
    unittest.main()
