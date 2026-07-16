from __future__ import annotations

import unittest

import cv2
import numpy as np

from measure_confirmed_shimask_traits import (
    _confirmed_organs,
    confirmed_corolla_masks,
    simple_corolla_metrics,
)


class ConfirmedTraitTests(unittest.TestCase):
    def test_closed_red_outline_is_filled_as_one_mask(self) -> None:
        red = np.zeros((220, 320), np.uint8)
        cv2.rectangle(red, (60, 50), (260, 170), 1, 5)
        masks = confirmed_corolla_masks(red, min_area_px=100)
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0].sum()), 20000)
        metrics = simple_corolla_metrics(masks[0], 0.1)
        self.assertGreater(float(metrics["corolla_length_mm"]), 18.0)
        self.assertGreater(float(metrics["corolla_width_mm"]), 10.0)

    def test_confirmed_organ_length_uses_green_trace(self) -> None:
        green = np.zeros((200, 300), np.uint8)
        cv2.line(green, (40, 100), (200, 100), 1, 5)
        rows = _confirmed_organs(green, 0.1, "sheet1", "Oshima")
        self.assertEqual(len(rows), 1)
        self.assertGreater(float(rows[0]["organ_length_mm"]), 12.0)
        self.assertEqual(rows[0]["measurement_status"], "direct_from_green_trace")

    def test_nearby_green_traces_are_not_morphologically_merged(self) -> None:
        green = np.zeros((200, 300), np.uint8)
        cv2.line(green, (40, 80), (200, 80), 1, 3)
        cv2.line(green, (40, 87), (200, 87), 1, 3)
        rows = _confirmed_organs(green, 0.1, "sheet1", "Oshima")
        self.assertEqual(len(rows), 2)

    def test_small_annotation_noise_is_ignored(self) -> None:
        green = np.zeros((100, 100), np.uint8)
        green[10:12, 10:12] = 1
        self.assertEqual(_confirmed_organs(green, 0.1, "sheet1", "Oshima"), [])


if __name__ == "__main__":
    unittest.main()
