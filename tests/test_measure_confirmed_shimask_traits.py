from __future__ import annotations

import unittest

import cv2
import numpy as np

from measure_confirmed_shimask_traits import _confirmed_organs


class ConfirmedTraitTests(unittest.TestCase):
    def test_confirmed_organ_length_uses_green_trace(self) -> None:
        green = np.zeros((200, 300), np.uint8)
        cv2.line(green, (40, 100), (200, 100), 1, 5)
        rows = _confirmed_organs(green, 0.1, "sheet1", "Oshima")
        self.assertEqual(len(rows), 1)
        self.assertGreater(float(rows[0]["organ_length_mm"]), 12.0)
        self.assertEqual(rows[0]["provenance"], "shimask_human_review")

    def test_small_annotation_noise_is_ignored(self) -> None:
        green = np.zeros((100, 100), np.uint8)
        green[10:12, 10:12] = 1
        self.assertEqual(_confirmed_organs(green, 0.1, "sheet1", "Oshima"), [])


if __name__ == "__main__":
    unittest.main()
