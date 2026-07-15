from __future__ import annotations

import unittest

import cv2
import numpy as np

from shimask_annotation_diff import annotation_masks


class AnnotationDifferenceTests(unittest.TestCase):
    def test_natural_red_spot_is_rejected_added_red_line_is_kept(self) -> None:
        raw = np.full((220, 320, 3), 240, np.uint8)
        cv2.circle(raw, (100, 110), 22, (35, 45, 190), -1)  # natural red flower spot
        annotated = raw.copy()
        cv2.line(annotated, (40, 40), (280, 40), (0, 0, 255), 5)
        red, green = annotation_masks(annotated, raw)
        self.assertGreater(int(red[35:46, 35:286].sum()), 500)
        self.assertEqual(int(red[85:136, 75:126].sum()), 0)
        self.assertEqual(int(green.sum()), 0)

    def test_added_green_organ_line_is_kept(self) -> None:
        raw = np.full((220, 320, 3), 240, np.uint8)
        annotated = raw.copy()
        cv2.line(annotated, (170, 50), (170, 180), (0, 255, 0), 6)
        red, green = annotation_masks(annotated, raw)
        self.assertGreater(int(green[:, 160:181].sum()), 500)
        self.assertEqual(int(red.sum()), 0)


if __name__ == "__main__":
    unittest.main()
