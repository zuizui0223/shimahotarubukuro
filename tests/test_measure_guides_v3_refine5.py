from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
from measure_guides_v3_refine5 import merge_removed_masks


class MergeRemovedMasksTests(unittest.TestCase):
    def test_nearby_fragments_merge(self) -> None:
        first = np.zeros((300, 300), np.uint8)
        second = np.zeros_like(first)
        cv2.rectangle(first, (50, 120), (90, 140), 1, -1)
        gap_px = int(round(5.0 / float(base.MM_PX)))
        cv2.rectangle(second, (90 + gap_px, 120), (120 + gap_px, 145), 1, -1)

        merged = merge_removed_masks([first, second])

        self.assertEqual(len(merged), 1)
        self.assertGreater(int(merged[0].sum()), int(first.sum()))

    def test_distant_fragments_remain_separate(self) -> None:
        first = np.zeros((400, 400), np.uint8)
        second = np.zeros_like(first)
        cv2.rectangle(first, (30, 100), (60, 130), 1, -1)
        cv2.rectangle(second, (250, 100), (280, 130), 1, -1)

        merged = merge_removed_masks([first, second])

        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
