from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine9 as refine9


class FragmentExclusionTests(unittest.TestCase):
    def test_narrow_small_fragment_is_excluded(self) -> None:
        filled = np.zeros((900, 900), np.uint8)
        # Roughly 8 x 18 mm: narrow and under the 150 mm2 fragment area limit.
        cv2.rectangle(filled, (250, 300), (344, 512), 255, -1)

        components = refine9.corollas_without_fragments(filled, auto_split=False)

        self.assertEqual(components, [])

    def test_small_real_corolla_is_retained(self) -> None:
        filled = np.zeros((900, 900), np.uint8)
        # About 18 x 24 mm, in the range of the smallest genuine corollas.
        cv2.ellipse(filled, (450, 450), (106, 142), 0, 0, 360, 255, -1)

        components = refine9.corollas_without_fragments(filled, auto_split=False)

        self.assertEqual(len(components), 1)


if __name__ == "__main__":
    unittest.main()
