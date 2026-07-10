from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v2 as v2


class MeasureGuidesV2Tests(unittest.TestCase):
    def test_specimen_top_falls_below_ruler_edge(self) -> None:
        image = np.full((1000, 800, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 300), (750, 300), (0, 0, 0), 4)

        top = v2.specimen_top(image)

        self.assertGreaterEqual(top, 315)
        self.assertLessEqual(top, 345)

    def test_touching_corollas_are_split_conservatively(self) -> None:
        # Two biologically plausible, side-by-side flattened corollas. Their overlap
        # makes one connected component, while the union is too long to be one flower.
        mask = np.zeros((1200, 1400), dtype=np.uint8)
        cv2.ellipse(mask, (480, 650), (175, 275), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (780, 650), (175, 275), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (640, 610), (660, 690), 1, -1)
        cv2.setRNGSeed(20260710)

        pieces, status = v2.try_split(mask)

        self.assertEqual(status, "auto_split")
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            measured = v2.metrics(piece)
            self.assertIsNotNone(measured)
            area_mm2 = measured["area_px"] * v2.base.MM2_PX
            length_mm = measured["length_px"] * v2.base.MM_PX
            self.assertGreaterEqual(area_mm2, v2.base.AREA_MM2_MIN)
            self.assertLessEqual(length_mm, 60.0)


if __name__ == "__main__":
    unittest.main()
