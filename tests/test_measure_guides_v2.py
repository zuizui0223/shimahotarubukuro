from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v2 as v2


class MeasureGuidesV2Tests(unittest.TestCase):
    def test_specimen_top_falls_below_top_scale_bar(self) -> None:
        # A ruler scale bar near the top: a wide band of regularly spaced vertical
        # ticks. specimen_top must cut just below it so the ruler is excluded.
        image = np.full((1000, 800, 3), 255, dtype=np.uint8)
        for x in range(40, 760, 6):
            cv2.line(image, (x, 200), (x, 262), (0, 0, 0), 2)

        top = v2.specimen_top(image)

        self.assertGreaterEqual(top, 262)
        self.assertLessEqual(top, 330)

    def test_canonical_rotation_brings_ruler_to_top(self) -> None:
        # Ruler position is fixed per scan and recorded by file stem. Bottom-ruler
        # sheets get 180 deg, left-ruler sheets 90 deg clockwise; ruler-at-top
        # sheets (e.g. shikine) and unknown files are left unrotated.
        self.assertEqual(v2.base.canonical_rotation("oshima/oshima7.jpg"), cv2.ROTATE_180)
        self.assertEqual(
            v2.base.canonical_rotation("x/oshima10~13.jpg"), cv2.ROTATE_90_CLOCKWISE
        )
        self.assertIsNone(v2.base.canonical_rotation("shikinejima/shikine1~4.jpg"))
        self.assertIsNone(v2.base.canonical_rotation("whatever/kozu1.jpg"))

    def test_specimen_top_is_minimal_without_a_top_ruler(self) -> None:
        # No ruler at the top (it would be at the bottom): specimen_top must NOT
        # clip the upper part of the sheet, or specimens there are dropped.
        image = np.full((1000, 800, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (400, 350), (150, 200), 0, 0, 360, (120, 90, 160), -1)

        top = v2.specimen_top(image)

        self.assertLessEqual(top, int(1000 * 0.05))

    def test_is_fragment_excludes_slivers_keeps_small_corollas(self) -> None:
        # oshima4 C6 (pistil) and toshima2~3 C11 (thin strip) are fragments;
        # the smallest genuine corollas (~229-284 mm2, ~15-18 mm wide) are kept.
        self.assertTrue(v2.is_fragment(90.0, 11.1))   # oshima4 C6 = pistil
        self.assertTrue(v2.is_fragment(89.0, 7.1))    # toshima2~3 C11 = sliver
        self.assertFalse(v2.is_fragment(229.0, 15.3)) # toshima3~6 C6 = real
        self.assertFalse(v2.is_fragment(284.0, 18.3)) # kozu1 C6 = real
        self.assertFalse(v2.is_fragment(646.0, 28.0)) # median corolla

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
