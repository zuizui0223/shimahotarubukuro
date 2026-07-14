from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine8 as refine8


class ConcaveTouchingSplitTests(unittest.TestCase):
    def test_moderate_touching_pair_is_split(self) -> None:
        # Two corollas sized like Niijima C8: below the old 55 mm / 1350 mm2
        # thresholds, separated enough to make a strong waist, and joined only
        # by a narrow preparation artefact.
        mask = np.zeros((900, 1200), np.uint8)
        cv2.ellipse(mask, (420, 470), (135, 185), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (770, 470), (135, 185), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (552, 458), (638, 482), 1, -1)

        measured = refine8.v2.metrics(mask)
        area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
        length_mm = float(measured["length_px"]) * float(base.MM_PX)
        self.assertLess(area_mm2, refine8.v2.SPLIT_AREA_MM2)
        self.assertLess(length_mm, refine8.v2.SPLIT_LEN_MM)
        self.assertLess(float(measured["solidity"]), refine8._CONCAVE_SOLIDITY_MAX)
        self.assertTrue(
            refine8.should_try_concave_split(
                area_mm2=area_mm2,
                length_mm=length_mm,
                solidity=float(measured["solidity"]),
            )
        )

        pieces, status = refine8.try_split_with_concavity(mask)

        self.assertEqual(status, "auto_split")
        self.assertEqual(len(pieces), 2)
        areas = [int(piece.sum()) for piece in pieces]
        self.assertGreater(min(areas) / max(areas), 0.75)

    def test_broad_single_corolla_is_not_split(self) -> None:
        # C16-like broad single flower: similar total size, but convex/solid.
        mask = np.zeros((900, 1200), np.uint8)
        cv2.ellipse(mask, (600, 470), (270, 180), 0, 0, 360, 1, -1)

        measured = refine8.v2.metrics(mask)
        area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
        length_mm = float(measured["length_px"]) * float(base.MM_PX)
        self.assertLess(area_mm2, refine8.v2.SPLIT_AREA_MM2)
        self.assertLess(length_mm, refine8.v2.SPLIT_LEN_MM)

        self.assertFalse(
            refine8.should_try_concave_split(
                area_mm2=area_mm2,
                length_mm=length_mm,
                solidity=float(measured["solidity"]),
            )
        )
        pieces, status = refine8.try_split_with_concavity(mask)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(status, "not_triggered")

    def test_trigger_requires_all_three_conditions(self) -> None:
        self.assertFalse(
            refine8.should_try_concave_split(
                area_mm2=700.0, length_mm=52.0, solidity=0.68
            )
        )
        self.assertFalse(
            refine8.should_try_concave_split(
                area_mm2=1000.0, length_mm=40.0, solidity=0.68
            )
        )
        self.assertFalse(
            refine8.should_try_concave_split(
                area_mm2=1000.0, length_mm=52.0, solidity=0.86
            )
        )


if __name__ == "__main__":
    unittest.main()
