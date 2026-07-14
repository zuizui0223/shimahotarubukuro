from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v3_refine8 as refine8


class ConcaveTouchingSplitTests(unittest.TestCase):
    def test_moderate_touching_pair_is_split(self) -> None:
        # Two overlapping corollas sized so that the old 55 mm / 1350 mm2
        # thresholds do not trigger, but the union has a pronounced concavity.
        mask = np.zeros((900, 1200), np.uint8)
        cv2.ellipse(mask, (430, 470), (150, 210), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (730, 470), (150, 210), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (575, 450), (585, 490), 1, -1)

        measured = refine8.v2.metrics(mask)
        area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
        length_mm = float(measured["length_px"]) * float(base.MM_PX)
        self.assertLess(area_mm2, refine8.v2.SPLIT_AREA_MM2)
        self.assertLess(length_mm, refine8.v2.SPLIT_LEN_MM)
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
        mask = np.zeros((900, 1200), np.uint8)
        cv2.ellipse(mask, (600, 470), (290, 220), 0, 0, 360, 1, -1)
        # Add broad lobes without creating a narrow waist.
        for x in (430, 520, 610, 700, 790):
            cv2.circle(mask, (x, 650), 70, 1, -1)
        cv2.rectangle(mask, (360, 470), (840, 650), 1, -1)

        measured = refine8.v2.metrics(mask)
        area_mm2 = float(measured["area_px"]) * float(base.MM2_PX)
        length_mm = float(measured["length_px"]) * float(base.MM_PX)

        self.assertFalse(
            refine8.should_try_concave_split(
                area_mm2=area_mm2,
                length_mm=length_mm,
                solidity=float(measured["solidity"]),
            )
        )
        pieces, status = refine8.try_split_with_concavity(mask)
        self.assertEqual(len(pieces), 1)
        self.assertNotEqual(status, "auto_split")

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
