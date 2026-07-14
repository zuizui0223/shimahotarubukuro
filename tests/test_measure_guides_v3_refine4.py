from __future__ import annotations

import unittest

from measure_guides_v3_refine4 import should_propagate


class ResidualSideBulbTests(unittest.TestCase):
    def test_mid_lateral_bulb_near_removed_appendage_is_removed(self) -> None:
        self.assertTrue(
            should_propagate(
                longitudinal_position=0.55,
                lateral_position=0.98,
                area_mm2=7.0,
                width_mm=3.7,
                distance_mm=4.6,
                same_side=True,
            )
        )

    def test_lobe_bearing_end_is_preserved(self) -> None:
        self.assertFalse(
            should_propagate(
                longitudinal_position=0.96,
                lateral_position=0.98,
                area_mm2=7.0,
                width_mm=3.7,
                distance_mm=4.6,
                same_side=True,
            )
        )

    def test_opposite_side_component_is_preserved(self) -> None:
        self.assertFalse(
            should_propagate(
                longitudinal_position=0.55,
                lateral_position=0.98,
                area_mm2=7.0,
                width_mm=3.7,
                distance_mm=4.6,
                same_side=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
