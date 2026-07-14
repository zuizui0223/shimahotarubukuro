from __future__ import annotations

import unittest

from measure_guides_v3_refine6 import should_remove_guide_free


class GuideSupportedPruningTests(unittest.TestCase):
    def test_mid_lateral_guide_free_loop_is_removed(self) -> None:
        self.assertTrue(
            should_remove_guide_free(
                area_mm2=16.0,
                aspect=1.7,
                spot_fraction=0.0,
                median_pigment_index=-11.0,
                median_chroma=13.0,
                neck_mm=7.0,
                longitudinal_position=0.45,
                lateral_position=1.05,
            )
        )

    def test_terminal_narrow_style_base_is_removed(self) -> None:
        self.assertTrue(
            should_remove_guide_free(
                area_mm2=11.0,
                aspect=1.75,
                spot_fraction=0.0,
                median_pigment_index=-9.0,
                median_chroma=9.5,
                neck_mm=2.7,
                longitudinal_position=1.09,
                lateral_position=0.45,
            )
        )

    def test_low_chroma_paper_fold_is_removed(self) -> None:
        self.assertTrue(
            should_remove_guide_free(
                area_mm2=9.7,
                aspect=2.0,
                spot_fraction=0.0,
                median_pigment_index=-6.0,
                median_chroma=4.5,
                neck_mm=6.3,
                longitudinal_position=0.87,
                lateral_position=0.87,
            )
        )

    def test_guide_supported_lobe_is_preserved(self) -> None:
        self.assertFalse(
            should_remove_guide_free(
                area_mm2=15.5,
                aspect=1.4,
                spot_fraction=0.49,
                median_pigment_index=4.0,
                median_chroma=11.0,
                neck_mm=6.6,
                longitudinal_position=0.94,
                lateral_position=0.85,
            )
        )

    def test_broad_terminal_pale_lobe_is_preserved(self) -> None:
        self.assertFalse(
            should_remove_guide_free(
                area_mm2=14.9,
                aspect=1.9,
                spot_fraction=0.03,
                median_pigment_index=-12.0,
                median_chroma=13.3,
                neck_mm=9.4,
                longitudinal_position=1.01,
                lateral_position=0.38,
            )
        )

    def test_pale_upper_tube_edge_is_preserved(self) -> None:
        self.assertFalse(
            should_remove_guide_free(
                area_mm2=3.7,
                aspect=1.9,
                spot_fraction=0.0,
                median_pigment_index=-4.0,
                median_chroma=4.1,
                neck_mm=4.7,
                longitudinal_position=0.05,
                lateral_position=0.44,
            )
        )


if __name__ == "__main__":
    unittest.main()
