from __future__ import annotations

import unittest

import measure_guides_review_fold as fold


class ReviewedFoldStateTests(unittest.TestCase):
    def test_shikine_manual_fold_states_match_sheet_review(self) -> None:
        expected = {
            1: ("opened_full", 5.0),
            2: ("folded_half", 2.5),
            3: ("folded_half", 2.5),
            4: ("opened_full", 5.0),
            5: ("opened_full", 5.0),
            6: ("opened_full", 5.0),
        }
        for corolla_id, (state, visible_lobes) in expected.items():
            reviewed = fold.get_reviewed_fold_state(
                "shikinejima", "shikine1~4", corolla_id
            )
            self.assertIsNotNone(reviewed)
            assert reviewed is not None
            self.assertEqual(reviewed["state"], state)
            self.assertEqual(reviewed["visible_lobes"], visible_lobes)

    def test_folded_half_doubles_width_and_area_proxies(self) -> None:
        summary = {
            "corolla_area_ruler_mm2": 100.0,
            "flat_throat_span_mm": 10.0,
            "corolla_max_span_ruler_mm": 20.0,
            "flat_mid_tube_width_mm": 8.0,
            "flat_basal_tube_width_mm": 6.0,
            "fold_state_auto": "opened_or_broad",
            "visitor_trait_qc": "folded_mouth_proxy|sinus",
        }
        fold.apply_reviewed_fold_state(
            summary,
            folder="shikinejima",
            sheet="shikine1~4",
            corolla_id=2,
        )
        self.assertEqual(summary["fold_state_reviewed"], "folded_half")
        self.assertEqual(summary["fold_width_area_correction_factor"], 2.0)
        self.assertEqual(summary["corolla_area_unfolded_est_mm2"], 200.0)
        self.assertEqual(summary["throat_span_unfolded_est_mm"], 20.0)
        self.assertNotIn("folded_mouth_proxy", summary["visitor_trait_qc"])

    def test_open_full_keeps_measured_width_and_area(self) -> None:
        summary = {
            "corolla_area_ruler_mm2": 100.0,
            "flat_throat_span_mm": 10.0,
            "corolla_max_span_ruler_mm": 20.0,
            "flat_mid_tube_width_mm": 8.0,
            "flat_basal_tube_width_mm": 6.0,
            "fold_state_auto": "folded_half",
            "visitor_trait_qc": "",
        }
        fold.apply_reviewed_fold_state(
            summary,
            folder="shikinejima",
            sheet="shikine1~4",
            corolla_id=1,
        )
        self.assertEqual(summary["fold_state_reviewed"], "opened_full")
        self.assertEqual(summary["fold_width_area_correction_factor"], 1.0)
        self.assertEqual(summary["corolla_area_unfolded_est_mm2"], 100.0)
        self.assertEqual(summary["throat_span_unfolded_est_mm"], 10.0)


if __name__ == "__main__":
    unittest.main()
