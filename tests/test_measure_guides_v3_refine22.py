from __future__ import annotations

import unittest

import measure_guides_v3_refine22 as refine22


class LowConfidenceGrowthGateTests(unittest.TestCase):
    def test_high_confidence_mask_is_not_eligible(self) -> None:
        row = {"mask_confidence": 1.0, "mask_qc_required": 0}
        self.assertFalse(refine22.should_attempt_growth(row))

    def test_qc_required_mask_is_eligible(self) -> None:
        row = {"mask_confidence": 1.0, "mask_qc_required": 1}
        self.assertTrue(refine22.should_attempt_growth(row))

    def test_low_confidence_mask_is_eligible(self) -> None:
        row = {"mask_confidence": 0.7, "mask_qc_required": 0}
        self.assertTrue(refine22.should_attempt_growth(row))


if __name__ == "__main__":
    unittest.main()
