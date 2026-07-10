from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_v2 as v2
import measure_guides_review as review
import measure_guides_review_fast as fast


class ReviewedPipelineTests(unittest.TestCase):
    def test_large_open_single_corolla_is_not_split_by_area_alone(self) -> None:
        mask = np.zeros((900, 900), dtype=np.uint8)
        cv2.ellipse(mask, (450, 450), (285, 225), 0, 0, 360, 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "not_triggered")
        self.assertEqual(len(pieces), 1)

    def test_broad_single_body_rejects_internal_kmeans_split(self) -> None:
        mask = np.zeros((1200, 1200), dtype=np.uint8)
        cv2.ellipse(mask, (600, 600), (380, 300), 0, 0, 360, 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "not_triggered")
        self.assertEqual(len(pieces), 1)

    def test_two_touching_corollas_remain_two_bodies(self) -> None:
        mask = np.zeros((1200, 1500), dtype=np.uint8)
        cv2.ellipse(mask, (420, 650), (165, 275), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (850, 650), (185, 275), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (580, 620), (670, 680), 1, -1)

        pieces, status = review.try_split_reviewed(mask)

        self.assertEqual(status, "auto_split")
        self.assertEqual(len(pieces), 2)

    def test_short_paper_noise_is_not_an_organ(self) -> None:
        image = np.full((800, 800, 3), 255, dtype=np.uint8)
        cv2.line(image, (250, 400), (300, 400), (150, 170, 190), 5)
        corolla = np.zeros((800, 800), dtype=np.uint8)

        organs = fast.organs_fast(image, corolla, 100)

        self.assertEqual(organs, [])

    def test_shikine_review_records_only_visible_candidates(self) -> None:
        rows = review.manual_organ_rows(
            (1900, 1189, 3),
            ("shikinejima", "shikine1~4"),
        )

        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(len(rows), 4)
        hints = {row["nearest_corolla_hint"] for row in rows}
        self.assertEqual(hints, {"C2", "C4", "C5", "C6"})
        self.assertNotIn("C3", hints)
        self.assertTrue(all(row["association_confirmed"] == 0 for row in rows))
        self.assertTrue(
            all(row["organ_type_auto"] == "visible_reproductive_organ_candidate" for row in rows)
        )

    def test_shikine_pair_uses_smaller_local_closing(self) -> None:
        rule = review.LOCAL_FOREGROUND_RULES[("shikinejima", "shikine1~4")]
        self.assertEqual(rule["close_size"], 17)
        self.assertLess(rule["close_size"], 27)

    def test_shikine_raw_pair_is_separate_without_kmeans_split(self) -> None:
        image_path = Path("shimahotarubukuro/shikinejima/shikine1~4.jpg")
        if not image_path.exists():
            self.skipTest("private Shikine scan is unavailable")

        image = base.load_bgr(str(image_path))
        top = v2.specimen_top(image)
        previous = review._CURRENT_SHEET
        review._CURRENT_SHEET = ("shikinejima", "shikine1~4")
        try:
            filled, _, _ = review.foreground_reviewed(image, top)
            components = v2.corollas(filled, auto_split=True)
        finally:
            review._CURRENT_SHEET = previous

        height, width = filled.shape
        pair = [
            component for component in components
            if 0.44 * height <= component["cy"] <= 0.68 * height
            and component["cx"] <= 0.60 * width
        ]
        self.assertEqual(len(components), 6)
        self.assertEqual(len(pair), 2)
        self.assertTrue(
            all(component["split_status"] == "not_triggered" for component in pair)
        )


if __name__ == "__main__":
    unittest.main()
