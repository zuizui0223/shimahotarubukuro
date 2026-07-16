from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v2 as v2
import shimask_input


def _scene():
    raw = np.full((640, 520, 3), 245, np.uint8)
    # Natural purple guide exists in both raw and reviewed images.
    cv2.circle(raw, (200, 200), 14, (150, 70, 150), -1)
    annotated = raw.copy()
    cv2.rectangle(annotated, (120, 110), (330, 400), (0, 0, 255), 5)
    points = np.array([[410, 150], [440, 210], [405, 270], [430, 330]], np.int32)
    cv2.polylines(annotated, [points], False, (0, 255, 0), 9)
    return raw, annotated


class StrokeExtractionTests(unittest.TestCase):
    def test_strokes_come_from_raw_difference(self) -> None:
        raw, annotated = _scene()
        red, green = shimask_input.stroke_masks(raw, annotated)
        self.assertGreater(int(red[110, 200]), 0)
        self.assertGreater(int(green[210, 440]), 0)

    def test_unchanged_natural_purple_is_never_red(self) -> None:
        raw, annotated = _scene()
        red, _ = shimask_input.stroke_masks(raw, annotated)
        self.assertEqual(int(red[200, 200]), 0)

    def test_small_jpeg_like_noise_is_ignored(self) -> None:
        raw, annotated = _scene()
        noise = np.zeros_like(annotated, dtype=np.int16)
        noise[:, :, 0] = 3
        noise[:, :, 1] = -2
        noise[:, :, 2] = 2
        noisy = np.clip(annotated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        red, green = shimask_input.stroke_masks(raw, noisy)
        self.assertEqual(int(red[200, 200]), 0)
        self.assertGreater(int(red[110, 200]), 0)
        self.assertGreater(int(green[210, 440]), 0)

    def test_colour_stats_are_measured_from_review_pixels(self) -> None:
        raw, annotated = _scene()
        rows = {row["stroke"]: row for row in shimask_input.stroke_colour_rows(raw, annotated)}
        self.assertEqual(rows["red_corolla_outline"]["R_mode"], 255)
        self.assertEqual(rows["red_corolla_outline"]["G_mode"], 0)
        self.assertEqual(rows["green_reproductive_organ"]["G_mode"], 255)


class RedCorollaTests(unittest.TestCase):
    def test_red_outline_becomes_one_filled_component(self) -> None:
        raw, annotated = _scene()
        components = shimask_input.red_corolla_components(raw, annotated)
        self.assertEqual(len(components), 1)
        component = components[0]
        for key in ("mask", "cx", "cy", "m", "source_component_id", "split_piece", "split_status"):
            self.assertIn(key, component)
        self.assertTrue(component["mask"].dtype == bool)
        self.assertGreater(int(component["mask"].sum()), 5000)
        self.assertTrue(120 < component["cx"] < 330 and 110 < component["cy"] < 400)
        self.assertIn("contour", component["m"])

    def test_boundary_is_not_eroded_inward(self) -> None:
        raw, annotated = _scene()
        mask = shimask_input.red_corolla_components(raw, annotated)[0]["mask"]
        ys, xs = np.where(mask)
        self.assertLessEqual(int(xs.min()), 120)
        self.assertGreaterEqual(int(xs.max()), 330)
        self.assertLessEqual(int(ys.min()), 110)
        self.assertGreaterEqual(int(ys.max()), 400)

    def test_component_is_consumable_by_v2_metrics(self) -> None:
        raw, annotated = _scene()
        component = shimask_input.red_corolla_components(raw, annotated)[0]
        self.assertIsNotNone(v2.metrics(component["mask"].astype(np.uint8)))


class GreenOrganTests(unittest.TestCase):
    def test_green_stroke_becomes_one_organ_row(self) -> None:
        raw, annotated = _scene()
        rows = shimask_input.green_organ_rows(raw, annotated)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in (
            "cx", "cy", "length_mm", "skeleton_length_mm", "endpoint_distance_mm",
            "width_mm", "aspect", "angle_deg", "organ_type_auto", "detection_source",
            "x1", "y1", "x2", "y2",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["detection_source"], "shimask_green_stroke_from_raw_difference")
        self.assertGreater(float(row["length_mm"]), float(row["width_mm"]))
        self.assertGreater(float(row["skeleton_length_mm"]), float(row["endpoint_distance_mm"]))
        self.assertTrue(390 < float(row["cx"]) < 450)


if __name__ == "__main__":
    unittest.main()
