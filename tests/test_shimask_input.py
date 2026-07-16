from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v2 as v2
import shimask_input


def _scene():
    raw = np.full((640, 520, 3), 245, np.uint8)
    annotated = raw.copy()
    cv2.rectangle(annotated, (120, 110), (330, 400), (0, 0, 255), 5)
    cv2.circle(annotated, (200, 200), 14, (150, 70, 150), -1)
    points = np.array([[410, 150], [440, 210], [405, 270], [430, 330]], np.int32)
    cv2.polylines(annotated, [points], False, (0, 255, 0), 9)
    return raw, annotated


class StrokeExtractionTests(unittest.TestCase):
    def test_strokes_isolated_from_natural_purple(self) -> None:
        _, annotated = _scene()
        red, green = shimask_input.stroke_masks(annotated)
        self.assertGreater(int(red[110, 200]), 0)
        self.assertEqual(int(red[200, 200]), 0)
        self.assertGreater(int(green[210, 440]), 0)


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
        self.assertEqual(row["detection_source"], "shimask_green_stroke")
        self.assertGreater(float(row["length_mm"]), float(row["width_mm"]))
        self.assertGreater(float(row["skeleton_length_mm"]), float(row["endpoint_distance_mm"]))
        self.assertTrue(390 < float(row["cx"]) < 450)


if __name__ == "__main__":
    unittest.main()
