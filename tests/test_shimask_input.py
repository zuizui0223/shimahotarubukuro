from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v2 as v2
import shimask_input


def _scene():
    """White sheet + a red corolla outline + a green organ bar, plus a purple
    natural guide spot that must NOT be mistaken for a red stroke."""
    raw = np.full((640, 520, 3), 245, np.uint8)
    annotated = raw.copy()
    cv2.rectangle(annotated, (120, 110), (330, 400), (0, 0, 255), 5)   # red corolla outline
    cv2.circle(annotated, (200, 200), 14, (150, 70, 150), -1)          # natural purple guide (BGR)
    cv2.line(annotated, (410, 150), (410, 300), (0, 255, 0), 9)        # green organ stroke
    return raw, annotated


class StrokeExtractionTests(unittest.TestCase):
    def test_strokes_isolated_from_natural_purple(self) -> None:
        _, annotated = _scene()
        red, green = shimask_input.stroke_masks(annotated)
        # red picks up the outline but not the purple guide dot
        self.assertGreater(int(red[110, 200]), 0)          # on the top red edge
        self.assertEqual(int(red[200, 200]), 0)            # purple guide is not red
        self.assertGreater(int(green[220, 410]), 0)        # on the green bar


class RedCorollaTests(unittest.TestCase):
    def test_red_outline_becomes_one_filled_component(self) -> None:
        raw, annotated = _scene()
        comps = shimask_input.red_corolla_components(raw, annotated)
        self.assertEqual(len(comps), 1)
        comp = comps[0]
        for key in ("mask", "cx", "cy", "m", "source_component_id", "split_piece", "split_status"):
            self.assertIn(key, comp)
        self.assertTrue(comp["mask"].dtype == bool)
        # interior is filled (not just the outline)
        self.assertGreater(int(comp["mask"].sum()), 5000)
        self.assertTrue(120 < comp["cx"] < 330 and 110 < comp["cy"] < 400)
        # matches v2.metrics contract used downstream
        self.assertIn("contour", comp["m"])

    def test_component_is_consumable_by_v2_metrics(self) -> None:
        raw, annotated = _scene()
        comp = shimask_input.red_corolla_components(raw, annotated)[0]
        measured = v2.metrics(comp["mask"].astype(np.uint8))
        self.assertIsNotNone(measured)


class GreenOrganTests(unittest.TestCase):
    def test_green_stroke_becomes_one_organ_row(self) -> None:
        raw, annotated = _scene()
        rows = shimask_input.green_organ_rows(raw, annotated)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("cx", "cy", "length_mm", "width_mm", "aspect", "angle_deg",
                    "organ_type_auto", "detection_source", "x1", "y1", "x2", "y2"):
            self.assertIn(key, row)
        self.assertEqual(row["detection_source"], "shimask_green_stroke")
        self.assertGreater(float(row["length_mm"]), float(row["width_mm"]))
        self.assertTrue(380 < float(row["cx"]) < 440)


if __name__ == "__main__":
    unittest.main()
