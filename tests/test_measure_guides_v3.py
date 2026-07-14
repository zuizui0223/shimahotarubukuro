from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np


base = types.ModuleType("measure_guides")
base.MM_PX = 0.1
base.MM2_PX = 0.01
base.AREA_MM2_MIN = 20.0
base.AREA_MM2_MAX = 2000.0
base.ASPECT_MAX = 6.0
base.SOLIDITY_MIN = 0.35

v2 = types.ModuleType("measure_guides_v2")


def metrics(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    rw, rh = cv2.minAreaRect(contour)[1]
    hull = cv2.contourArea(cv2.convexHull(contour))
    area = float(mask.sum())
    return {
        "contour": contour,
        "area_px": area,
        "length_px": max(rw, rh),
        "width_px": min(rw, rh),
        "solidity": area / hull if hull else 0.0,
        "aspect": max(rw, rh) / max(min(rw, rh), 1e-6),
    }


v2.metrics = metrics
v2.is_fragment = lambda area, width: area < 15.0 and width < 5.0
spec = importlib.util.spec_from_file_location(
    "measure_guides_v3", Path(__file__).resolve().parents[1] / "measure_guides_v3.py"
)
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
with patch.dict(
    sys.modules,
    {"measure_guides": base, "measure_guides_v2": v2},
):
    spec.loader.exec_module(v3)


class MeasureGuidesV3Tests(unittest.TestCase):
    def test_simple_corolla_is_high_confidence(self):
        mask = np.zeros((500, 500), np.uint8)
        cv2.ellipse(mask, (250, 250), (105, 155), 0, 0, 360, 1, -1)
        component = {
            "mask": mask.astype(bool),
            "split_status": "not_triggered",
            "m": metrics(mask),
        }
        result = v3.corolla_confidence(component, mm_per_px=0.1, mm2_per_px=0.01)
        self.assertEqual(result["mask_confidence_label"], "high")
        self.assertEqual(result["mask_qc_required"], 0)
        self.assertGreater(result["auto_max_width_mm"], 15.0)
        self.assertEqual(result["opening_width_status"], "deferred")

    def test_uncertain_split_requires_qc(self):
        mask = np.zeros((500, 500), np.uint8)
        cv2.rectangle(mask, (20, 220), (495, 260), 1, -1)
        component = {
            "mask": mask.astype(bool),
            "split_status": "split_rejected",
            "m": metrics(mask),
        }
        result = v3.corolla_confidence(component, mm_per_px=0.1, mm2_per_px=0.01)
        self.assertEqual(result["mask_qc_required"], 1)
        self.assertIn("split_rejected", result["mask_qc_reasons"])
        self.assertIn("touches_image_border", result["mask_qc_reasons"])

    def test_skeleton_length_is_close_to_straight_line(self):
        mask = np.zeros((120, 120), np.uint8)
        cv2.line(mask, (20, 60), (100, 60), 1, 7)
        features = v3.component_features(mask, mm_per_px=0.1)
        self.assertGreater(features["length_mm"], 7.5)
        self.assertLess(features["length_mm"], 9.5)
        self.assertEqual(features["n_branch_points"], 0)

    def test_swollen_tip_is_pistil_candidate(self):
        mask = np.zeros((220, 140), np.uint8)
        cv2.line(mask, (70, 190), (70, 45), 1, 7)
        cv2.ellipse(mask, (70, 35), (10, 15), 0, 0, 360, 1, -1)
        features = v3.component_features(mask, mm_per_px=0.1)
        label, confidence, _ = v3.classify_organ(features)
        self.assertEqual(label, "pistil_candidate")
        self.assertGreaterEqual(confidence, 0.72)

    def test_branched_candidate_is_stamen_bundle(self):
        mask = np.zeros((220, 180), np.uint8)
        cv2.line(mask, (90, 190), (90, 75), 1, 5)
        cv2.line(mask, (90, 80), (55, 35), 1, 5)
        cv2.line(mask, (90, 80), (125, 35), 1, 5)
        features = v3.component_features(mask, mm_per_px=0.1)
        label, confidence, _ = v3.classify_organ(features)
        self.assertEqual(label, "stamen_bundle_candidate")
        self.assertGreaterEqual(confidence, 0.65)

    def test_association_prefers_near_aligned_corolla(self):
        first = np.zeros((400, 600), np.uint8)
        second = np.zeros((400, 600), np.uint8)
        cv2.ellipse(first, (150, 210), (65, 110), 0, 0, 360, 1, -1)
        cv2.ellipse(second, (450, 210), (65, 110), 0, 0, 360, 1, -1)
        organ = {
            "cx": 165.0,
            "cy": 70.0,
            "angle_deg": 90.0,
            "endpoints": np.array([[165.0, 70.0], [160.0, 105.0]]),
        }
        result = v3.associate_organ(
            organ,
            [{"mask": first.astype(bool)}, {"mask": second.astype(bool)}],
            mm_per_px=0.1,
        )
        self.assertEqual(result["nearest_corolla"], 1)
        self.assertGreater(result["association_confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
