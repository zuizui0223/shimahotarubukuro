import unittest

import cv2
import numpy as np

import measure_guides_symmetry_axis as symmetry


class SymmetryAxisTests(unittest.TestCase):
    def test_recovers_oblique_axis_from_symmetric_polygon(self):
        mask = np.zeros((420, 320), dtype=np.uint8)
        centre = (160, 210)
        cv2.ellipse(mask, centre, (70, 145), 18, 0, 360, 1, -1)
        result = symmetry.estimate_symmetry_axis(mask)
        self.assertGreater(result.score_iou, 0.90)
        # The ellipse major axis is roughly 18 degrees from vertical.
        tilt = abs(90.0 - result.angle_deg_from_x)
        self.assertLess(abs(tilt - 18.0), 3.0)
        self.assertLess(result.base_xy[1], result.tip_xy[1])

    def test_rotation_places_base_above_tip(self):
        mask = np.zeros((300, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 150), (55, 105), -12, 0, 360, 1, -1)
        result = symmetry.estimate_symmetry_axis(mask)
        rotated = symmetry.rotate_mask_to_symmetry_axis(mask, result)
        self.assertGreater(int(rotated.sum()), 0)
        self.assertGreater(rotated.shape[0], rotated.shape[1])


if __name__ == "__main__":
    unittest.main()
