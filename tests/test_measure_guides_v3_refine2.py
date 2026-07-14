from __future__ import annotations

import unittest

import cv2
import numpy as np

import measure_guides_v3_refine as refine
import measure_guides_v3_refine2 as refine2
import measure_guides_v3_refine3 as refine3  # noqa: F401  (installs safe wrapper)


class MultiscaleRefinementTests(unittest.TestCase):
    def _channels(self, image: np.ndarray) -> None:
        refine2._CURRENT_CHANNELS = refine._lab_channels(image)
        self.addCleanup(setattr, refine2, "_CURRENT_CHANNELS", None)

    def test_removes_narrow_neck_appendage_but_preserves_lobe(self) -> None:
        image = np.full((560, 560, 3), 250, np.uint8)
        mask = np.zeros((560, 560), np.uint8)

        # Broad corolla body and a genuine lower lobe with a wide attachment.
        cv2.ellipse(mask, (250, 260), (125, 175), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (250, 415), (62, 70), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (210, 350), (290, 410), 1, -1)

        # C7-like side style/ovary: only about 4-5 mm wide and linked through
        # a narrow neck. It is broad enough to evade the former aspect>=4 rule.
        cv2.rectangle(mask, (368, 244), (388, 256), 1, -1)
        cv2.ellipse(mask, (410, 250), (24, 58), 0, 0, 360, 1, -1)

        image[mask > 0] = (155, 175, 215)
        self._channels(image)

        cleaned, removed = refine.detach_thin_appendages(mask)

        self.assertTrue(removed)
        self.assertEqual(int(cleaned[250, 410]), 0, "side appendage should be removed")
        self.assertEqual(int(cleaned[420, 250]), 1, "genuine broad lobe should remain")
        self.assertEqual(int(cleaned[260, 250]), 1, "corolla body should remain")

    def test_safety_guard_does_not_recurse(self) -> None:
        image = np.full((260, 260, 3), 250, np.uint8)
        mask = np.zeros((260, 260), np.uint8)
        cv2.rectangle(mask, (105, 25), (145, 235), 1, -1)
        image[mask > 0] = (160, 180, 220)
        self._channels(image)

        cleaned, removed = refine.detach_thin_appendages(mask)

        self.assertEqual(cleaned.shape, mask.shape)
        self.assertIsInstance(removed, list)


if __name__ == "__main__":
    unittest.main()
