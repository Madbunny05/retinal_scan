"""Deterministic smoke tests for safe optic-disc topography."""
import unittest

import cv2
import numpy as np

import reconstruction3d


def synthetic_fundus() -> np.ndarray:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 235, (80, 45, 25), -1)
    cv2.circle(img, (310, 245), 58, (225, 205, 145), -1)
    cv2.circle(img, (310, 245), 25, (248, 238, 190), -1)
    for angle in range(0, 360, 30):
        end = (int(310 + 160 * np.cos(np.deg2rad(angle))), int(245 + 160 * np.sin(np.deg2rad(angle))))
        cv2.line(img, (310, 245), end, (45, 22, 14), 3)
    return img


class RelativeTopographyTests(unittest.TestCase):
    def test_returns_figure_for_clear_central_disc(self):
        result = reconstruction3d.analyze_optic_disc(synthetic_fundus())
        self.assertTrue(result["available"], result["reason"])
        self.assertIsNotNone(result["figure"])
        self.assertIn("not physical depth", result["note"])

    def test_rejects_non_fundus_without_throwing(self):
        result = reconstruction3d.analyze_optic_disc(np.full((512, 512, 3), 128, dtype=np.uint8))
        self.assertFalse(result["available"])
        self.assertIsNone(result["figure"])


if __name__ == "__main__":
    unittest.main()
