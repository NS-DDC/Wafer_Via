import runpy
import tempfile
import unittest
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np

from codex.wafer_notch_v5_adaptive_background import (
    draw_aligned_wafer_notch_guide,
)


def synthetic_wafer(background_bgr, size=700):
    center = (350, 350)
    radius = 300
    image = np.full((size, size, 3), background_bgr, dtype=np.uint8)
    cv2.circle(image, center, radius, (105, 125, 138), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (175, 188, 196), 4, cv2.LINE_AA)
    for x in range(center[0] - 250, center[0] + 251, 32):
        cv2.line(
            image,
            (x, center[1] - 270),
            (x, center[1] + 270),
            (150, 165, 154),
            1,
            cv2.LINE_AA,
        )
    yy, xx = np.indices((size, size))
    outside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 > radius ** 2
    image[outside] = background_bgr
    notch_mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(
        notch_mask,
        (center[0], center[1] + radius + 5),
        (30, 14),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    image[notch_mask > 0] = background_bgr
    return image, center, radius


class AdaptiveBackgroundNotchTests(unittest.TestCase):
    def test_black_blue_and_pink_backgrounds_return_same_geometry(self):
        backgrounds = ((0, 0, 0), (145, 62, 18), (150, 82, 220))
        results = []
        for background in backgrounds:
            image, center, radius = synthetic_wafer(background)
            result = draw_aligned_wafer_notch_guide(image)
            self.assertTrue(result.found)
            self.assertEqual(result.overlay_image.shape, image.shape)
            self.assertLess(
                np.linalg.norm(np.asarray(result.wafer_center_px) - center), 3.0
            )
            self.assertLess(abs(result.wafer_radius_px - radius), 7.0)
            self.assertLess(abs(result.residual_angle_deg), 0.5)
            self.assertIsNotNone(result.notch_point_px)
            results.append(result)

        reference = results[0]
        for result in results[1:]:
            self.assertTrue(np.allclose(
                result.wafer_center_px, reference.wafer_center_px, atol=0.2
            ))
            self.assertAlmostEqual(
                result.wafer_radius_px, reference.wafer_radius_px, places=3
            )
            self.assertTrue(np.allclose(
                result.notch_point_px, reference.notch_point_px, atol=0.2
            ))
            self.assertAlmostEqual(
                result.residual_angle_deg,
                reference.residual_angle_deg,
                places=3,
            )

    def test_file_is_copy_paste_standalone(self):
        source = (
            Path(__file__).parents[1]
            / "codex"
            / "wafer_notch_v5_adaptive_background.py"
        )
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("import wafer_notch_angle", text)
        self.assertNotIn("import wafer_via", text)
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / source.name
            copy2(source, isolated)
            namespace = runpy.run_path(str(isolated))

        image, _, _ = synthetic_wafer((145, 62, 18))
        result = namespace["draw_aligned_wafer_notch_guide"](image)
        self.assertTrue(result.found)
        self.assertEqual(result.overlay_image.shape, image.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
