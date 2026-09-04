import io
import runpy
import tempfile
import unittest
import tokenize
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

    def test_full_adaptive_standalone_changes_only_angle_source(self):
        source = (
            Path(__file__).parents[1]
            / "codex"
            / "wafer_via_notch_adaptive_standalone.py"
        )
        text = source.read_text(encoding="utf-8")
        original_source = (
            Path(__file__).parents[1]
            / "codex"
            / "wafer_via_notch_standalone.py"
        ).read_text(encoding="utf-8")
        builder_start = original_source.index("def build_die_map_from_yolo(")
        builder_end = original_source.index("\n# INPUT", builder_start)
        original_builder = original_source[builder_start:builder_end].rstrip()
        self.assertNotIn("import wafer_notch_angle", text)
        self.assertNotIn("import wafer_via", text)
        comments = [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT
        ]
        self.assertEqual(len(comments), 13)
        self.assertEqual(comments[0].string, "# INPUT")
        self.assertEqual(comments[6].string, "# OUTPUT")
        self.assertTrue(
            text.rstrip().endswith(
                "# dm.notch_overlay_image / dm.notch_zoom_image: "
                "return_notch_visuals=True일 때만 생성"
            )
        )
        self.assertIn(original_builder, text)
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / source.name
            copy2(source, isolated)
            namespace = runpy.run_path(str(isolated))

        geometry = []
        for background in ((0, 0, 0), (145, 62, 18), (150, 82, 220)):
            image, center, _ = synthetic_wafer(background)
            clip_origin = (94, 94)
            clip = image[94:606, 94:606]
            detections = np.asarray([
                (
                    center[0] + ix * 70.0 - clip_origin[0],
                    center[1] + iy * 82.0 - clip_origin[1],
                )
                for iy in range(-2, 3)
                for ix in range(-2, 3)
            ])
            die_map = namespace["build_die_map_from_yolo"](
                image,
                clip,
                detections,
                detection_format="point",
                clip_origin=clip_origin,
                refine=False,
                return_aligned_image=True,
                notch_angle_samples=14400,
            )
            self.assertEqual(
                die_map.notch_detection_method,
                "v5_border_adaptive_angle_only",
            )
            self.assertEqual(die_map.grid_angle_deg, 0.0)
            self.assertAlmostEqual(die_map.pitch_x, 70.0, places=5)
            self.assertAlmostEqual(die_map.pitch_y, 82.0, places=5)
            self.assertEqual(die_map.aligned_image.shape, image.shape)
            self.assertIsNone(die_map.notch_overlay_image)
            self.assertIsNone(die_map.notch_zoom_image)
            located = namespace["locate_die"](
                die_map, point=(die_map.x0, die_map.y0)
            )
            self.assertIsNotNone(located)
            geometry.append((
                die_map.wafer_cx,
                die_map.wafer_cy,
                die_map.wafer_r,
                die_map.notch_angle_deg,
                die_map.image_rotation_deg,
            ))

        for values in geometry[1:]:
            self.assertTrue(np.allclose(values, geometry[0], atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
