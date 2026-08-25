import math
import unittest

import cv2
import numpy as np

from codex.wafer_via_die_render import (
    build_die_map_from_yolo,
    measure_wafer_angle_die_render,
)


def synthetic_rotated_wafer(angle_deg=2.4):
    size = 900
    center = (450, 450)
    radius = 390
    pitch_x, pitch_y = 70, 82
    base = np.full((size, size, 3), 30, np.uint8)
    cv2.circle(base, center, radius, (150, 150, 150), -1)
    for x in range(30, size, pitch_x):
        cv2.line(base, (x, 0), (x, size - 1), (245, 245, 245), 3)
    for y in range(40, size, pitch_y):
        cv2.line(base, (0, y), (size - 1, y), (245, 245, 245), 3)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    base[mask == 0] = 0

    rotation = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
    wafer = cv2.warpAffine(base, rotation, (size, size), flags=cv2.INTER_CUBIC)
    return wafer, rotation, center, radius, pitch_x, pitch_y


class DieRenderAngleTests(unittest.TestCase):
    def test_projection_and_fft_recover_full_wafer_angle(self):
        wafer, _, center, radius, _, _ = synthetic_rotated_wafer(2.4)
        result = measure_wafer_angle_die_render(
            wafer,
            center[0],
            center[1],
            radius,
            max_dim=700,
            fft_max_dim=700,
        )

        self.assertLess(abs(result["angle"] - 2.4), 0.03)
        self.assertGreaterEqual(result["confidence"], 0.9)
        self.assertTrue(result["agree"])

    def test_separate_builder_uses_die_render_option(self):
        wafer, rotation, center, _, pitch_x, pitch_y = synthetic_rotated_wafer(2.4)
        clip_x = clip_y = 194
        clip = wafer[clip_y:clip_y + 512, clip_x:clip_x + 512]
        detections = []
        for iy in range(-2, 3):
            for ix in range(-2, 3):
                full_point = rotation @ np.asarray(
                    (center[0] + ix * pitch_x, center[1] + iy * pitch_y, 1.0)
                )
                detections.append(
                    (float(full_point[0] - clip_x), float(full_point[1] - clip_y))
                )

        die_map = build_die_map_from_yolo(
            wafer,
            clip,
            np.asarray(detections),
            detection_format="point",
            clip_origin=(clip_x, clip_y),
            refine=False,
            angle_align_method="die_render",
            die_render_max_dim=700,
            die_render_fft_max_dim=700,
            return_aligned_image=False,
        )

        self.assertEqual(die_map.angle_align_method, "die_render")
        self.assertLess(abs(die_map.grid_angle_deg - 2.4), 0.03)
        self.assertAlmostEqual(die_map.pitch_x, pitch_x, places=5)
        self.assertAlmostEqual(die_map.pitch_y, pitch_y, places=5)
        self.assertGreater(die_map.num_dies, 50)
        self.assertTrue(die_map.angle_agree)
        self.assertLess(abs(die_map.die_render_info["final_residual"]), 0.01)
        self.assertTrue(math.isfinite(die_map.yolo_angle_deg))
        self.assertEqual(die_map.angle_pairs_full, ())
        self.assertEqual(die_map.grid_estimate.angle_pairs_clip, ())
        self.assertGreater(len(die_map.yolo_angle_pairs_full), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
