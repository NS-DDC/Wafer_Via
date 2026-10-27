"""Exercise new refinement through grid estimation and the public map builder."""
import importlib
import unittest

import cv2
import numpy as np

from codex import wafer_via as base


def white_corner_grid():
    image = np.full((401, 401, 3), (30, 150, 45), np.uint8)
    truth = np.asarray([(x, y) for y in (100, 200, 300) for x in (100, 200, 300)], dtype=float)
    for x, y in truth.astype(int):
        image[y - 40:y - 4, x - 40:x - 4] = 255
    for p in (100, 200, 300):
        image[:, p - 4:p + 5] = 255
        image[p - 4:p + 5, :] = 255
    return image, truth, truth + (2.0, -2.0)


class CrossLinesPipelineTests(unittest.TestCase):
    def test_estimators_apply_threshold_and_preserve_raw_detection_points(self):
        image, truth, detections = white_corner_grid()
        for name in ("codex.wafer_via", "codex.wafer_via_die_render", "codex.wafer_via_notch",
                     "codex.wafer_via_notch_standalone", "codex.wafer_via_notch_adaptive_standalone"):
            module = importlib.import_module(name)
            notch = hasattr(module, "estimate_grid_from_yolo_notch")
            estimate = module.estimate_grid_from_yolo_notch if notch else module.estimate_grid_from_yolo
            common = dict(detection_format="point", normalized=False, refine_mode="cross_lines")
            if notch:
                common.update(notch_correction_angle_deg=0.0, notch_confidence=1.0)
            with self.subTest(module=name):
                corrected = estimate(image, detections, **common)
                self.assertTrue(np.allclose(corrected.points_clip, truth, atol=0.5))
                self.assertEqual(corrected.raw_points_clip, tuple(map(tuple, detections)))
                self.assertEqual(corrected.refinement_mode, "cross_lines")
                self.assertTrue(all(0.15 <= c < 0.9 for c in corrected.refinement_confidences))
                rejected = estimate(image, detections, refine_min_confidence=0.9, **common)
                self.assertEqual(rejected.points_clip, tuple(map(tuple, detections)))
                self.assertEqual(rejected.refinement_confidences, corrected.refinement_confidences)
                bypassed = estimate(image, detections, refine=False, **common)
                self.assertEqual(bypassed.points_clip, tuple(map(tuple, detections)))
                self.assertEqual(bypassed.refinement_mode, "none")

    def test_public_map_builder_uses_corrected_origin_and_retains_lookup(self):
        clip, truth, detections = white_corner_grid()
        wafer = np.zeros((800, 800, 3), np.uint8)
        cv2.circle(wafer, (400, 400), 340, (30, 150, 45), -1)
        wafer[200:601, 200:601] = clip
        dm = base.build_die_map_from_yolo(
            wafer, clip, detections, clip_origin=(200, 200), detection_format="point",
            normalized=False, refine=True, refine_mode="cross_lines", return_aligned_image=False,
        )
        self.assertTrue(np.allclose(dm.grid_estimate.points_clip, truth, atol=0.5))
        self.assertTrue(np.allclose((dm.x0, dm.y0), (400.0, 400.0), atol=0.5))
        self.assertAlmostEqual(dm.pitch_x, 100.0, places=4)
        self.assertAlmostEqual(dm.pitch_y, 100.0, places=4)
        self.assertEqual(len(base.locate_die(dm, point=(450, 350), return_all=True)), 1)


if __name__ == "__main__":
    unittest.main()
