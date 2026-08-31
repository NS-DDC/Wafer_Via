import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "codex" / "wafer_via_notch_standalone.py"
)
SPEC = importlib.util.spec_from_file_location("roi_semicircle_standalone", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_semicircle_notch(rotation_deg=0.0):
    size = 900
    center = np.asarray((450.0, 450.0))
    radius = 390
    notch_center = np.asarray((450.0, 840.0))
    image = np.zeros((size, size, 3), np.uint8)
    cv2.circle(image, tuple(center.astype(int)), radius, (120, 140, 160), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(center.astype(int)), radius, (190, 200, 210), 5, cv2.LINE_AA)
    cv2.circle(image, tuple(notch_center.astype(int)), 30, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.rectangle(image, (410, 840), (490, size - 1), (0, 0, 0), -1)
    if not rotation_deg:
        return image, center, notch_center
    matrix = cv2.getRotationMatrix2D(tuple(center), float(rotation_deg), 1.0)
    rotated = cv2.warpAffine(image, matrix, (size, size))
    rotated_notch = matrix @ np.asarray((*notch_center, 1.0))
    return rotated, center, rotated_notch


def synthetic_noisy_exterior_semicircle_notch():
    size = 900
    center = np.asarray((450.0, 450.0))
    radius = 390
    notch_center = np.asarray((450.0, 840.0))
    rng = np.random.default_rng(1907)
    background = np.full((size, size, 3), (72, 86, 105), np.uint8)
    for _ in range(260):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        colour = tuple(int(value) for value in rng.integers(45, 145, size=3))
        cv2.circle(background, (x, y), int(rng.integers(2, 10)), colour, -1)
    image = background.copy()
    cv2.circle(image, tuple(center.astype(int)), radius, (120, 140, 160), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(center.astype(int)), radius, (190, 200, 210), 5, cv2.LINE_AA)
    notch_mask = np.zeros((size, size), np.uint8)
    cv2.circle(notch_mask, tuple(notch_center.astype(int)), 30, 255, -1, cv2.LINE_AA)
    cv2.rectangle(notch_mask, (410, 840), (490, size - 1), 255, -1)
    image[notch_mask > 0] = background[notch_mask > 0]
    return image, center, notch_center


class RoiSemicircleStandaloneTests(unittest.TestCase):
    def test_roi_semicircle_supplies_opposite_rotation_correction(self):
        image, _, roi_center = synthetic_semicircle_notch(rotation_deg=13.0)
        result = MODULE.detect_wafer_notch(
            image,
            notch_roi_center_px=tuple(roi_center),
            notch_roi_half_size_px=(110, 90),
            notch_semicircle_radius_range_px=(15, 45),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.detection_method, "roi_background_connected_semicircle")
        self.assertLess(abs(result.correction_angle_deg + 13.0), 0.6)
        self.assertLess(result.semicircle_fit_residual_px, 3.0)
        self.assertTrue(result.background_segmentation_used)

    def test_roi_background_rejects_noisy_patterns_outside_wafer(self):
        image, center, roi_center = synthetic_noisy_exterior_semicircle_notch()
        result = MODULE.detect_wafer_notch(
            image,
            notch_roi_center_px=tuple(roi_center),
            notch_roi_half_size_px=(110, 90),
            notch_semicircle_radius_range_px=(15, 45),
            failure_mode="error",
        )
        self.assertTrue(result.found)
        self.assertLess(abs(result.notch_angle_deg - 90.0), 0.8)
        self.assertLess(
            np.linalg.norm(np.asarray(result.wafer_center_px) - center), 5.0
        )
        self.assertLess(abs(result.wafer_radius_px - 390.0), 8.0)
        self.assertLess(result.semicircle_fit_residual_px, 4.0)

    def test_roi_mode_rejects_a_circle_without_notch(self):
        image, center, _ = synthetic_semicircle_notch()
        cv2.circle(image, tuple(center.astype(int)), 390, (120, 140, 160), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(center.astype(int)), 390, (190, 200, 210), 5, cv2.LINE_AA)
        result = MODULE.detect_wafer_notch(
            image,
            notch_roi_center_px=(450, 840),
            notch_roi_half_size_px=(110, 90),
            notch_semicircle_radius_range_px=(15, 45),
            failure_mode="zero",
        )
        self.assertFalse(result.found)
        self.assertEqual(result.correction_angle_deg, 0.0)

    def test_builder_rotates_image_and_returns_aligned_die_map(self):
        image, center, roi_center = synthetic_semicircle_notch()
        clip_origin = (194, 194)
        clip = image[194:706, 194:706]
        detections = np.asarray([
            (
                center[0] + ix * 70.0 - clip_origin[0],
                center[1] + iy * 82.0 - clip_origin[1],
            )
            for iy in range(-2, 3)
            for ix in range(-2, 3)
        ])
        dm = MODULE.build_die_map_from_yolo(
            image,
            clip,
            detections,
            detection_format="point",
            clip_origin=clip_origin,
            refine=False,
            notch_roi_center_px=tuple(roi_center),
            notch_roi_half_size_px=(110, 90),
            notch_semicircle_radius_range_px=(15, 45),
            return_aligned_image=True,
        )
        self.assertEqual(dm.angle_align_method, "notch")
        self.assertEqual(dm.coordinate_space, "aligned_image")
        self.assertEqual(dm.grid_angle_deg, 0.0)
        self.assertIsNotNone(dm.aligned_image)
        self.assertIsNotNone(dm.notch_semicircle_center_px)
        self.assertGreater(dm.notch_semicircle_score, 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
