import importlib.util
import inspect
import sys
import unittest
from unittest import mock
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


def synthetic_wide_shallow_notch(width_px, height_px, background_bgr):
    size = 1400
    center = np.asarray((700.0, 700.0))
    radius = 620.0
    image = np.full((size, size, 3), background_bgr, np.uint8)
    cv2.circle(
        image, tuple(center.astype(int)), int(radius),
        (120, 140, 160), -1, cv2.LINE_AA
    )
    cv2.circle(
        image, tuple(center.astype(int)), int(radius),
        (190, 200, 210), 7, cv2.LINE_AA
    )
    half_width = float(width_px) * 0.5
    x_values = np.linspace(-half_width, half_width, int(round(width_px)) + 1)
    normal_y = center[1] + np.sqrt(
        np.maximum(0.0, radius * radius - x_values * x_values)
    )
    boundary_y = normal_y - float(height_px) * np.sqrt(
        np.maximum(0.0, 1.0 - (x_values / half_width) ** 2)
    )
    polygon = np.column_stack((center[0] + x_values, boundary_y))
    polygon = np.vstack((
        polygon,
        (center[0] + half_width, size - 1),
        (center[0] - half_width, size - 1),
    ))
    cv2.fillPoly(
        image, [np.rint(polygon).astype(np.int32)], background_bgr, cv2.LINE_AA
    )
    return image, center, (center[0], center[1] + radius)


def synthetic_noisy_wide_shallow_notch(
    width_px, height_px, background_bgr, seed, noise_sigma
):
    size = 1400
    center = np.asarray((700.0, 700.0))
    radius = 620.0
    rng = np.random.default_rng(seed)
    base = np.asarray(background_bgr, dtype=np.float32)
    background = np.broadcast_to(base, (size, size, 3)).copy()
    x_gradient = np.linspace(-8.0, 8.0, size, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(5.0, -5.0, size, dtype=np.float32)[:, None, None]
    band = 3.0 * np.sin(
        np.linspace(0.0, 14.0 * np.pi, size, dtype=np.float32)
    )[None, :, None]
    low = rng.normal(0.0, 1.0, (20, 20)).astype(np.float32)
    low = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)[:, :, None]
    gaussian = rng.normal(0.0, noise_sigma, (size, size, 1)).astype(np.float32)
    background += x_gradient + y_gradient + band + low * 4.0 + gaussian
    background = np.clip(np.rint(background), 0, 255).astype(np.uint8)
    for _ in range(18):
        location = tuple(int(value) for value in rng.integers(0, size, size=2))
        colour = np.clip(
            base.astype(np.int16) + rng.integers(-12, 13, size=3), 0, 255
        )
        cv2.circle(
            background,
            location,
            int(rng.integers(4, 20)),
            tuple(int(value) for value in colour),
            -1,
            cv2.LINE_AA,
        )

    image = background.copy()
    cv2.circle(
        image, tuple(center.astype(int)), int(radius),
        (120, 140, 160), -1, cv2.LINE_AA
    )
    cv2.circle(
        image, tuple(center.astype(int)), int(radius),
        (190, 200, 210), 7, cv2.LINE_AA
    )
    half_width = float(width_px) * 0.5
    x_values = np.linspace(-half_width, half_width, int(round(width_px)) + 1)
    normal_y = center[1] + np.sqrt(
        np.maximum(0.0, radius * radius - x_values * x_values)
    )
    boundary_y = normal_y - float(height_px) * np.sqrt(
        np.maximum(0.0, 1.0 - (x_values / half_width) ** 2)
    )
    polygon = np.column_stack((center[0] + x_values, boundary_y))
    polygon = np.vstack((
        polygon,
        (center[0] + half_width, size - 1),
        (center[0] - half_width, size - 1),
    ))
    notch_mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(
        notch_mask, [np.rint(polygon).astype(np.int32)], 255, cv2.LINE_AA
    )
    image[notch_mask != 0] = background[notch_mask != 0]
    return image, center, (center[0], center[1] + radius)


class RoiSemicircleStandaloneTests(unittest.TestCase):
    def test_detect_reuses_a_single_bgr_to_lab_conversion(self):
        image, _, roi_center = synthetic_wide_shallow_notch(
            108.0, 38.0, (185, 185, 185)
        )
        original_cvt_color = MODULE.cv2.cvtColor
        bgr_to_lab_calls = 0

        def counting_cvt_color(source, code, *args, **kwargs):
            nonlocal bgr_to_lab_calls
            if code == cv2.COLOR_BGR2LAB:
                bgr_to_lab_calls += 1
            return original_cvt_color(source, code, *args, **kwargs)

        with mock.patch.object(MODULE.cv2, "cvtColor", side_effect=counting_cvt_color):
            result = MODULE.detect_wafer_notch(
                image,
                notch_roi_center_px=roi_center,
                notch_roi_half_size_px=(140, 90),
                notch_semicircle_radius_range_px=(35, 80),
                notch_background_palette_size=5,
                notch_background_morph_px=4,
                failure_mode="error",
            )

        self.assertTrue(result.found)
        self.assertEqual(bgr_to_lab_calls, 1)

    def test_roi_semicircle_supplies_opposite_rotation_correction(self):
        image, _, roi_center = synthetic_semicircle_notch(rotation_deg=13.0)
        result = MODULE.detect_wafer_notch(
            image,
            notch_roi_center_px=tuple(roi_center),
            notch_roi_half_size_px=(110, 90),
            notch_semicircle_radius_range_px=(15, 45),
        )
        self.assertTrue(result.found)
        self.assertEqual(result.detection_method, "roi_background_connected_notch_arc")
        self.assertLess(abs(result.correction_angle_deg + 13.0), 0.6)
        self.assertLess(result.semicircle_fit_residual_px, 3.0)
        self.assertTrue(result.background_segmentation_used)

    def test_wide_shallow_semiellipse_sizes_and_backgrounds(self):
        cases = (
            (105.0, 36.0, (0, 0, 0)),
            (106.0, 37.0, (185, 185, 185)),
            (108.0, 38.0, (220, 239, 219)),
            (110.0, 40.0, (211, 210, 244)),
        )
        for width, height, background in cases:
            with self.subTest(width=width, height=height, background=background):
                image, center, roi_center = synthetic_wide_shallow_notch(
                    width, height, background
                )
                result = MODULE.detect_wafer_notch(
                    image,
                    notch_roi_center_px=roi_center,
                    notch_roi_half_size_px=(140, 90),
                    notch_semicircle_radius_range_px=(35, 80),
                    notch_background_morph_px=4,
                    failure_mode="error",
                )
                self.assertTrue(result.found)
                self.assertEqual(result.semicircle_shape, "semiellipse")
                self.assertLess(abs(result.notch_angle_deg - 90.0), 0.5)
                self.assertLess(abs(result.notch_width_px - width), 8.0)
                self.assertLess(abs(result.notch_depth_px - height), 6.0)
                self.assertLess(
                    np.linalg.norm(np.asarray(result.wafer_center_px) - center), 4.0
                )

    def test_wide_shallow_notch_with_varied_noisy_backgrounds(self):
        cases = (
            ((24, 27, 31), 105.0, 36.0, 4.0),
            ((150, 158, 166), 106.0, 37.0, 3.0),
            ((178, 202, 218), 108.0, 38.0, 4.0),
            ((218, 196, 166), 110.0, 40.0, 5.0),
            ((154, 190, 158), 105.0, 36.0, 5.0),
            ((208, 177, 203), 106.0, 37.0, 6.0),
        )
        for index, (background, width, height, noise) in enumerate(cases):
            with self.subTest(background=background, noise=noise):
                image, center, roi_center = synthetic_noisy_wide_shallow_notch(
                    width, height, background, 7000 + index, noise
                )
                result = MODULE.detect_wafer_notch(
                    image,
                    notch_roi_center_px=roi_center,
                    notch_roi_half_size_px=(150, 110),
                    notch_semicircle_radius_range_px=(35, 80),
                    notch_background_palette_size=5,
                    notch_background_morph_px=4,
                    failure_mode="error",
                )
                self.assertTrue(result.found)
                self.assertLess(abs(result.notch_angle_deg - 90.0), 0.6)
                self.assertLess(abs(result.notch_width_px - width), 10.0)
                self.assertLess(abs(result.notch_depth_px - height), 7.0)
                self.assertLess(
                    np.linalg.norm(np.asarray(result.wafer_center_px) - center), 6.0
                )

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

    def test_isolated_inner_semicircle_is_not_used_as_notch(self):
        image, _, roi_center = synthetic_wide_shallow_notch(
            108.0, 38.0, (185, 185, 185)
        )
        # Deliberately add a larger semicircular decoy inside the wafer.  It is
        # not connected to the image-border background and must be ignored.
        cv2.ellipse(
            image,
            (700, 1195),
            (95, 45),
            0.0,
            180.0,
            360.0,
            (185, 185, 185),
            12,
            cv2.LINE_AA,
        )
        result = MODULE.detect_wafer_notch(
            image,
            notch_roi_center_px=roi_center,
            notch_roi_half_size_px=(150, 150),
            notch_semicircle_radius_range_px=(35, 80),
            notch_background_morph_px=4,
            failure_mode="error",
        )
        self.assertTrue(result.found)
        self.assertLess(abs(result.notch_angle_deg - 90.0), 0.5)
        self.assertLess(abs(result.notch_width_px - 108.0), 8.0)
        self.assertLess(abs(result.notch_depth_px - 38.0), 6.0)

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
        self.assertIsNotNone(dm.notch_overlay_image)
        self.assertEqual(dm.notch_overlay_image.shape, image.shape)
        self.assertIsNotNone(dm.notch_semicircle_center_px)
        self.assertGreater(dm.notch_semicircle_score, 0.55)

    def test_builder_defaults_angle_result_image_to_5000_limit(self):
        parameter = inspect.signature(
            MODULE.build_die_map_from_yolo
        ).parameters["notch_visual_max_dimension"]
        self.assertEqual(parameter.default, 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
