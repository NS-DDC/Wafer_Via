import runpy
import tempfile
import unittest
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np

from codex.wafer_notch_angle import (
    align_wafer_by_notch,
    detect_wafer_notch,
)
from codex.wafer_via_notch import build_die_map_from_yolo


def synthetic_notched_wafer(size=900):
    center = (450, 450)
    radius = 390
    image = np.zeros((size, size, 3), np.uint8)
    cv2.circle(image, center, radius, (110, 130, 145), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (185, 195, 205), 5, cv2.LINE_AA)
    notch = np.asarray(
        [
            (center[0] - 22, center[1] + radius + 24),
            (center[0], center[1] + radius - 15),
            (center[0] + 22, center[1] + radius + 24),
        ],
        np.int32,
    )
    cv2.fillConvexPoly(image, notch, (0, 0, 0), cv2.LINE_AA)
    return image, center


def synthetic_wide_shallow_notch(size=900):
    center = (450, 450)
    radius = 390
    image = np.zeros((size, size, 3), np.uint8)
    cv2.circle(image, center, radius, (110, 130, 145), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (185, 195, 205), 5, cv2.LINE_AA)
    # A long, shallow semicircular black intrusion rather than a sharp V.
    cv2.ellipse(
        image,
        (center[0], center[1] + radius + 5),
        (70, 12),
        0,
        0,
        360,
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )
    return image, center


def synthetic_nonblack_variable_background_notch(size=900):
    center = (472, 432)
    radius = 365
    yy, xx = np.indices((size, size), dtype=np.float32)
    background = np.dstack((
        65.0 + 55.0 * xx / size,
        82.0 + 45.0 * yy / size,
        105.0 + 18.0 * np.sin(xx / 95.0),
    )).clip(0, 255).astype(np.uint8)
    image = background.copy()
    cv2.circle(image, center, radius, (118, 137, 151), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (150, 161, 171), 4, cv2.LINE_AA)
    notch_mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(
        notch_mask,
        (center[0], center[1] + radius + 4),
        (58, 11),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    alpha = notch_mask.astype(np.float32)[:, :, None] / 255.0
    image = np.rint(image * (1.0 - alpha) + background * alpha).astype(np.uint8)
    return image, center, radius


class WaferNotchAngleTests(unittest.TestCase):
    def test_standalone_notch_pipeline_has_no_local_import(self):
        source = (
            Path(__file__).parents[1]
            / "codex"
            / "wafer_via_notch_standalone.py"
        )
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("import wafer_via", text)
        self.assertNotIn("import wafer_notch_angle", text)
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / source.name
            copy2(source, isolated)
            namespace = runpy.run_path(str(isolated))
        self.assertTrue(callable(namespace["build_die_map_from_yolo"]))
        self.assertTrue(callable(namespace["detect_wafer_notch"]))
        image, center = synthetic_notched_wafer()
        clip = image[194:706, 194:706]
        detections = np.asarray(
            [
                (center[0] + ix * 70.0 - 194, center[1] + iy * 82.0 - 194)
                for iy in range(-2, 3)
                for ix in range(-2, 3)
            ]
        )
        die_map = namespace["build_die_map_from_yolo"](
            image,
            clip,
            detections,
            detection_format="point",
            clip_origin=(194, 194),
            refine=False,
            return_aligned_image=False,
        )
        self.assertEqual(die_map.angle_align_method, "notch")
        self.assertLess(abs(die_map.grid_angle_deg), 0.3)

    def test_bottom_notch_is_zero_correction(self):
        image, _ = synthetic_notched_wafer()
        result = detect_wafer_notch(image)

        self.assertTrue(result.found)
        self.assertLess(abs(result.notch_angle_deg - 90.0), 0.3)
        self.assertLess(abs(result.correction_angle_deg), 0.3)
        self.assertGreater(result.notch_depth_px, 10.0)
        outer_radius = np.linalg.norm(
            np.asarray(result.notch_point_px) - np.asarray(result.wafer_center_px)
        )
        deepest_radius = np.linalg.norm(
            np.asarray(result.notch_deepest_point_px)
            - np.asarray(result.wafer_center_px)
        )
        self.assertAlmostEqual(outer_radius, result.wafer_radius_px, places=3)
        self.assertLess(deepest_radius, outer_radius)

    def test_rotated_notch_recovers_opposite_correction(self):
        image, center = synthetic_notched_wafer()
        applied_rotation = 17.0
        matrix = cv2.getRotationMatrix2D(center, applied_rotation, 1.0)
        rotated = cv2.warpAffine(image, matrix, image.shape[1::-1])

        result = detect_wafer_notch(rotated)
        self.assertLess(
            abs(result.correction_angle_deg + applied_rotation), 0.5
        )

        aligned, _, _, _ = align_wafer_by_notch(rotated, result)
        residual = detect_wafer_notch(aligned)
        self.assertLess(abs(residual.correction_angle_deg), 0.5)

    def test_wide_shallow_semicircle_notch_is_detected(self):
        image, center = synthetic_wide_shallow_notch()
        result = detect_wafer_notch(image)

        self.assertTrue(result.found)
        self.assertLess(abs(result.notch_angle_deg - 90.0), 0.5)
        self.assertGreater(result.notch_width_deg, 8.0)
        self.assertGreater(result.notch_depth_px, 2.0)

        matrix = cv2.getRotationMatrix2D(center, -23.0, 1.0)
        rotated = cv2.warpAffine(image, matrix, image.shape[1::-1])
        rotated_result = detect_wafer_notch(rotated)
        self.assertLess(abs(rotated_result.correction_angle_deg - 23.0), 0.6)

    def test_nonblack_variable_background_and_offset_center(self):
        image, center, radius = synthetic_nonblack_variable_background_notch()
        result = detect_wafer_notch(image)

        self.assertTrue(result.found)
        self.assertEqual(result.detection_method, "geometry_edge_bottom_sector")
        self.assertLess(abs(result.notch_angle_deg - 90.0), 0.8)
        self.assertLess(np.linalg.norm(np.asarray(result.wafer_center_px) - center), 4.0)
        self.assertLess(abs(result.wafer_radius_px - radius), 8.0)
        self.assertGreater(result.notch_depth_px, 3.0)

    def test_circle_without_notch_is_rejected(self):
        image, center = synthetic_notched_wafer()
        cv2.circle(image, center, 390, (110, 130, 145), -1, cv2.LINE_AA)
        cv2.circle(image, center, 390, (185, 195, 205), 5, cv2.LINE_AA)

        with self.assertRaises(RuntimeError):
            detect_wafer_notch(image)
        result = detect_wafer_notch(image, failure_mode="zero")
        self.assertFalse(result.found)
        self.assertEqual(result.correction_angle_deg, 0.0)
        self.assertEqual(result.failure_mode, "zero")
        with self.assertRaises(ValueError):
            detect_wafer_notch(image, failure_mode="invalid")

    def test_die_map_zero_policy_returns_zero_angle_when_notch_is_missing(self):
        image, center = synthetic_notched_wafer()
        cv2.circle(image, center, 390, (110, 130, 145), -1, cv2.LINE_AA)
        cv2.circle(image, center, 390, (185, 195, 205), 5, cv2.LINE_AA)
        clip = image[194:706, 194:706]
        detections = np.asarray(
            [
                (center[0] + ix * 70.0 - 194, center[1] + iy * 82.0 - 194)
                for iy in range(-2, 3)
                for ix in range(-2, 3)
            ]
        )

        die_map = build_die_map_from_yolo(
            image,
            clip,
            detections,
            detection_format="point",
            clip_origin=(194, 194),
            refine=False,
            notch_failure_mode="zero",
            return_aligned_image=False,
        )

        self.assertEqual(die_map.grid_angle_deg, 0.0)
        self.assertEqual(die_map.angle_align_method, "notch_zero_fallback")
        self.assertFalse(die_map.notch_result.found)

    def test_die_map_uses_notch_as_only_angle_source(self):
        image, center = synthetic_notched_wafer()
        applied_rotation = 17.0
        matrix = cv2.getRotationMatrix2D(center, applied_rotation, 1.0)
        rotated = cv2.warpAffine(image, matrix, image.shape[1::-1])
        clip_origin = (194, 194)
        clip = rotated[194:706, 194:706]
        pitch_x, pitch_y = 70.0, 82.0
        detections = []
        for iy in range(-2, 3):
            for ix in range(-2, 3):
                point = matrix @ np.asarray(
                    (center[0] + ix * pitch_x, center[1] + iy * pitch_y, 1.0)
                )
                detections.append(
                    (point[0] - clip_origin[0], point[1] - clip_origin[1])
                )

        die_map = build_die_map_from_yolo(
            rotated,
            clip,
            np.asarray(detections),
            detection_format="point",
            clip_origin=clip_origin,
            refine=False,
            return_aligned_image=True,
        )

        self.assertEqual(die_map.angle_align_method, "notch")
        self.assertLess(abs(die_map.grid_angle_deg + applied_rotation), 0.5)
        self.assertAlmostEqual(die_map.pitch_x, pitch_x, places=4)
        self.assertAlmostEqual(die_map.pitch_y, pitch_y, places=4)
        self.assertEqual(die_map.angle_pairs_full, ())
        self.assertEqual(die_map.grid_estimate.angle_candidate_count, 0)
        self.assertIsNotNone(die_map.aligned_image)
        residual = detect_wafer_notch(die_map.aligned_image)
        self.assertLess(abs(residual.correction_angle_deg), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
