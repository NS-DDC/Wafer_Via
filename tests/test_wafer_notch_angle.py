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
    draw_aligned_wafer_notch_guide,
    make_notch_overlay,
    make_notch_zoom,
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
        self.assertNotIn("def estimate_grid_from_yolo(", text)
        self.assertNotIn("_estimate_grid_orientation", text)
        self.assertNotIn("_legacy_build_die_map_from_yolo", text)
        self.assertNotIn("robust_angle_deg", text)
        self.assertNotIn("local_angle_deg", text)
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / source.name
            copy2(source, isolated)
            namespace = runpy.run_path(str(isolated))
        self.assertTrue(callable(namespace["build_die_map_from_yolo"]))
        self.assertTrue(callable(namespace["detect_wafer_notch"]))
        self.assertTrue(callable(namespace["draw_aligned_wafer_notch_guide"]))
        image, center = synthetic_notched_wafer()
        standalone_guide = namespace["draw_aligned_wafer_notch_guide"](image)
        self.assertTrue(standalone_guide.found)
        self.assertEqual(standalone_guide.overlay_image.shape, image.shape)
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
        self.assertEqual(die_map.coordinate_space, "aligned_image")
        self.assertIsNone(die_map.notch_overlay_image)
        self.assertIsNone(die_map.notch_zoom_image)

    def test_aligned_v5_guide_returns_writable_full_resolution_overlay(self):
        image, center = synthetic_notched_wafer()
        guide = draw_aligned_wafer_notch_guide(image)

        self.assertTrue(guide.found)
        self.assertEqual(guide.detection_method, "v5_silhouette_radial_aligned")
        self.assertEqual(guide.overlay_image.shape, image.shape)
        self.assertTrue(guide.overlay_image.flags.writeable)
        self.assertLess(
            np.linalg.norm(np.asarray(guide.wafer_center_px) - center), 3.0
        )
        self.assertLess(abs(guide.residual_angle_deg), 0.5)
        self.assertIsNotNone(guide.notch_center_px)
        self.assertIsNotNone(guide.notch_point_px)
        self.assertIsNotNone(guide.notch_left_px)
        self.assertIsNotNone(guide.notch_right_px)
        outer_radius = np.linalg.norm(
            np.asarray(guide.notch_point_px) - np.asarray(guide.wafer_center_px)
        )
        self.assertAlmostEqual(outer_radius, guide.wafer_radius_px, places=4)
        self.assertGreater(np.count_nonzero(guide.overlay_image != image), 0)

        # The caller can draw ground truth directly on the returned image.
        cv2.circle(guide.overlay_image, (123, 234), 8, (255, 0, 0), -1)
        self.assertTrue(np.array_equal(guide.overlay_image[234, 123], (255, 0, 0)))

    def test_aligned_v5_guide_zero_mode_keeps_ring_without_notch(self):
        image, center = synthetic_notched_wafer()
        cv2.circle(image, center, 390, (110, 130, 145), -1, cv2.LINE_AA)
        guide = draw_aligned_wafer_notch_guide(image, failure_mode="zero")

        self.assertFalse(guide.found)
        self.assertIsNone(guide.notch_point_px)
        self.assertIsNone(guide.notch_center_px)
        self.assertEqual(guide.residual_angle_deg, 0.0)
        self.assertEqual(guide.overlay_image.shape, image.shape)

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
        overview = make_notch_overlay(image, result, max_dimension=300)
        zoom = make_notch_zoom(image, result, size_px=120, scale=2.0)
        self.assertEqual(max(overview.shape[:2]), 300)
        self.assertLessEqual(max(zoom.shape[:2]), 480)

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
            return_notch_visuals=False,
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
            return_notch_visuals=True,
        )

        self.assertEqual(die_map.angle_align_method, "notch")
        self.assertEqual(die_map.coordinate_space, "aligned_image")
        self.assertAlmostEqual(die_map.grid_angle_deg, 0.0, places=8)
        self.assertLess(abs(die_map.image_rotation_deg + applied_rotation), 0.5)
        self.assertLess(abs(die_map.source_grid_angle_deg + applied_rotation), 0.5)
        self.assertAlmostEqual(die_map.pitch_x, pitch_x, places=4)
        self.assertAlmostEqual(die_map.pitch_y, pitch_y, places=4)
        self.assertEqual(die_map.angle_pairs_full, ())
        self.assertEqual(die_map.grid_estimate.angle_candidate_count, 0)
        self.assertIsNotNone(die_map.aligned_image)
        self.assertIsNotNone(die_map.notch_overlay_image)
        self.assertIsNotNone(die_map.notch_zoom_image)
        self.assertEqual(die_map.notch_overlay_coordinate_space, "original_image")
        self.assertEqual(die_map.notch_overlay_image.shape, rotated.shape)
        self.assertLessEqual(max(die_map.notch_zoom_image.shape[:2]), 1024)
        aligned_origin = die_map.original_to_aligned_matrix @ np.asarray(
            (die_map.source_x0, die_map.source_y0, 1.0)
        )
        self.assertTrue(np.allclose(aligned_origin, (die_map.x0, die_map.y0)))
        aligned_x_vector = (
            np.asarray(die_map.pitch_x_points_full[1])
            - np.asarray(die_map.pitch_x_points_full[0])
        )
        aligned_y_vector = (
            np.asarray(die_map.pitch_y_points_full[1])
            - np.asarray(die_map.pitch_y_points_full[0])
        )
        self.assertLess(abs(aligned_x_vector[1]), 1e-5)
        self.assertLess(abs(aligned_y_vector[0]), 1e-5)
        residual = detect_wafer_notch(die_map.aligned_image)
        self.assertLess(abs(residual.correction_angle_deg), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
