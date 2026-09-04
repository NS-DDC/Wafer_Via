"""Shape-free fallback geometry: synthetic evidence, not camera accuracy."""

import math
import unittest

import cv2
import numpy as np

from codex import wafer_notch_angle as notch
from codex import wafer_via_notch as pipeline


def irregular_fixture(shape="jagged", rotation=0.0, background=(180, 190, 200)):
    image = np.full((900, 900, 3), background, np.uint8)
    cv2.circle(image, (450, 450), 390, (90, 140, 110), -1)
    profiles = {
        "jagged": [(-44, 0), (-41, 18), (-33, 31), (-25, 23), (-18, 38),
                   (-10, 24), (0, 25), (12, 21), (24, 30), (33, 18), (41, 15), (44, 0)],
        "step": [(-44, 0), (-43, 20), (-21, 20), (-20, 40), (2, 40),
                 (3, 20), (43, 20), (44, 0)],
        "rectangle": [(-44, 0), (-43, 30), (43, 30), (44, 0)],
        "double": [(-72, 0), (-70, 30), (-40, 30), (-38, 0),
                   (38, 0), (40, 30), (70, 30), (72, 0)],
        "none": [],
    }
    points = profiles[shape]
    if points:
        xy = np.asarray([(450 + x, 450 + math.sqrt(390 ** 2 - x ** 2) - depth)
                         for x, depth in points])
        xy = np.vstack((xy, (450 + points[-1][0], 899), (450 + points[0][0], 899)))
        cv2.fillPoly(image, [np.rint(xy).astype(np.int32)], background)
    matrix = cv2.getRotationMatrix2D((450, 450), rotation, 1.0)
    if rotation:
        image = cv2.warpAffine(image, matrix, (900, 900), borderValue=background)
    roi = matrix @ np.asarray((450.0, 840.0, 1.0))
    return image, tuple(roi)


def irregular_options(roi, **overrides):
    options = dict(
        notch_roi_center_px=roi, notch_roi_half_size_px=(130, 90),
        notch_background_morph_px=2, failure_mode="zero",
    )
    options.update(overrides)
    return options


class RimGeometryTests(unittest.TestCase):
    def test_chipped_shape_recovers_without_mocking_primary(self):
        image, roi = irregular_fixture("step")
        original = notch.detect_wafer_notch(image, **irregular_options(roi, notch_fallback_mode="none"))
        recovered = notch.detect_wafer_notch(image, **irregular_options(roi))
        self.assertFalse(original.found)
        self.assertTrue(recovered.fallback_used, recovered.fallback_reason)
        self.assertLess(abs(recovered.notch_angle_deg - 90.0), 0.4)
        self.assertIsNone(recovered.semicircle_center_px)
        self.assertEqual(recovered.semicircle_shape, "none")
        depth_from_point = recovered.wafer_radius_px - np.linalg.norm(
            np.asarray(recovered.notch_deepest_point_px) - recovered.wafer_center_px
        )
        self.assertAlmostEqual(depth_from_point, recovered.notch_depth_px, places=6)

    def test_rotation_downscale_and_angle_wrap(self):
        for rotation in (13.0, -17.0, 90.0):
            for dimension in (900, 600):
                with self.subTest(rotation=rotation, dimension=dimension):
                    image, roi = irregular_fixture("step", rotation)
                    result = notch.detect_wafer_notch(
                        image, **irregular_options(roi, max_dimension=dimension)
                    )
                    self.assertTrue(result.fallback_used, result.fallback_reason)
                    error = (result.notch_angle_deg - (90.0 - rotation) + 180.0) % 360.0 - 180.0
                    self.assertLess(abs(error), 0.4)
                    self.assertLess(abs(result.correction_angle_deg + rotation), 0.4)

    def test_ambiguous_or_missing_mouth_is_not_guessed(self):
        for shape, roi_size in (("double", (130, 90)), ("none", (130, 90)), ("step", (30, 90))):
            with self.subTest(shape=shape, roi_size=roi_size):
                image, roi = irregular_fixture(shape)
                result = notch.detect_wafer_notch(
                    image, **irregular_options(roi, notch_roi_half_size_px=roi_size)
                )
                self.assertFalse(result.found)
                self.assertTrue(result.fallback_attempted)
                self.assertFalse(result.fallback_used)
                self.assertEqual(result.correction_angle_deg, 0.0)
                self.assertEqual(result.candidate_arc_px, ())
                if shape == "double":
                    self.assertEqual(result.fallback_reason, "ambiguous_multiple_intrusions")

    def test_varied_backgrounds_and_width_constraint(self):
        for colour in ((0, 0, 0), (130, 100, 140), (230, 210, 180)):
            with self.subTest(colour=colour):
                image, roi = irregular_fixture("step", background=colour)
                result = notch.detect_wafer_notch(image, **irregular_options(roi))
                self.assertTrue(result.fallback_used, result.fallback_reason)
                self.assertLess(abs(result.notch_angle_deg - 90.0), 0.4)
        result = notch.detect_wafer_notch(
            image, **irregular_options(roi, notch_semicircle_radius_range_px=(10, 20))
        )
        self.assertFalse(result.found)
        self.assertEqual(result.fallback_reason, "intrusion_width_outside_radius_range")

    def test_successful_irregular_primary_is_not_overridden(self):
        image, roi = irregular_fixture("jagged")
        original = notch.detect_wafer_notch(image, **irregular_options(roi, notch_fallback_mode="none"))
        result = notch.detect_wafer_notch(image, **irregular_options(roi))
        self.assertTrue(original.found)
        self.assertFalse(result.fallback_attempted)
        self.assertEqual(result.notch_point_px, original.notch_point_px)
        self.assertEqual(result.notch_angle_deg, original.notch_angle_deg)

    def test_fallback_rotates_image_and_xywh_grid_together(self):
        image, roi = irregular_fixture("step", 13.0)
        raw = np.asarray([(450 + ix * 70, 450 + iy * 82, 1)
                          for iy in range(-2, 3) for ix in range(-2, 3)])
        matrix = cv2.getRotationMatrix2D((450, 450), 13.0, 1.0)
        points_clip = raw @ matrix.T - (194, 194)
        boxes = np.column_stack((points_clip, np.full((len(raw), 2), 12.0)))
        options = irregular_options(roi)
        options.pop("failure_mode")
        dm = pipeline.build_die_map_from_yolo(
            image, image[194:706, 194:706], boxes,
            detection_format="xywh", clip_origin=(194, 194), refine=False,
            return_notch_visuals=True, **options,
        )
        self.assertTrue(dm.notch_fallback_used)
        self.assertLess(abs(dm.image_rotation_deg + 13.0), 0.4)
        self.assertEqual(dm.coordinate_space, "aligned_image")
        self.assertEqual(dm.grid_angle_deg, 0.0)
        self.assertEqual(dm.aligned_image.shape, image.shape)
        self.assertFalse(np.array_equal(dm.aligned_image, image))
        self.assertIsNotNone(dm.notch_overlay_image)
        self.assertIsNotNone(dm.notch_zoom_image)
        self.assertLess(abs(dm.pitch_x - 70.0), 0.1)
        self.assertLess(abs(dm.pitch_y - 82.0), 0.1)


if __name__ == "__main__":
    unittest.main()
