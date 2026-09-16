"""Regression contract for the failure-only ROI rim-intrusion fallback."""

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from codex import wafer_notch_angle as SOURCE
from codex import wafer_via_notch as PIPELINE


STANDALONE_PATH = (
    Path(__file__).parents[1] / "codex" / "wafer_via_notch_standalone.py"
)
SPEC = importlib.util.spec_from_file_location(
    "roi_fallback_standalone", STANDALONE_PATH
)
STANDALONE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STANDALONE
SPEC.loader.exec_module(STANDALONE)
ADAPTIVE_SPEC = importlib.util.spec_from_file_location(
    "roi_fallback_adaptive_standalone",
    STANDALONE_PATH.with_name("wafer_via_notch_adaptive_standalone.py"),
)
ADAPTIVE = importlib.util.module_from_spec(ADAPTIVE_SPEC)
sys.modules[ADAPTIVE_SPEC.name] = ADAPTIVE
ADAPTIVE_SPEC.loader.exec_module(ADAPTIVE)


def semicircle_fixture(notched=True):
    """A visible rim and known bottom opening, in full-resolution pixels."""
    image = np.zeros((900, 900, 3), np.uint8)
    cv2.circle(image, (450, 450), 390, (120, 140, 160), -1, cv2.LINE_AA)
    cv2.circle(image, (450, 450), 390, (190, 200, 210), 5, cv2.LINE_AA)
    if notched:
        cv2.circle(image, (450, 840), 30, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(image, (410, 840), (490, 899), (0, 0, 0), -1)
    return image


def roi_options(**overrides):
    values = dict(
        notch_roi_center_px=(450, 840),
        notch_roi_half_size_px=(110, 90),
        notch_semicircle_radius_range_px=(15, 45),
        notch_background_morph_px=4,
    )
    values.update(overrides)
    return values


class RoiRimIntrusionFallbackTests(unittest.TestCase):
    modules = (SOURCE, STANDALONE, ADAPTIVE)

    def assert_same_original_result(self, expected, actual):
        # New attempt/reason metadata may describe an explicit opt-out; all
        # pre-existing result values, arrays and points must stay identical.
        for field in dataclasses.fields(expected):
            if field.name.startswith("fallback_"):
                continue
            with self.subTest(field=field.name):
                left, right = getattr(expected, field.name), getattr(actual, field.name)
                if isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    self.assertEqual(left, right)

    def assert_outer_circle_midpoint(self, result):
        self.assertIsNotNone(result.notch_shoulder_points_px)
        center = np.asarray(result.wafer_center_px)
        shoulders = np.asarray(result.notch_shoulder_points_px)
        self.assertEqual(shoulders.shape, (2, 2))
        angles = np.unwrap(np.arctan2(
            shoulders[:, 1] - center[1], shoulders[:, 0] - center[0]
        ))
        midpoint = float(np.mean(angles))
        expected = center + result.wafer_radius_px * np.asarray(
            (np.cos(midpoint), np.sin(midpoint))
        )
        np.testing.assert_allclose(result.notch_point_px, expected, atol=1e-6)

    def test_accepted_original_result_never_calls_fallback(self):
        image = semicircle_fixture()
        for module in self.modules:
            with self.subTest(module=module.__name__):
                original = module.detect_wafer_notch(
                    image, **roi_options(notch_fallback_mode="none")
                )
                self.assertTrue(original.found)
                with mock.patch.object(
                    module, "_detect_roi_rim_intrusion",
                    side_effect=AssertionError("A successful original result must not retry"),
                ) as fallback:
                    result = module.detect_wafer_notch(image, **roi_options())
                fallback.assert_not_called()
                self.assertFalse(result.fallback_attempted)
                self.assertFalse(result.fallback_used)
                self.assert_same_original_result(original, result)

    def test_none_optout_does_not_call_fallback_after_shape_failure(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_fit_semicircle_from_background_boundary", return_value=None
            ), mock.patch.object(module, "_detect_roi_rim_intrusion") as fallback:
                result = module.detect_wafer_notch(
                    semicircle_fixture(),
                    **roi_options(notch_fallback_mode="none", failure_mode="zero"),
                )
                fallback.assert_not_called()
                self.assertFalse(result.found)
                self.assertFalse(result.fallback_attempted)
                self.assertFalse(result.fallback_used)
                self.assertEqual(result.correction_angle_deg, 0.0)

    def test_no_roi_does_not_enter_roi_fallback(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_detect_roi_rim_intrusion",
                side_effect=AssertionError("The fallback requires an explicit ROI"),
            ) as fallback:
                result = module.detect_wafer_notch(
                    semicircle_fixture(), failure_mode="zero"
                )
                fallback.assert_not_called()
                self.assertFalse(result.fallback_attempted)
                self.assertFalse(result.fallback_used)

    def test_edge_only_roi_does_not_enter_background_fallback(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_detect_semicircle_in_roi", return_value=None
            ), mock.patch.object(
                module, "_detect_roi_rim_intrusion",
                side_effect=AssertionError("Exterior-connected background is required"),
            ) as fallback:
                result = module.detect_wafer_notch(
                    semicircle_fixture(),
                    **roi_options(notch_use_roi_background=False, failure_mode="zero"),
                )
                fallback.assert_not_called()
                self.assertFalse(result.found)
                self.assertFalse(result.fallback_attempted)
                self.assertFalse(result.fallback_used)

    def test_missing_shape_candidate_uses_observed_rim_midpoint(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_fit_semicircle_from_background_boundary", return_value=None
            ), mock.patch.object(
                module, "_detect_roi_rim_intrusion", wraps=module._detect_roi_rim_intrusion
            ) as fallback:
                result = module.detect_wafer_notch(semicircle_fixture(), **roi_options())
                fallback.assert_called_once()
                self.assertTrue(result.found)
                self.assertTrue(result.fallback_attempted)
                self.assertTrue(result.fallback_used)
                self.assertEqual(
                    result.detection_method, "roi_background_rim_intrusion_fallback"
                )
                self.assertLess(abs(result.notch_angle_deg - 90.0), 0.5)
                self.assert_outer_circle_midpoint(result)

    def test_low_score_candidate_also_triggers_fallback(self):
        for module in self.modules:
            original_fit = module._fit_semicircle_from_background_boundary

            def rejected_candidate(*args, **kwargs):
                candidate = original_fit(*args, **kwargs)
                self.assertIsNotNone(candidate)
                return dataclasses.replace(candidate, score=0.0)

            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_fit_semicircle_from_background_boundary",
                side_effect=rejected_candidate,
            ), mock.patch.object(
                module, "_detect_roi_rim_intrusion", wraps=module._detect_roi_rim_intrusion
            ) as fallback:
                result = module.detect_wafer_notch(semicircle_fixture(), **roi_options())
                fallback.assert_called_once()
                self.assertTrue(result.found)
                self.assertTrue(result.fallback_used)
                self.assert_outer_circle_midpoint(result)

    def test_original_detector_exceptions_are_not_hidden_by_fallback(self):
        for module in self.modules:
            for error_type in (RuntimeError, ValueError):
                with self.subTest(module=module.__name__, error=error_type), mock.patch.object(
                    module, "_fit_semicircle_from_background_boundary",
                    side_effect=error_type("original detector sentinel"),
                ), mock.patch.object(module, "_detect_roi_rim_intrusion") as fallback:
                    with self.assertRaisesRegex(error_type, "original detector sentinel"):
                        module.detect_wafer_notch(
                            semicircle_fixture(), **roi_options(failure_mode="zero")
                        )
                    fallback.assert_not_called()

    def test_fallback_internal_exceptions_are_not_converted_to_zero(self):
        for module in self.modules:
            for error_type in (RuntimeError, ValueError):
                with self.subTest(module=module.__name__, error=error_type), mock.patch.object(
                    module, "_fit_semicircle_from_background_boundary", return_value=None
                ), mock.patch.object(
                    module, "_detect_roi_rim_intrusion",
                    side_effect=error_type("fallback detector sentinel"),
                ):
                    with self.assertRaisesRegex(error_type, "fallback detector sentinel"):
                        module.detect_wafer_notch(
                            semicircle_fixture(), **roi_options(failure_mode="zero")
                        )

    def test_circle_without_notch_keeps_zero_or_error_policy(self):
        for module in self.modules:
            for mode in ("zero", "error"):
                with self.subTest(module=module.__name__, mode=mode), mock.patch.object(
                    module, "_fit_semicircle_from_background_boundary", return_value=None
                ):
                    if mode == "error":
                        with self.assertRaisesRegex(RuntimeError, "notch was not found"):
                            module.detect_wafer_notch(
                                semicircle_fixture(False), **roi_options(failure_mode=mode)
                            )
                    else:
                        result = module.detect_wafer_notch(
                            semicircle_fixture(False), **roi_options(failure_mode=mode)
                        )
                        self.assertFalse(result.found)
                        self.assertTrue(result.fallback_attempted)
                        self.assertFalse(result.fallback_used)
                        self.assertTrue(result.fallback_reason)
                        self.assertEqual(result.correction_angle_deg, 0.0)
                        self.assertIsNone(result.notch_shoulder_points_px)

    def test_invalid_fallback_mode_is_input_error(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "notch_fallback_mode"):
                    module.detect_wafer_notch(
                        semicircle_fixture(), **roi_options(notch_fallback_mode="guess")
                    )

    def test_visual_transform_scales_and_offsets_shoulders_without_mutation(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_fit_semicircle_from_background_boundary", return_value=None
            ):
                result = module.detect_wafer_notch(semicircle_fixture(), **roi_options())
                self.assertTrue(result.fallback_used)
                shoulders = np.asarray(result.notch_shoulder_points_px).copy()
                scale, offset = 0.37, (-121.25, 32.75)
                transformed = module._transform_result_for_visual(
                    result, scale=scale, offset=offset
                )
                np.testing.assert_allclose(
                    transformed.notch_shoulder_points_px,
                    shoulders * scale + np.asarray(offset),
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    transformed.notch_point_px,
                    np.asarray(result.notch_point_px) * scale + np.asarray(offset),
                    atol=1e-10,
                )
                np.testing.assert_array_equal(result.notch_shoulder_points_px, shoulders)
                self.assertEqual(transformed.notch_angle_deg, result.notch_angle_deg)
                self.assertEqual(transformed.fallback_used, result.fallback_used)
                self.assertEqual(transformed.fallback_reason, result.fallback_reason)
                self.assert_outer_circle_midpoint(transformed)

    def test_resized_overlay_draws_shoulders_in_display_coordinates(self):
        image = semicircle_fixture()
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.object(
                module, "_fit_semicircle_from_background_boundary", return_value=None
            ):
                result = module.detect_wafer_notch(image, **roi_options())
            with mock.patch.object(
                module.cv2, "drawMarker", wraps=module.cv2.drawMarker
            ) as draw_marker:
                overlay = module.make_notch_overlay(image, result, max_dimension=450)
            self.assertEqual(overlay.shape, (450, 450, 3))
            observed = [
                call.args[1] for call in draw_marker.call_args_list
                if call.args[2] == (0, 165, 255)
            ]
            expected = [
                tuple(int(round(value * 0.5)) for value in point)
                for point in result.notch_shoulder_points_px
            ]
            self.assertEqual(observed, expected)

    def test_two_deep_pockets_with_one_shallow_mouth_are_not_ambiguous(self):
        # One damaged opening can have two deep portions and an uneven shallow
        # floor between them. High-threshold groups share the same low-threshold
        # shoulders: they are one physical mouth, not two competing notches.
        size, center, radius = 500, (250.0, 250.0), 180.0
        yy, xx = np.indices((size, size))
        angles = np.arctan2(yy - center[1], xx - center[0])
        arc_offset = radius * (angles - np.pi / 2.0)
        depth = np.where(
            np.abs(arc_offset) < 30.0,
            np.where(np.abs(arc_offset) < 10.0, 2.5, 15.0),
            0.0,
        )
        exterior = (
            np.hypot(xx - center[0], yy - center[1]) > radius - depth
        ).astype(np.uint8) * 255
        for module in self.modules:
            with self.subTest(module=module.__name__):
                geometry = module._RoiBackgroundGeometry(
                    palette_lab=np.zeros((1, 3)), distance_threshold_lab=8.0,
                    sample_mask=exterior, background_like_mask=exterior,
                    exterior_background_mask=exterior, wafer_mask=255 - exterior,
                    wafer_contour=np.zeros((0, 1, 2), np.int32),
                    wafer_center=center, wafer_radius=radius,
                    wafer_circle_residual=1.5, roi_bounds=(170, 380, 330, 480),
                    outward_unit=(0.0, 1.0),
                )
                candidate, reason = module._detect_roi_rim_intrusion(
                    geometry, center, radius, (250.0, 430.0),
                    search_half_width_deg=35.0, radial_inner_ratio=0.88,
                    min_depth=3.0, radius_range=None,
                )
                self.assertIsNotNone(candidate, reason)
                self.assertEqual(reason, "rim_intrusion_accepted")
                self.assertLess(abs(candidate.angle_deg - 90.0), 0.5)

    def test_builder_forwards_both_fallback_modes(self):
        image = semicircle_fixture()
        clip = image[194:706, 194:706]
        detections = np.asarray([
            (256 + ix * 70.0, 256 + iy * 82.0)
            for iy in range(-2, 3) for ix in range(-2, 3)
        ])
        for module in (PIPELINE, STANDALONE, ADAPTIVE):
            for mode in ("none", "rim_intrusion"):
                with self.subTest(module=module.__name__, mode=mode), mock.patch.object(
                    module, "detect_wafer_notch", wraps=module.detect_wafer_notch
                ) as detector:
                    dm = module.build_die_map_from_yolo(
                        image, clip, detections,
                        detection_format="point", clip_origin=(194, 194),
                        refine=False, return_aligned_image=True,
                        notch_fallback_mode=mode, **roi_options(),
                    )
                    self.assertEqual(detector.call_args.kwargs["notch_fallback_mode"], mode)
                    self.assertTrue(dm.notch_result.found)
                    self.assertFalse(dm.notch_result.fallback_used)
                    self.assertEqual(dm.grid_angle_deg, 0.0)
                    self.assertEqual(dm.aligned_image.shape, image.shape)
                    self.assertEqual(dm.coordinate_space, "aligned_image")


if __name__ == "__main__":
    unittest.main(verbosity=2)
