"""Synthetic, known-geometry evidence for the opt-in cross-lines refiner.

These tests do not constitute validation on the user's camera image or model.
"""

import importlib
import inspect
import math
import unittest

import cv2
import numpy as np


MODULE_NAMES = (
    "codex.wafer_via", "codex.wafer_via_die_render",
    "codex.wafer_via_notch_standalone", "codex.wafer_via_notch_adaptive_standalone",
)


def cross_image(*, dark=False, white_corner=False, white_quadrant=False):
    background = (245, 245, 245) if dark else (30, 150, 45)
    street = (20, 50, 25) if dark else (255, 255, 255)
    image = np.full((201, 201, 3), background, np.uint8)
    if white_corner:
        image[80:91, 80:91] = 255
    if white_quadrant:
        image[:96, :96] = 255
    image[:, 96:105] = street
    image[96:105, :] = street
    return image


def noisy_cross(*, dark=False):
    image = cross_image(dark=dark)
    rng = np.random.default_rng(20261027)
    image = np.clip(image.astype(np.float32) + rng.normal(0.0, 10.0, image.shape), 0, 255).astype(np.uint8)
    impulses = rng.random(image.shape[:2]) < 0.01
    image[impulses] = rng.integers(0, 256, (int(impulses.sum()), 3), dtype=np.uint8)
    return image


class CrossLinesRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [importlib.import_module(name) for name in MODULE_NAMES]

    def assert_corrects(self, module, image, seed=(102.0, 98.0), tolerance=1.0, **kwargs):
        before = image.copy()
        point, confidence = module.refine_cross_point(image, seed, mode="cross_lines", **kwargs)
        self.assertLess(math.dist(point, (100.0, 100.0)), tolerance)
        self.assertNotEqual(point, seed)
        self.assertGreater(confidence, 0.25)
        self.assertTrue(np.array_equal(image, before), "Refinement changed its input image")

    def assert_abstains(self, module, image, seed, **kwargs):
        point, confidence = module.refine_cross_point(image, seed, mode="cross_lines", **kwargs)
        self.assertEqual(point, seed)
        self.assertEqual(confidence, 0.0)

    def test_white_corner_and_white_die_quadrant_do_not_define_street_colour(self):
        for module in self.modules:
            for kind in ("corner", "quadrant"):
                image = cross_image(white_corner=kind == "corner", white_quadrant=kind == "quadrant")
                with self.subTest(module=module.__name__, fixture=kind):
                    self.assert_corrects(module, image, search_radius=24, max_street_width=16)

    def test_dark_cross_on_white_dies_is_also_corrected(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assert_corrects(module, cross_image(dark=True), search_radius=24, max_street_width=16)

    def test_small_rotation_recovers_known_crossing(self):
        source = cross_image()
        for angle in (-10.0, 10.0):
            matrix = cv2.getRotationMatrix2D((100.0, 100.0), angle, 1.0)
            image = cv2.warpAffine(source, matrix, (201, 201), flags=cv2.INTER_CUBIC,
                                   borderValue=(30, 150, 45))
            for module in self.modules:
                with self.subTest(module=module.__name__, angle=angle):
                    self.assert_corrects(module, image, tolerance=1.5,
                                         search_radius=28, max_street_width=16)

    def test_noisy_white_and_dark_crosses_still_move_the_seed(self):
        for dark in (False, True):
            image = noisy_cross(dark=dark)
            for module in self.modules:
                with self.subTest(module=module.__name__, dark=dark):
                    self.assert_corrects(module, image, tolerance=1.5,
                                         search_radius=28, max_street_width=16, noise_kernel=5)

    def test_flat_white_single_axis_and_clipped_context_abstain(self):
        flat = np.full((201, 201, 3), 255, np.uint8)
        single_axis = np.full((201, 201, 3), (30, 150, 45), np.uint8)
        single_axis[:, 96:105] = 255
        clipped = np.full((201, 201, 3), (30, 150, 45), np.uint8)
        clipped[:, :9] = 255
        clipped[:9, :] = 255
        for module in self.modules:
            for image, seed, kind in ((flat, (100.25, 99.75), "flat"),
                                      (single_axis, (102.0, 98.0), "one_axis"),
                                      (clipped, (4.25, 4.25), "clipped")):
                with self.subTest(module=module.__name__, fixture=kind):
                    self.assert_abstains(module, image, seed)

    def test_equally_supported_upper_left_and_lower_right_crosses_abstain(self):
        image = np.full((201, 201, 3), (30, 150, 45), np.uint8)
        # Equal-width, equal-strength crosses at (93,93) and (101,101),
        # with additional mixed intersections. No unique crossing justifies
        # moving the midpoint model seed toward the upper-left candidate.
        for center in (93, 101):
            image[:, center - 1:center + 2] = 255
            image[center - 1:center + 2, :] = 255
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assert_abstains(module, image, (97.0, 97.0),
                                     search_radius=24, max_street_width=12, noise_kernel=3)

    def test_previous_adjacent_pattern_failures_do_not_pull_the_crossing_away(self):
        dark_pattern = np.empty((128, 128, 3), np.uint8)
        dark_pattern[:64, :64] = (10, 80, 20)
        dark_pattern[:64, 64:] = (120, 180, 120)
        dark_pattern[64:, :64] = (120, 180, 120)
        dark_pattern[64:, 64:] = (239, 245, 239)
        dark_pattern[:, 57:60] = (0, 30, 0)
        dark_pattern[57:60, :] = (0, 30, 0)
        dark_pattern[:, 62:67] = (245, 245, 245)
        dark_pattern[62:67, :] = (245, 245, 245)
        pale_pattern = cross_image(white_corner=True)
        pale_pattern[:, 90:96] = (230, 255, 230)
        pale_pattern[90:96, :] = (230, 255, 230)
        pale_pattern[:, 96:105] = 255
        pale_pattern[96:105, :] = 255
        for module in self.modules:
            for image, seed, kind in ((dark_pattern, (64.0, 64.0), "dark"),
                                      (pale_pattern, (100.0, 100.0), "pale")):
                with self.subTest(module=module.__name__, fixture=kind):
                    point, confidence = module.refine_cross_point(image, seed, mode="cross_lines")
                    if confidence == 0.0:
                        self.assertEqual(point, seed)
                    else:
                        self.assertLess(math.dist(point, seed), 1.0)
                        self.assertGreater(confidence, 0.25)

    def test_corner_reference_parameters_do_not_change_cross_lines_result(self):
        image = cross_image(white_corner=True)
        for module in self.modules:
            expected = module.refine_cross_point(
                image, (102.0, 98.0), mode="cross_lines", search_radius=24, max_street_width=16,
            )
            for ratio, weight in ((0.05, 0.0), (0.33, 1.0)):
                with self.subTest(module=module.__name__, ratio=ratio, weight=weight):
                    actual = module.refine_cross_point(
                        image, (102.0, 98.0), mode="cross_lines", search_radius=24,
                        max_street_width=16, corner_patch_ratio=ratio, corner_reference_weight=weight,
                    )
                    self.assertEqual(actual, expected)

    def test_existing_modes_and_default_keep_current_fixture_behavior(self):
        image = cross_image()
        ambiguous = image.copy()
        ambiguous[:, 90:96] = (230, 255, 230)
        ambiguous[90:96, :] = (230, 255, 230)
        ambiguous[:, 96:105] = 255
        ambiguous[96:105, :] = 255
        for module in self.modules:
            with self.subTest(module=module.__name__, default=True):
                self.assertEqual(inspect.signature(module.refine_cross_point).parameters["mode"].default, "auto")
                self.assertEqual(module.refine_cross_point(image, (100.0, 100.0)),
                                 module.refine_cross_point(image, (100.0, 100.0), mode="auto"))
            for mode in ("auto", "gradient", "corner_color"):
                with self.subTest(module=module.__name__, mode=mode):
                    point, confidence = module.refine_cross_point(image, (100.0, 100.0), mode=mode)
                    self.assertLess(math.dist(point, (100.0, 100.0)), 0.01)
                    self.assertGreater(confidence, 0.5)
                    self.assertEqual(module.refine_cross_point(ambiguous, (100.0, 100.0), mode=mode),
                                     ((100.0, 100.0), 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
