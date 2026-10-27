"""Regression evidence for conservative gradient-only point refinement.

The fixtures have known street centres. No model inference or external data
is needed, and failed image evidence must preserve the model point exactly.
"""

import math
import unittest

import numpy as np

from codex.wafer_via import _profile_street_center, refine_cross_point


def patterned_cross(with_pattern=True):
    image = np.empty((128, 128, 3), np.uint8)
    image[:64, :64] = (10, 80, 20)
    image[:64, 64:] = (120, 180, 120)
    image[64:, :64] = (120, 180, 120)
    image[64:, 64:] = (239, 245, 239)
    if with_pattern:
        image[:, 57:60] = (0, 30, 0)
        image[57:60, :] = (0, 30, 0)
    image[:, 62:67] = (245, 245, 245)
    image[62:67, :] = (245, 245, 245)
    return image


class RefinementSafetyTests(unittest.TestCase):
    def assert_preserved(self, image, seed, **kwargs):
        point, confidence = refine_cross_point(image, seed, **kwargs)
        self.assertEqual(point, seed)
        self.assertEqual(confidence, 0.0)

    def test_stronger_same_side_pattern_preserves_exact_model_point(self):
        # Previous output was (59,59) despite the true centre being (64,64).
        self.assert_preserved(patterned_cross(), (64.0, 64.0))

    def test_pattern_fallback_is_independent_of_quadrant(self):
        image = patterned_cross()
        seed = (64.0, 64.0)
        for turns in range(4):
            with self.subTest(turns=turns):
                self.assert_preserved(image, seed)
            image = np.rot90(image).copy()
            seed = (seed[1], 127.0 - seed[0])

    def test_pale_adjacent_band_cannot_merge_into_a_wider_street(self):
        image = np.full((201, 201, 3), (30, 150, 45), np.uint8)
        image[:, 90:96] = (230, 255, 230)
        image[90:96, :] = (230, 255, 230)
        image[:, 96:105] = 255
        image[96:105, :] = 255
        # The wrong broad pair contains the seed too: bracketing alone is
        # insufficient. Previously its midpoint was (97.5,97.5), confidence 1.
        for turns in range(4):
            with self.subTest(turns=turns):
                self.assert_preserved(np.rot90(image, turns).copy(), (100.0, 100.0))

    def test_clear_cross_remains_confident(self):
        point, confidence = refine_cross_point(patterned_cross(False), (64.0, 64.0))
        self.assertLess(math.dist(point, (64.0, 64.0)), 1.0)
        self.assertGreater(confidence, 0.5)

    def test_clear_offset_points_are_actually_corrected_across_palettes(self):
        palettes = (
            ((20, 20, 220), (220, 40, 40), (40, 200, 40), (180, 40, 200)),
            ((10, 120, 180), (180, 120, 10), (120, 10, 180), (30, 180, 120)),
            ((30, 30, 30), (210, 210, 210), (150, 70, 20), (30, 160, 190)),
        )
        for palette in palettes:
            image = np.full((200, 200, 3), 230, np.uint8)
            image[:92, :96] = palette[0]
            image[:92, 105:] = palette[1]
            image[103:, :96] = palette[2]
            image[103:, 105:] = palette[3]
            for seed in ((102.0, 95.0), (98.0, 99.0)):
                with self.subTest(palette=palette, seed=seed):
                    point, confidence = refine_cross_point(
                        image, seed, search_radius=24, max_street_width=18,
                    )
                    self.assertNotEqual(point, seed)
                    self.assertLess(math.dist(point, (100.0, 97.0)), 1.1)
                    self.assertGreater(confidence, 0.5)

    def test_weak_quantized_cross_does_not_get_normalized_into_confidence(self):
        image = np.full((128, 128, 3), (120, 122, 120), np.uint8)
        image[:, 62:67] = (123, 123, 123)
        image[62:67, :] = (123, 123, 123)
        self.assert_preserved(image, (65.25, 63.75))

    def test_flat_and_low_level_noise_keep_fractional_seed(self):
        seed = (64.25, 63.75)
        flat = np.full((128, 128, 3), 128, np.uint8)
        self.assert_preserved(flat, seed)
        rng = np.random.default_rng(20261027)
        noisy = np.clip(flat.astype(np.int16) + rng.integers(-1, 2, flat.shape), 0, 255).astype(np.uint8)
        self.assert_preserved(noisy, seed)

    def test_distinct_equally_plausible_edge_pairs_are_rejected(self):
        profile = np.zeros(41, np.float32)
        profile[[8, 16, 24, 32]] = 1.0
        self.assertEqual(_profile_street_center(profile, 18.0, 18), (18.0, 0.0))

    def test_equal_same_side_pairs_do_not_inherit_peak_iteration_bias(self):
        profile = np.zeros(37, np.float32)
        profile[[5, 8, 28, 31]] = 1.0
        self.assertEqual(_profile_street_center(profile, 18.0, 18), (18.0, 0.0))

    def test_truncated_context_at_each_image_edge_keeps_model_point(self):
        for seed in ((5.25, 64.0), (123.0, 64.25), (64.0, 5.25), (64.25, 123.0)):
            image = np.full((128, 128, 3), (10, 100, 20), np.uint8)
            x, y = map(round, seed)
            image[:, max(0, x - 3):min(128, x + 4)] = 245
            image[max(0, y - 3):min(128, y + 4), :] = 245
            with self.subTest(seed=seed):
                self.assert_preserved(image, seed)

    def test_single_axis_or_seed_outside_street_is_not_partially_applied(self):
        image = np.full((128, 128, 3), (10, 100, 20), np.uint8)
        image[:, 60:69] = 245
        self.assert_preserved(image, (65.0, 63.0))
        self.assert_preserved(patterned_cross(False), (70.25, 64.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
