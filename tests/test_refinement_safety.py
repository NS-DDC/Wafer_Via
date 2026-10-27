"""Safety regressions for model-coordinate refinement and distributed copies."""

import ast
import importlib
import inspect
import math
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "codex.wafer_via",
    "codex.wafer_via_die_render",
    "codex.wafer_via_notch_standalone",
    "codex.wafer_via_notch_adaptive_standalone",
)
HELPERS = (
    "_profile_street_center", "_profile_street_candidate", "_refinement_odd_kernel",
    "_profile_colour_band_center", "_corner_colour_candidate", "refine_cross_point",
)


def white_green_cross(*, adjacent_pattern=False, contaminated_corner=False):
    image = np.full((201, 201, 3), (30, 150, 45), np.uint8)
    if adjacent_pattern:
        image[:, 90:96] = (190, 250, 190)
        image[90:96, :] = (190, 250, 190)
    if contaminated_corner:
        image[80:91, 80:91] = 255
    # Draw the true cross last: neither adversarial feature overwrites it.
    image[:, 96:105] = 255
    image[96:105, :] = 255
    return image


def noisy_off_seed_cross():
    rng = np.random.default_rng(20260820)
    image = np.full((200, 200, 3), 185, np.uint8)
    image[:91, :95] = (24, 65, 210)
    image[:91, 106:] = (205, 55, 45)
    image[104:, :95] = (55, 185, 85)
    image[104:, 106:] = (165, 45, 190)
    image = np.clip(image.astype(np.float32) + rng.normal(0, 28, image.shape), 0, 255).astype(np.uint8)
    impulses = rng.random(image.shape[:2]) < 0.035
    image[impulses] = rng.integers(0, 256, (int(impulses.sum()), 3), dtype=np.uint8)
    return image


def dark_pattern_cross():
    image = np.empty((128, 128, 3), np.uint8)
    image[:64, :64] = (10, 80, 20)
    image[:64, 64:] = (120, 180, 120)
    image[64:, :64] = (120, 180, 120)
    image[64:, 64:] = (239, 245, 239)
    image[:, 57:60] = (0, 30, 0)
    image[57:60, :] = (0, 30, 0)
    image[:, 62:67] = (245, 245, 245)
    image[62:67, :] = (245, 245, 245)
    return image


class RefinementSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [importlib.import_module(name) for name in MODULES]

    def assert_abstains(self, module, image, seed, **kwargs):
        point, confidence = module.refine_cross_point(image, seed, **kwargs)
        self.assertEqual(point, seed)
        self.assertEqual(confidence, 0.0)

    def test_pattern_conflict_keeps_exact_seed_in_auto_and_explicit_colour(self):
        for module in self.modules:
            for contaminated in (False, True):
                image = white_green_cross(adjacent_pattern=True, contaminated_corner=contaminated)
                for flipped in (False, True):
                    view = image[::-1, ::-1].copy() if flipped else image
                    for mode in ("auto", "corner_color"):
                        with self.subTest(module=module.__name__, contaminated=contaminated, flipped=flipped, mode=mode):
                            self.assert_abstains(module, view, (100.0, 100.0), mode=mode)

    def test_unpatterned_white_green_cross_remains_supported(self):
        image = white_green_cross()
        for module in self.modules:
            for mode in ("auto", "gradient", "corner_color"):
                with self.subTest(module=module.__name__, mode=mode):
                    point, confidence = module.refine_cross_point(image, (100.0, 100.0), mode=mode)
                    self.assertLess(math.dist(point, (100.0, 100.0)), 0.01)
                    self.assertGreater(confidence, 0.5)

    def test_band_width_and_green_contrast_variations_abstain(self):
        for module in self.modules:
            for start in (89, 91, 93):
                for colour in ((120, 240, 130), (190, 250, 190), (0, 60, 0)):
                    image = np.full((201, 201, 3), (30, 150, 45), np.uint8)
                    image[:, start:96] = colour
                    image[start:96, :] = colour
                    image[:, 96:105] = 255
                    image[96:105, :] = 255
                    for mode in ("auto", "corner_color"):
                        with self.subTest(module=module.__name__, start=start, colour=colour, mode=mode):
                            self.assert_abstains(module, image, (100.0, 100.0), mode=mode)

    def test_near_white_band_cannot_override_seed_supported_street(self):
        image = np.full((201, 201, 3), (30, 150, 45), np.uint8)
        image[:, 90:96] = (230, 255, 230)
        image[90:96, :] = (230, 255, 230)
        image[:, 96:105] = 255
        image[96:105, :] = 255
        for module in self.modules:
            for flipped in (False, True):
                view = image[::-1, ::-1].copy() if flipped else image
                for mode in ("auto", "gradient", "corner_color"):
                    with self.subTest(module=module.__name__, flipped=flipped, mode=mode):
                        self.assert_abstains(module, view, (100.0, 100.0), mode=mode)

    def test_same_side_dark_pattern_is_rejected_in_every_mode(self):
        for module in self.modules:
            image, seed = dark_pattern_cross(), (64.0, 64.0)
            for turns in range(4):
                for mode in ("auto", "gradient", "corner_color"):
                    with self.subTest(module=module.__name__, turns=turns, mode=mode):
                        self.assert_abstains(module, image, seed, mode=mode)
                image = np.rot90(image).copy()
                seed = (seed[1], 127.0 - seed[0])

    def test_confident_noisy_off_seed_correction_is_retained(self):
        image = noisy_off_seed_cross()
        for module in self.modules:
            for mode in ("auto", "corner_color"):
                with self.subTest(module=module.__name__, mode=mode):
                    point, confidence = module.refine_cross_point(
                        image, (108.0, 88.0), search_radius=30,
                        max_street_width=22, mode=mode, noise_kernel=7,
                    )
                    self.assertLess(math.dist(point, (100.5, 97.0)), 1.0)
                    self.assertGreater(confidence, 0.5)

    def test_flat_weak_one_axis_and_clipped_windows_abstain(self):
        rng = np.random.default_rng(3721)
        flat = np.full((201, 201, 3), 120, np.uint8)
        weak = np.clip(flat.astype(np.int16) + rng.integers(-1, 2, flat.shape), 0, 255).astype(np.uint8)
        one_axis = np.full_like(flat, (30, 150, 45))
        one_axis[:, 96:105] = 255
        clipped = np.full_like(flat, (30, 150, 45))
        clipped[:, :9] = 255
        clipped[:9, :] = 255
        for module in self.modules:
            for image, seed, kind in ((flat, (100.0, 100.0), "flat"), (weak, (100.0, 100.0), "weak"),
                                      (one_axis, (100.0, 100.0), "one_axis"), (clipped, (4.0, 4.0), "clipped")):
                for mode in ("auto", "gradient", "corner_color"):
                    with self.subTest(module=module.__name__, kind=kind, mode=mode):
                        self.assert_abstains(module, image, seed, mode=mode)

    def test_uncorroborated_gradient_pair_cannot_jump_off_seed(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assert_abstains(module, white_green_cross(), (107.0, 107.0), mode="gradient")

    def test_colour_parameters_remain_accepted_without_bypassing_safety(self):
        image = white_green_cross(adjacent_pattern=True, contaminated_corner=True)
        for module in self.modules:
            for weight in (0.0, 0.70, 1.0):
                with self.subTest(module=module.__name__, weight=weight):
                    self.assert_abstains(
                        module, image, (100.0, 100.0), mode="corner_color",
                        search_radius=18, max_street_width=18,
                        corner_patch_ratio=0.22, corner_reference_weight=weight,
                        noise_kernel=5,
                    )

    def test_distributed_refinement_code_and_public_defaults_match(self):
        expected = None
        expected_signature = None
        for module in self.modules:
            source = Path(module.__file__).read_text(encoding="utf-8")
            nodes = {node.name: ast.dump(node, include_attributes=False)
                     for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in HELPERS}
            self.assertEqual(set(nodes), set(HELPERS))
            signature = str(inspect.signature(module.refine_cross_point))
            if expected is None:
                expected, expected_signature = nodes, signature
            else:
                self.assertEqual(nodes, expected, module.__name__)
                self.assertEqual(signature, expected_signature, module.__name__)
            defaults = inspect.signature(module.refine_cross_point).parameters
            self.assertEqual(defaults["search_radius"].default, 18)
            self.assertIsNone(defaults["max_street_width"].default)
            self.assertEqual(defaults["mode"].default, "auto")
            self.assertEqual(defaults["corner_patch_ratio"].default, 0.22)
            self.assertEqual(defaults["corner_reference_weight"].default, 0.70)
            self.assertEqual(defaults["noise_kernel"].default, 5)


if __name__ == "__main__":
    unittest.main()
