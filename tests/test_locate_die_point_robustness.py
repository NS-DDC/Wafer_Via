"""Point lookup regressions for scalar inputs and extreme numeric grid data."""

import importlib
import math
import unittest

import numpy as np


MODULE_NAMES = (
    "codex.wafer_via", "codex.wafer_via_notch",
    "codex.wafer_via_notch_standalone", "codex.wafer_via_notch_adaptive_standalone",
)


class FloatOnlyScalar:
    """A scalar protocol accepted by single lookup, without a NumPy adapter."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("This scalar deliberately provides only __float__")


class NoIterationDict(dict):
    def __iter__(self):
        raise AssertionError("A point lookup must not scan all stored cell keys")


def make_map(module, *, pitch=1.0, angle=0.0, indices=()):
    dm = module.WaferDieMap(
        wafer_cx=0, wafer_cy=0, wafer_r=5000,
        pitch_x=float(pitch), pitch_y=float(pitch), x0=0.0, y0=0.0,
        die_w=1, die_h=1, pixel_per_unit=1.0,
        image_shape=(5000, 5000), grid_angle_deg=float(angle),
    )
    for index in indices:
        # Build valid geometry with Python ints; retain the caller's key dtype
        # to exercise deserialized/imported NumPy integer dictionary keys.
        ix, iy = int(index[0]), int(index[1])
        corners = np.asarray(((ix, -iy - 1), (ix + 1, -iy - 1),
                              (ix + 1, -iy), (ix, -iy)), dtype=np.float64) * float(pitch)
        radians = math.radians(angle)
        rotation = np.asarray(((math.cos(radians), -math.sin(radians)),
                               (math.sin(radians), math.cos(radians))))
        polygon = corners @ rotation.T
        lower = np.floor(polygon.min(axis=0)).astype(int)
        upper = np.ceil(polygon.max(axis=0)).astype(int)
        rect = tuple(int(v) for v in (*lower, *upper))
        entry = {
            "is_edge_partial": False, "is_edge_ring": False,
            "rect_px": rect, "crop_rect_px": rect,
        }
        dm.dies_by_index[index] = entry
    return dm


class LocateDiePointRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [importlib.import_module(name) for name in MODULE_NAMES]

    def test_float_scalar_protocol_matches_single_result(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                dm = make_map(module, indices=((0, 0),))
                point = (FloatOnlyScalar(0.5), FloatOnlyScalar(-0.5))
                single = module.locate_die(dm, point=point)
                self.assertEqual(len(single), 24)
                self.assertEqual(module.locate_die(dm, point=point, return_all=True), [single])

    def test_tiny_pitch_keeps_geometric_point_hit_count_bounded(self):
        indices = tuple((ix, iy) for iy in range(10) for ix in range(10))
        cases = (
            ((5.5e-10, -5.5e-10), [(5, 5)]),
            ((5.0e-10, -5.5e-10), [(4, 5), (5, 5)]),
            ((5.0e-10, -5.0e-10), [(4, 4), (5, 4), (4, 5), (5, 5)]),
        )
        for module in self.modules:
            dm = make_map(module, pitch=1e-10, indices=indices)
            for point, expected in cases:
                with self.subTest(module=module.__name__, point=point):
                    hits = module.locate_die(dm, point=point, return_all=True)
                    self.assertEqual([hit["die_index"] for hit in hits], expected)
                    self.assertTrue(all("is_fallback" not in hit for hit in hits))

    def test_numpy_int32_extreme_key_does_not_overflow_geometry(self):
        index = (np.int32(2147483647), np.int32(0))
        point = (float(index[0]) * 1e-6, 0.0)
        for module in self.modules:
            with self.subTest(module=module.__name__):
                dm = make_map(module, pitch=1e-6, indices=(index,))
                single = module.locate_die(dm, point=point)
                with np.errstate(over="raise", invalid="raise"):
                    hits = module.locate_die(dm, point=point, return_all=True)
                self.assertEqual(hits, [single])
                xs = [p[0] for p in hits[0]["die_polygon_px"]]
                self.assertGreater(min(xs), 2147.0)
                self.assertLess(max(xs) - min(xs), 2e-6)

    def test_empty_map_fallback_never_iterates_stored_keys(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                dm = make_map(module, angle=23.0)
                dm.dies_by_index = NoIterationDict()
                point = (12.25, -34.5)
                expected = module.locate_die(dm, point=point)
                expected["is_fallback"] = True
                self.assertEqual(module.locate_die(dm, point=point, return_all=True), [expected])

    def test_normal_rotated_corner_and_near_boundary_keep_multiple_hits(self):
        indices = ((0, 0), (1, 0), (0, 1), (1, 1))
        for module in self.modules:
            for angle in (0.0, 23.0):
                dm = make_map(module, pitch=70.0, angle=angle, indices=indices)
                axis_x, axis_y = np.asarray(dm.axis_x), np.asarray(dm.axis_y)
                point = tuple(70.0 * axis_x - 70.0 * axis_y)
                with self.subTest(module=module.__name__, angle=angle):
                    hits = module.locate_die(dm, point=point, return_all=True)
                    self.assertEqual([hit["die_index"] for hit in hits], list(indices))
                    interior = tuple(35.0 * axis_x - 35.0 * axis_y)
                    self.assertEqual(module.locate_die(dm, point=interior, return_all=True),
                                     [module.locate_die(dm, point=interior)])
            dm = make_map(module, pitch=70.0, indices=indices)
            for x in (70.0 - 5e-8, 70.0 + 5e-8):
                with self.subTest(module=module.__name__, near_boundary=x):
                    hits = module.locate_die(dm, point=(x, -35.0), return_all=True)
                    self.assertEqual([hit["die_index"] for hit in hits], [(0, 0), (1, 0)])

    def test_bbox_numpy_integer_keys_keep_finite_local_geometry(self):
        index = (np.int32(2147483647), np.int32(0))
        edge = float(index[0]) * 1e-6
        bounds = (edge - 0.25e-6, -0.5e-6, edge + 0.25e-6, 0.5e-6)
        for module in self.modules:
            with self.subTest(module=module.__name__):
                dm = make_map(module, pitch=1e-6, indices=(index,))
                with np.errstate(over="raise", invalid="raise"):
                    hits = module.locate_die(dm, bbox=bounds, return_all=True)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["die_index"], (2147483647, 0))
                self.assertTrue(all(math.isfinite(v) for p in hits[0]["die_polygon_px"] for v in p))
                self.assertGreater(min(p[0] for p in hits[0]["die_polygon_px"]), 2147.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
