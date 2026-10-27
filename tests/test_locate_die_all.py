import unittest

import cv2
import numpy as np

from codex import wafer_via as base
from codex import wafer_via_notch as pipeline
from codex import wafer_via_notch_standalone as standalone
from codex import wafer_via_notch_adaptive_standalone as adaptive


MODULES = (base, pipeline, standalone, adaptive)


def make_map(module, angle=0.0, include_edge=True):
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    contour = np.rint(np.column_stack((
        500.0 + 450.0 * np.cos(angles), 500.0 + 450.0 * np.sin(angles)
    ))).astype(np.int32).reshape(-1, 1, 2)
    boundary = module.WaferBoundary((500.0, 500.0), 450.0, contour)
    return module.generate_die_map(
        boundary, (1000, 1000), (500.0, 500.0), 70.0, 82.0, angle,
        include_edge=include_edge,
    )


class LocateDieAllTests(unittest.TestCase):
    def assert_point_fallback(self, module, dm, point):
        default = module.locate_die(dm, point=point)
        self.assertNotIn("is_fallback", default)
        expected = dict(default, is_fallback=True)
        hits = module.locate_die(dm, point=point, return_all=True)
        self.assertEqual(hits, [expected])
        self.assertTrue(hits[0].get("is_fallback", False))

    def assert_stored_hit_schema(self, module, dm, point, hits):
        expected_keys = set(module.locate_die(dm, point=point))
        for hit in hits:
            self.assertEqual(set(hit), expected_keys)
            self.assertNotIn("is_fallback", hit)
            self.assertFalse(hit.get("is_fallback", False))
            self.assertEqual(hit["query_px"], tuple(point))
            entry = dm.get_die(*hit["die_index"])
            self.assertIsNotNone(entry)
            self.assertEqual(hit["die_rect_px"], entry["rect_px"])
            self.assertEqual(hit["crop_rect_px"], entry["crop_rect_px"])

    def test_default_and_explicit_false_keep_point_and_bbox_results(self):
        for module in MODULES:
            dm = make_map(module)
            for kwargs in (dict(point=(535.0, 459.0)), dict(bbox=(520, 400, 590, 480))):
                with self.subTest(module=module.__name__, query=kwargs):
                    default = module.locate_die(dm, **kwargs)
                    self.assertIsInstance(default, dict)
                    self.assertEqual(default, module.locate_die(dm, return_all=False, **kwargs))
                    self.assertEqual(default["crop_rect_px"], dm.get_die(*default["die_index"])["crop_rect_px"])

    def test_interior_point_returns_one_unchanged_result(self):
        for module in MODULES:
            dm = make_map(module, 17.0)
            point = tuple(np.mean(dm.get_die(2, -3)["polygon_px"], axis=0))
            hits = module.locate_die(dm, point=point, return_all=True)
            self.assertEqual(hits, [module.locate_die(dm, point=point)])
            self.assert_stored_hit_schema(module, dm, point, hits)

    def test_point_on_shared_edge_returns_two_dies(self):
        for module in MODULES:
            dm = make_map(module)
            hits = module.locate_die(dm, point=(570.0, 459.0), return_all=True)
            self.assertEqual([h["die_index"] for h in hits], [(0, 0), (1, 0)])
            self.assert_stored_hit_schema(module, dm, (570.0, 459.0), hits)
            left = module.locate_die(dm, point=(570.0 - 1e-4, 459.0), return_all=True)
            right = module.locate_die(dm, point=(570.0 + 1e-4, 459.0), return_all=True)
            self.assertEqual([h["die_index"] for h in left], [(0, 0)])
            self.assertEqual([h["die_index"] for h in right], [(1, 0)])

    def test_point_on_corner_returns_four_for_rotated_grid(self):
        for module in MODULES:
            dm = make_map(module, 23.0)
            point = (
                np.asarray((dm.x0, dm.y0)) + dm.pitch_x * np.asarray(dm.axis_x)
                - dm.pitch_y * np.asarray(dm.axis_y)
            )
            hits = module.locate_die(dm, point=tuple(point), return_all=True)
            self.assertEqual([h["die_index"] for h in hits], [(0, 0), (1, 0), (0, 1), (1, 1)])
            self.assert_stored_hit_schema(module, dm, tuple(point), hits)

    def test_bbox_spans_nine_cells_including_cells_without_query_corners(self):
        for module in MODULES:
            dm = make_map(module)
            bounds = (505, 255, 705, 495)
            hits = module.locate_die(dm, bbox=bounds, return_all=True)
            self.assertEqual([h["die_index"] for h in hits], [(ix, iy) for iy in range(3) for ix in range(3)])
            self.assertEqual(module.locate_die(dm, bbox=bounds)["die_index"], (1, 1))
            for hit in hits:
                entry = dm.get_die(*hit["die_index"])
                self.assertEqual(hit["die_rect_px"], entry["rect_px"])
                self.assertEqual(hit["crop_rect_px"], entry["crop_rect_px"])
                self.assertEqual(hit["query_px"], (605, 375))
                self.assertNotIn("is_fallback", hit)

    def test_bbox_edge_contact_does_not_add_neighbours(self):
        for module in MODULES:
            dm = make_map(module)
            hits = module.locate_die(dm, bbox=(500, 418, 570, 500), return_all=True)
            self.assertEqual([h["die_index"] for h in hits], [(0, 0)])

    def test_rotated_bbox_matches_independent_polygon_intersections(self):
        rng = np.random.default_rng(20260916)
        for module in MODULES:
            dm = make_map(module, 23.0)
            for _ in range(12):
                x, y = rng.uniform(150, 800, 2)
                w, h = rng.uniform(5, 170, 2)
                bounds = (x, y, x + w, y + h)
                rectangle = np.asarray(((x, y), (x + w, y), (x + w, y + h), (x, y + h)), np.float32)
                expected = []
                for entry in dm.dies:
                    area, _ = cv2.intersectConvexConvex(np.asarray(entry["polygon_px"], np.float32), rectangle)
                    if area > 1e-4:
                        expected.append(entry["index"])
                hits = module.locate_die(dm, bbox=bounds, return_all=True)
                self.assertEqual([h["die_index"] for h in hits], expected)

    def test_unmapped_point_falls_back_but_bbox_stays_empty(self):
        for module in MODULES:
            dm = make_map(module, include_edge=False)
            self.assert_point_fallback(module, dm, (-500, -500))
            self.assertEqual(
                module.locate_die(dm, bbox=(-500, -500, -400, -400), return_all=True), [],
            )
            self.assertIsInstance(module.locate_die(dm, bbox=(-500, -500, -400, -400)), dict)
            all_hits = module.locate_die(dm, bbox=(-1e12, -1e12, 1e12, 1e12), return_all=True)
            self.assertEqual([h["die_index"] for h in all_hits], [d["index"] for d in dm.dies])
            self.assertTrue(all("is_fallback" not in hit for hit in all_hits))

    def test_empty_map_point_returns_one_fallback_and_bbox_stays_empty(self):
        for module in MODULES:
            dm = make_map(module)
            dm.dies.clear()
            dm.dies_by_index.clear()
            # Even a shared grid corner produces one default-calculated fallback,
            # rather than inventing four stored neighbours in an empty map.
            self.assert_point_fallback(module, dm, (500, 500))
            self.assertEqual(module.locate_die(dm, bbox=(10, 10, 900, 900), return_all=True), [])
            self.assertEqual(dm.dies, [])
            self.assertEqual(dm.dies_by_index, {})

    def test_point_in_edge_cell_excluded_from_map_falls_back(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                complete = make_map(module, 17.0, include_edge=True)
                filtered = make_map(module, 17.0, include_edge=False)
                contour = complete.wafer_boundary.contour_px
                removed = next(
                    entry for entry in complete.dies
                    if entry["is_edge_partial"]
                    and entry["index"] not in filtered.dies_by_index
                    and cv2.pointPolygonTest(
                        contour, tuple(float(v) for v in np.mean(entry["polygon_px"], axis=0)), False,
                    ) >= 0
                )
                point = tuple(float(v) for v in np.mean(removed["polygon_px"], axis=0))
                normal_hits = module.locate_die(complete, point=point, return_all=True)
                self.assertEqual([hit["die_index"] for hit in normal_hits], [removed["index"]])
                self.assert_stored_hit_schema(module, complete, point, normal_hits)
                count_before = len(filtered.dies_by_index)
                self.assert_point_fallback(module, filtered, point)
                self.assertEqual(module.locate_die(filtered, point=point)["die_index"], removed["index"])
                self.assertEqual(len(filtered.dies_by_index), count_before)
                self.assertIsNone(filtered.get_die(*removed["index"]))

    def test_unmapped_rotated_and_negative_points_match_single_result(self):
        for module in MODULES:
            for angle, point in ((23.0, (-500.25, -900.5)), (-31.0, (1500.25, -100.75))):
                with self.subTest(module=module.__name__, angle=angle, point=point):
                    dm = make_map(module, angle)
                    default = module.locate_die(dm, point=point)
                    self.assertNotIn(default["die_index"], dm.dies_by_index)
                    self.assert_point_fallback(module, dm, point)

    def test_new_mode_validates_input(self):
        invalid = (
            {}, dict(point=(1, 2), bbox=(0, 0, 1, 1)), dict(point=(float("nan"), 0)),
            dict(point=(float("inf"), 0)), dict(point=(0, float("-inf"))),
            dict(point=(1, 2, 3)), dict(point=()), dict(bbox=(0, 0, float("inf"), 1)),
            dict(bbox=(2, 0, 1, 1)), dict(bbox=(0, 0, 0, 1)),
        )
        for module in MODULES:
            dm = make_map(module)
            for query in invalid:
                with self.subTest(module=module.__name__, query=query), self.assertRaises(ValueError):
                    module.locate_die(dm, return_all=True, **query)


if __name__ == "__main__":
    unittest.main()
