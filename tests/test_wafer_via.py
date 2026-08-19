import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from codex.wafer_via import (
    WaferBoundary,
    build_die_map_from_yolo,
    detect_wafer_boundary,
    estimate_grid_from_yolo,
    generate_die_map,
    locate_die,
    make_clip_overlay,
    make_wafer_overlay,
    parse_yolo_points,
    refine_cross_point,
)


ROOT = Path(__file__).resolve().parents[1]


def rotated_points(center=(256.0, 256.0), pitch_x=82.5, pitch_y=91.75, angle_deg=3.25):
    angle = math.radians(angle_deg)
    axis_x = np.array((math.cos(angle), math.sin(angle)))
    axis_y = np.array((-math.sin(angle), math.cos(angle)))
    origin = np.asarray(center, dtype=float)
    return [origin + ix * pitch_x * axis_x + iy * pitch_y * axis_y
            for iy in range(-2, 3) for ix in range(-2, 3)]


class YoloCoordinateTests(unittest.TestCase):
    def test_parse_standard_normalized_yolo_txt_rows(self):
        points = parse_yolo_points(
            [[0, 0.25, 0.50, 0.02, 0.02], [0, 0.75, 0.80, 0.02, 0.02]],
            (512, 512), detection_format="yolo_txt",
        )
        self.assertEqual(points, [(128.0, 256.0), (384.0, 409.6)])

    def test_auto_parses_point_conf_and_six_column_normalized_yolo(self):
        points = parse_yolo_points(
            [[128, 256, 0.90], [384, 410, 0.10]],
            (512, 512), confidence_threshold=0.25,
        )
        self.assertEqual(points, [(128.0, 256.0)])
        normalized = parse_yolo_points(
            [[0, 0.25, 0.50, 0.02, 0.02, 0.95]],
            (512, 512),
        )
        self.assertEqual(normalized, [(128.0, 256.0)])

    def test_center_side_below_pitch_and_angle(self):
        image = np.zeros((512, 512, 3), np.uint8)
        points = rotated_points()
        boxes = [[p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3] for p in points]
        estimate = estimate_grid_from_yolo(image, boxes)
        self.assertTrue(np.allclose(estimate.center_corner_clip, (256.0, 256.0), atol=1e-6))
        self.assertAlmostEqual(estimate.pitch_x, 82.5, places=5)
        self.assertAlmostEqual(estimate.pitch_y, 91.75, places=5)
        self.assertAlmostEqual(estimate.angle_deg, 3.25, places=5)
        self.assertGreater(estimate.angle_confidence, 0.99)
        self.assertEqual(make_clip_overlay(image, estimate).shape, image.shape)


class ColourInvariantRefinementTests(unittest.TestCase):
    def _cross_image(self, colours):
        image = np.full((200, 200, 3), 230, np.uint8)
        image[:92, :96] = colours[0]
        image[:92, 105:] = colours[1]
        image[103:, :96] = colours[2]
        image[103:, 105:] = colours[3]
        return image

    def test_same_cross_for_different_die_colours(self):
        palettes = [
            ((20, 20, 220), (220, 40, 40), (40, 200, 40), (180, 40, 200)),
            ((10, 120, 180), (180, 120, 10), (120, 10, 180), (30, 180, 120)),
            ((30, 30, 30), (210, 210, 210), (150, 70, 20), (30, 160, 190)),
        ]
        refined = []
        for colours in palettes:
            point, confidence = refine_cross_point(
                self._cross_image(colours), (102.0, 95.0),
                search_radius=24, max_street_width=18,
            )
            refined.append(point)
            self.assertGreater(confidence, 0.5)
            self.assertLess(math.hypot(point[0] - 100.5, point[1] - 96.5), 1.0)
        self.assertLess(max(math.dist(refined[0], point) for point in refined[1:]), 1.0)

    def test_supplied_clip_sample_stays_near_green_target(self):
        sample = ROOT / "codex" / "sample_img" / "Clip_sample.png"
        point, confidence = refine_cross_point(sample, (66.5, 56.5))
        self.assertGreater(confidence, 0.2)
        self.assertLess(math.hypot(point[0] - 66.5, point[1] - 56.5), 4.0)


class WaferBoundaryAndMapTests(unittest.TestCase):
    def test_colour_independent_wafer_boundary(self):
        image = np.zeros((1000, 1200, 3), np.uint8)
        cv2.circle(image, (602, 498), 430, (45, 175, 210), -1, cv2.LINE_AA)
        cv2.circle(image, (602, 498), 350, (170, 55, 160), -1, cv2.LINE_AA)
        boundary = detect_wafer_boundary(image, max_dimension=800)
        self.assertLess(math.dist(boundary.center_px, (602, 498)), 4.0)
        self.assertLess(abs(boundary.radius_px - 430), 5.0)

    def test_10000_geometry_die_map_and_locate_die(self):
        angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        contour = np.rint(np.column_stack((5000 + 4700 * np.cos(angles),
                                           5000 + 4700 * np.sin(angles)))).astype(np.int32).reshape(-1, 1, 2)
        boundary = WaferBoundary(
            center_px=(5000.0, 5000.0), radius_px=4700.0,
            contour_px=contour, area_px=float(cv2.contourArea(contour)),
            bbox_px=(300, 300, 9700, 9700), method="synthetic_test",
        )
        die_map = generate_die_map(
            boundary, (10000, 10000), (5000.0, 5000.0),
            90.0, 92.0, 3.25,
        )
        self.assertGreater(die_map.num_dies, 8000)
        target = die_map.get_die(2, -3)
        self.assertIsNotNone(target)
        result = locate_die(die_map, point=target["center_px"])
        self.assertEqual(result["die_index"], (2, -3))
        self.assertTrue(result["in_wafer"])
        self.assertAlmostEqual(result["angle_deg"], 3.25)

    def test_end_to_end_and_overlays(self):
        wafer = np.zeros((1200, 1200, 3), np.uint8)
        cv2.circle(wafer, (600, 600), 520, (80, 160, 205), -1)
        clip = wafer[344:856, 344:856].copy()
        points = rotated_points(pitch_x=80.0, pitch_y=90.0, angle_deg=2.0)
        detections = [[p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2] for p in points]
        die_map = build_die_map_from_yolo(wafer, clip, detections)
        self.assertAlmostEqual(die_map.pitch_x, 80.0, places=4)
        self.assertAlmostEqual(die_map.pitch_y, 90.0, places=4)
        self.assertAlmostEqual(die_map.grid_angle_deg, 2.0, places=4)
        self.assertGreater(die_map.num_dies, 100)
        self.assertEqual(make_wafer_overlay(wafer, die_map).shape, wafer.shape)

    def test_end_to_end_memory_arrays_with_six_column_yolo(self):
        wafer = np.zeros((1200, 1200, 3), np.uint8)
        cv2.circle(wafer, (600, 600), 520, (80, 160, 205), -1)
        clip_view = wafer[344:856, 344:856]  # non-contiguous in-memory view
        points = rotated_points(pitch_x=80.0, pitch_y=90.0, angle_deg=2.0)
        detections = np.asarray([
            [0, p[0] / 512.0, p[1] / 512.0, 4.0 / 512.0, 4.0 / 512.0, 0.95]
            for p in points
        ], dtype=np.float32)
        die_map = build_die_map_from_yolo(wafer, clip_view, detections)
        self.assertAlmostEqual(die_map.pitch_x, 80.0, places=3)
        self.assertAlmostEqual(die_map.pitch_y, 90.0, places=3)
        self.assertAlmostEqual(die_map.grid_angle_deg, 2.0, places=3)
        self.assertEqual((die_map.x0, die_map.y0), (600.0, 600.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
