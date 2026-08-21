import math
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from codex.wafer_via import (
    WaferBoundary,
    align_wafer_image,
    build_die_map_from_yolo,
    detect_wafer_boundary,
    estimate_grid_from_yolo,
    generate_die_map,
    inspect_yolo_results,
    locate_die,
    make_clip_overlay,
    make_wafer_overlay,
    parse_yolo_points,
    refine_cross_point,
    transform_point_to_aligned,
    transform_point_to_original,
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
    def test_inspect_ultralytics_style_results(self):
        class FakeBoxes(SimpleNamespace):
            def __len__(self):
                return len(self.data)

        data = np.asarray([
            [10.0, 20.0, 30.0, 40.0, 0.90, 0.0],
            [50.0, 60.0, 90.0, 100.0, 0.80, 0.0],
            [12.0, 14.0, 22.0, 24.0, 0.70, 1.0],
        ], dtype=np.float32)
        xywh = np.column_stack((
            (data[:, 0] + data[:, 2]) / 2.0,
            (data[:, 1] + data[:, 3]) / 2.0,
            data[:, 2] - data[:, 0],
            data[:, 3] - data[:, 1],
        )).astype(np.float32)
        boxes = FakeBoxes(
            data=data,
            xywh=xywh,
            xywhn=xywh / np.asarray((512, 512, 512, 512), np.float32),
            xyxy=data[:, :4],
            xyxyn=data[:, :4] / np.asarray((512, 512, 512, 512), np.float32),
            conf=data[:, 4], cls=data[:, 5], id=None,
            is_track=False, orig_shape=(512, 512),
        )
        result = SimpleNamespace(boxes=boxes, orig_shape=(512, 512), path="memory")
        output = io.StringIO()
        with redirect_stdout(output):
            summary = inspect_yolo_results([result], max_rows=2)
        printed = output.getvalue()
        self.assertIn("results length: 1", printed)
        self.assertIn("detection count: 3", printed)
        self.assertIn("xywh: shape=(3, 4)", printed)
        self.assertIn("... 1 more row(s)", printed)
        self.assertEqual(summary["results_count"], 1)
        self.assertEqual(summary["items"][0]["boxes"]["detection_count"], 3)
        self.assertEqual(summary["items"][0]["boxes"]["arrays"]["data"]["preview"].shape, (2, 6))

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
        self.assertTrue(estimate.refined)
        self.assertEqual(estimate.refinement_mode, "auto")
        self.assertEqual(estimate.raw_points_clip, estimate.points_clip)
        self.assertEqual(len(estimate.refinement_confidences), len(points))
        self.assertEqual(
            estimate.pitch_x_points_clip,
            (estimate.center_corner_clip, estimate.side_corner_clip),
        )
        self.assertEqual(
            estimate.pitch_y_points_clip,
            (estimate.center_corner_clip, estimate.below_corner_clip),
        )
        self.assertEqual(estimate.pitch_x_points_raw_clip, estimate.pitch_x_points_clip)
        self.assertIn("pitch_x_points_clip", estimate.to_dict())
        self.assertEqual(make_clip_overlay(image, estimate).shape, image.shape)

    def test_robust_angle_ignores_one_bad_center_point(self):
        image = np.zeros((512, 512, 3), np.uint8)
        points = rotated_points(angle_deg=3.25)
        points[12] += np.asarray((5.0, -5.0))

        robust = estimate_grid_from_yolo(
            image,
            points,
            detection_format="point",
            refine=False,
            angle_mode="robust",
        )
        local = estimate_grid_from_yolo(
            image,
            points,
            detection_format="point",
            refine=False,
            angle_mode="local",
            strict=False,
        )

        self.assertAlmostEqual(robust.angle_deg, 3.25, places=6)
        self.assertGreater(abs(local.angle_deg - 3.25), 3.0)
        self.assertEqual(robust.angle_mode, "robust")
        self.assertGreater(len(robust.angle_pairs_clip), 20)
        self.assertGreater(robust.angle_candidate_count, len(robust.angle_pairs_clip))
        self.assertEqual(len(robust.angle_pairs_clip), len(robust.angle_pair_axes))
        self.assertEqual(len(robust.angle_pairs_clip), len(robust.angle_pair_angles_deg))


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

    def test_corner_colour_mode_rejects_severe_noise(self):
        rng = np.random.default_rng(20260820)
        image = np.full((200, 200, 3), (185, 185, 185), np.uint8)
        colours = ((24, 65, 210), (205, 55, 45), (55, 185, 85), (165, 45, 190))
        image[:91, :95] = colours[0]
        image[:91, 106:] = colours[1]
        image[104:, :95] = colours[2]
        image[104:, 106:] = colours[3]
        noisy = np.clip(
            image.astype(np.float32) + rng.normal(0.0, 28.0, image.shape),
            0,
            255,
        ).astype(np.uint8)
        impulse_mask = rng.random(image.shape[:2]) < 0.035
        noisy[impulse_mask] = rng.integers(
            0, 256, (int(impulse_mask.sum()), 3), dtype=np.uint8
        )

        expected = (100.5, 97.0)
        for mode in ("corner_color", "auto"):
            point, confidence = refine_cross_point(
                noisy,
                (108.0, 88.0),
                search_radius=30,
                max_street_width=22,
                mode=mode,
                noise_kernel=7,
            )
            self.assertGreater(confidence, 0.5)
            self.assertLess(math.dist(point, expected), 1.0)

    def test_noisy_multicolour_grid_recovers_pitch_and_angle(self):
        rng = np.random.default_rng(20260820)
        image = np.full((512, 512, 3), 210, np.uint8)
        xs = np.asarray((84, 170, 256, 342, 428))
        ys = np.asarray((72, 164, 256, 348, 440))
        x_boundaries = [0, 127, 213, 299, 385, 512]
        y_boundaries = [0, 118, 210, 302, 394, 512]
        for row in range(5):
            for column in range(5):
                image[
                    y_boundaries[row]:y_boundaries[row + 1],
                    x_boundaries[column]:x_boundaries[column + 1],
                ] = rng.integers(25, 225, 3, dtype=np.uint8)
        for x in xs:
            image[:, x - 5:x + 6] = 210
        for y in ys:
            image[y - 5:y + 6, :] = 210
        image = np.clip(
            image.astype(np.float32) + rng.normal(0.0, 18.0, image.shape),
            0,
            255,
        ).astype(np.uint8)

        rotation = cv2.getRotationMatrix2D((256.0, 256.0), 3.0, 1.0)
        rotated = cv2.warpAffine(
            image, rotation, (512, 512), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        detections = np.asarray(
            [(x, y, 1.0) for y in ys for x in xs], dtype=np.float64
        ) @ rotation.T
        detections += rng.uniform(-7.0, 7.0, detections.shape)

        estimate = estimate_grid_from_yolo(
            rotated,
            detections,
            detection_format="point",
            refine=True,
            refine_radius=22,
            refine_mode="auto",
        )
        self.assertLess(abs(estimate.pitch_x - 86.0), 0.6)
        self.assertLess(abs(estimate.pitch_y - 92.0), 0.1)
        self.assertLess(abs(estimate.angle_deg - (-3.0)), 0.2)
        self.assertLess(math.dist(estimate.center_corner_clip, (256.0, 256.0)), 0.3)


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

    def test_edge_dies_are_clipped_but_keep_indices_outside_image(self):
        angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        contour = np.rint(np.column_stack((50 + 55 * np.cos(angles),
                                           50 + 55 * np.sin(angles)))).astype(np.int32).reshape(-1, 1, 2)
        boundary = WaferBoundary(
            center_px=(50.0, 50.0), radius_px=55.0,
            contour_px=contour, area_px=float(cv2.contourArea(contour)),
            bbox_px=(-5, -5, 105, 105), method="synthetic_edge_test",
        )
        die_map = generate_die_map(
            boundary, (100, 100), (50.0, 50.0), 40.0, 40.0, 0.0,
            include_edge=True,
        )

        edge_die = die_map.get_die(1, 0)
        self.assertIsNotNone(edge_die)  # centre x=110 is outside image and wafer
        self.assertEqual(edge_die["rect_px"], (90, 10, 130, 50))
        self.assertEqual(edge_die["crop_rect_px"][2], 100)
        self.assertTrue(edge_die["is_edge_partial"])
        self.assertTrue(edge_die["is_image_partial"])
        self.assertGreater(edge_die["wafer_area_px"], 0.0)
        self.assertGreater(edge_die["visible_area_px"], 0.0)
        self.assertTrue(all(point[0] <= 105.0 for point in edge_die["wafer_polygon_px"]))
        self.assertTrue(all(0.0 <= point[0] <= 100.0 for point in edge_die["visible_polygon_px"]))

        without_edges = generate_die_map(
            boundary, (100, 100), (50.0, 50.0), 40.0, 40.0, 0.0,
            include_edge=False,
        )
        self.assertIsNone(without_edges.get_die(1, 0))

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
        self.assertIsNotNone(die_map.pitch_x_points_full)
        self.assertIsNotNone(die_map.pitch_y_points_full)
        self.assertTrue(np.allclose(die_map.pitch_x_points_full[0], (die_map.x0, die_map.y0)))
        self.assertAlmostEqual(
            math.dist(*die_map.pitch_x_points_full), die_map.pitch_x, places=6
        )
        self.assertAlmostEqual(
            math.dist(*die_map.pitch_y_points_full), die_map.pitch_y, places=6
        )
        self.assertGreater(len(die_map.angle_pairs_full), 10)
        self.assertEqual(
            len(die_map.angle_pairs_full),
            len(die_map.grid_estimate.angle_pairs_clip),
        )
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

    def test_manual_pitch_size_overrides_detected_pitch(self):
        wafer = np.zeros((1200, 1200, 3), np.uint8)
        cv2.circle(wafer, (600, 600), 520, (80, 160, 205), -1)
        clip = wafer[344:856, 344:856].copy()
        detections = np.asarray(rotated_points(pitch_x=80.0, pitch_y=90.0, angle_deg=2.0))

        die_map = build_die_map_from_yolo(
            wafer,
            clip,
            detections,
            detection_format="point",
            refine=False,
            pitch_size=(81.25, 93.5),
            return_aligned_image=False,
        )

        self.assertEqual(die_map.pitch_source, "manual")
        self.assertAlmostEqual(die_map.pitch_x, 81.25)
        self.assertAlmostEqual(die_map.pitch_y, 93.5)
        self.assertAlmostEqual(die_map.detected_pitch_x, 80.0, places=5)
        self.assertAlmostEqual(die_map.detected_pitch_y, 90.0, places=5)
        self.assertAlmostEqual(die_map.grid_estimate.pitch_x, 80.0, places=5)
        self.assertAlmostEqual(die_map.grid_estimate.pitch_y, 90.0, places=5)

    def test_grid_origin_uses_wafer_center_not_image_center(self):
        wafer = np.zeros((1200, 1200, 3), np.uint8)
        cv2.circle(wafer, (650, 600), 500, (80, 160, 205), -1)
        clip = wafer[344:856, 344:856].copy()  # image-centred 512x512 clip
        detections = np.asarray([
            (256.0 + 80.0 * ix, 256.0 + 90.0 * iy)
            for iy in range(-2, 3)
            for ix in range(-2, 3)
        ])

        die_map = build_die_map_from_yolo(
            wafer,
            clip,
            detections,
            detection_format="point",
            refine=False,
            return_aligned_image=False,
        )

        self.assertEqual((die_map.wafer_cx, die_map.wafer_cy), (650, 600))
        self.assertEqual(die_map.grid_estimate.center_corner_clip, (336.0, 256.0))
        self.assertEqual((die_map.x0, die_map.y0), (680.0, 600.0))
        self.assertLess(
            math.dist((die_map.x0, die_map.y0), (650.0, 600.0)),
            math.dist((600.0, 600.0), (650.0, 600.0)),
        )

    def test_aligned_image_and_coordinate_round_trip(self):
        wafer = np.zeros((1200, 1200, 3), np.uint8)
        cv2.circle(wafer, (600, 600), 520, (80, 160, 205), -1)
        clip = wafer[344:856, 344:856]
        points = rotated_points(pitch_x=80.0, pitch_y=90.0, angle_deg=2.0)
        boxes = np.asarray([[p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2] for p in points])
        die_map = build_die_map_from_yolo(wafer, clip, boxes, detection_format="xyxy")

        self.assertIsNotNone(die_map.aligned_image)
        self.assertEqual(die_map.aligned_image.shape, wafer.shape)
        self.assertIsNotNone(die_map.original_to_aligned_matrix)
        self.assertIsNotNone(die_map.aligned_to_original_matrix)

        center_original = (600.0, 600.0)
        side_clip = points[2 * 5 + 3]
        side_original = (344.0 + float(side_clip[0]), 344.0 + float(side_clip[1]))
        center_aligned = transform_point_to_aligned(die_map, center_original)
        side_aligned = transform_point_to_aligned(die_map, side_original)
        self.assertAlmostEqual(center_aligned[1], side_aligned[1], places=6)
        self.assertAlmostEqual(side_aligned[0] - center_aligned[0], 80.0, places=5)

        original = (777.25, 433.75)
        round_trip = transform_point_to_original(
            die_map, transform_point_to_aligned(die_map, original)
        )
        self.assertTrue(np.allclose(round_trip, original, atol=1e-8))

        aligned, forward, inverse = align_wafer_image(
            wafer, (die_map.wafer_cx, die_map.wafer_cy), die_map.grid_angle_deg
        )
        self.assertEqual(aligned.shape, wafer.shape)
        identity = np.vstack((forward, (0.0, 0.0, 1.0))) @ np.vstack((inverse, (0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(identity, np.eye(3), atol=1e-10))

        without_image = build_die_map_from_yolo(
            wafer, clip, boxes, detection_format="xyxy", return_aligned_image=False
        )
        self.assertIsNone(without_image.aligned_image)
        self.assertIsNotNone(without_image.original_to_aligned_matrix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
