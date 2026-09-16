"""Reproduce lookup/crop coordinate risks without modifying the algorithm."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codex import wafer_via_notch_standalone as module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location("lookup_baseline", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)

    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    contour = np.rint(np.column_stack((
        5000 + 4700 * np.cos(angles), 5000 + 4700 * np.sin(angles)
    ))).astype(np.int32).reshape(-1, 1, 2)
    boundary = module.WaferBoundary((5000.0, 5000.0), 4700.0, contour)
    summary = {"evidence": "synthetic geometry, not a supplied failing production crop"}
    audits = []
    for angle in (0.0, 3.25):
        dm = module.generate_die_map(
            boundary, (10000, 10000), (5013.17, 4992.31), 90.2, 92.15, angle
        )
        mismatches = dict(index=0, die_rect=0, crop_rect=0, baseline=0)
        for entry in dm.dies:
            result = module.locate_die(dm, point=entry["center_px"])
            mismatches["index"] += result["die_index"] != entry["index"]
            mismatches["die_rect"] += result["die_rect_px"] != entry["rect_px"]
            mismatches["crop_rect"] += result["crop_rect_px"] != entry["crop_rect_px"]
            if baseline:
                mismatches["baseline"] += result != baseline.locate_die(dm, point=entry["center_px"])
        if baseline:
            rng = np.random.default_rng(16)
            for _ in range(300):
                x, y = rng.uniform(-1000, 11000, 2)
                width, height = rng.uniform(1, 500, 2)
                bbox = (x, y, x + width, y + height)
                mismatches["baseline"] += module.locate_die(dm, bbox=bbox) != baseline.locate_die(dm, bbox=bbox)
        audits.append(dict(angle=angle, centers_checked=dm.num_dies, random_bboxes_checked=300 if baseline else 0, mismatches=mismatches))
    summary["lookup_audits"] = audits

    dm = module.generate_die_map(
        boundary, (10000, 10000), (5013.17, 4992.31), 90.2, 92.15, 0.0
    )
    boundary_cases = []
    for index in range(1, 41):
        point = (dm.x0 + index * dm.pitch_x, dm.y0 - dm.pitch_y / 2)
        quotient = (point[0] - dm.x0) / dm.pitch_x
        hit = module.locate_die(dm, point=point)
        if math.floor(quotient) != index:
            boundary_cases.append(dict(
                nominal_right_cell=index, quotient=quotient,
                actual_cell=hit["die_index"], point=point,
                all_cells=[h["die_index"] for h in module.locate_die(dm, point=point, return_all=True)],
            ))
    summary["floating_boundary_examples"] = boundary_cases[:3]

    rounded_boundary = dm.x0 + dm.pitch_x
    rounded_hit = module.locate_die(dm, point=(round(rounded_boundary), dm.y0 - 20))
    summary["rounded_overlay_boundary"] = dict(
        exact_x=rounded_boundary, rounded_x=round(rounded_boundary),
        rounded_lookup_cell=rounded_hit["die_index"], nominal_right_cell=(1, 0),
    )

    correction_deg = 0.2
    matrix = cv2.getRotationMatrix2D((5000.0, 5000.0), correction_deg, 1.0)
    inverse = cv2.invertAffineTransform(matrix)
    aligned = (dm.x0 + 2 * dm.pitch_x + 5, dm.y0 + 40 * dm.pitch_y + 20)
    original = inverse @ np.asarray((*aligned, 1.0))
    converted = matrix @ np.asarray((*original, 1.0))
    summary["coordinate_space_example"] = dict(
        correction_deg=correction_deg, grid_angle_deg=dm.grid_angle_deg,
        aligned_point=aligned, original_point=original.tolist(),
        delta_x_px=float(converted[0] - original[0]),
        correct_index=module.locate_die(dm, point=tuple(converted))["die_index"],
        mixed_space_index=module.locate_die(dm, point=tuple(original))["die_index"],
    )
    summary["integer_pitch_drift"] = dict(
        pitch_x=dm.pitch_x, die_w=dm.die_w, columns=40,
        exact_boundary_x=dm.x0 + 40 * dm.pitch_x,
        rounded_pitch_boundary_x=dm.x0 + 40 * dm.die_w,
        drift_px=40 * (dm.die_w - dm.pitch_x),
    )

    sample = dm.get_die(2, -3)
    hit = module.locate_die(dm, point=sample["center_px"])
    x1, y1, x2, y2 = hit["crop_rect_px"]
    x_encoded = np.broadcast_to(np.arange(10000, dtype=np.uint16), (10000, 10000))
    crop = x_encoded[y1:y2, x1:x2]
    summary["stored_crop_slice"] = dict(
        rect=(x1, y1, x2, y2), shape=crop.shape,
        first_pixel_x=int(crop[0, 0]), last_pixel_x=int(crop[-1, -1]),
        matches_stored_rect=sample["crop_rect_px"] == hit["crop_rect_px"],
    )
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
