"""Reproducible synthetic fixtures and timing data for point-refinement modes.

Run from the repository root, for example:
    python -B tools/probe_cross_lines.py --output probe-results.json
No output file is written unless --output is supplied. These are synthetic
checks, not validation on a production camera frame or detector output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex.wafer_via import refine_cross_point


def make_fixtures() -> list[dict[str, Any]]:
    """Return fresh BGR uint8 images, model seeds, and known geometric truth.

    The white-quadrant fixture deliberately makes a whole die quadrant the
    same white as the street, not merely a detached white reference patch.
    Drawing the cross last preserves its known width and centre.
    """
    size, centre = 201, 100.0
    seed = (102.0, 98.0)
    green = (30, 150, 45)
    base = np.full((size, size, 3), green, np.uint8)
    base[:, 96:105] = 255
    base[96:105, :] = 255

    white_corner = base.copy()
    white_corner[:96, :96] = 255

    dark = np.full_like(base, 245)
    dark[:, 96:105] = (20, 50, 25)
    dark[96:105, :] = (20, 50, 25)

    rng = np.random.default_rng(20261027)
    noisy = np.clip(base.astype(np.float32) + rng.normal(0.0, 10.0, base.shape), 0, 255).astype(np.uint8)
    impulses = rng.random(base.shape[:2]) < 0.01
    noisy[impulses] = rng.integers(0, 256, (int(impulses.sum()), 3), dtype=np.uint8)

    distracting = np.full_like(base, green)
    distracting[:, 90:96] = (190, 250, 190)
    distracting[90:96, :] = (190, 250, 190)
    distracting[80:91, 80:91] = 255
    distracting[:, 96:105] = 255
    distracting[96:105, :] = 255

    return [
        {"name": "white_corner_white_cross", "image": white_corner, "seed": seed,
         "truth": (centre, centre), "description": "White upper-left die and white 9px crossing streets"},
        {"name": "dark_cross_white", "image": dark, "seed": seed,
         "truth": (centre, centre), "description": "Dark 9px crossing streets on white dies"},
        {"name": "noisy", "image": noisy, "seed": seed,
         "truth": (centre, centre), "description": "White cross; Gaussian sigma 10 and 1% impulse noise, seed 20261027"},
        {"name": "distractinggreen", "image": distracting, "seed": seed,
         "truth": (centre, centre), "description": "Adjacent pale-green bands and detached white corner patch; true cross drawn last"},
        {"name": "flatwhite", "image": np.full_like(base, 255), "seed": seed,
         "truth": None, "description": "No visible contrast; required result is the unchanged model seed and confidence zero"},
    ]


def measure_fixture(fixture: dict[str, Any], mode: str, **parameters: Any) -> dict[str, Any]:
    """Measure one actual Python invocation; confidence is a heuristic score."""
    started = time.perf_counter()
    point, confidence = refine_cross_point(fixture["image"], fixture["seed"], mode=mode, **parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    truth = fixture.get("truth")
    return {
        "fixture": fixture["name"], "mode": mode, "parameters": parameters,
        "point": list(point), "seed": list(fixture["seed"]),
        "truth": list(truth) if truth is not None else None,
        "error_px": math.dist(point, truth) if truth is not None else None,
        "movement_px": math.dist(point, fixture["seed"]),
        "confidence": confidence, "elapsed_ms": elapsed_ms,
        "abstained": confidence == 0.0 and tuple(point) == tuple(fixture["seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON results file")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--max-street-width", type=int, default=16)
    parser.add_argument("--noise-kernel", type=int, default=5)
    parser.add_argument("--rotations", action="store_true", help="Include -10 and +10 degree rotated fixtures")
    args = parser.parse_args()
    fixtures = make_fixtures()
    if args.rotations:
        rotated = []
        for fixture in fixtures:
            if fixture["truth"] is None:
                continue
            for angle in (-10.0, 10.0):
                sample = dict(fixture)
                sample["name"] = fixture["name"] + f"_rotation_{angle:+g}"
                matrix = cv2.getRotationMatrix2D(fixture["truth"], angle, 1.0)
                sample["image"] = cv2.warpAffine(fixture["image"], matrix, (201, 201), flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_REFLECT_101)
                rotated.append(sample)
        fixtures.extend(rotated)
    # Warm up OpenCV's Lab conversion so per-row latency excludes its one-time LUT.
    cv2.cvtColor(np.zeros((5, 5, 3), np.uint8), cv2.COLOR_BGR2LAB)
    parameters = dict(search_radius=args.radius, max_street_width=args.max_street_width,
                      noise_kernel=args.noise_kernel)
    results = [measure_fixture(fixture, mode, **parameters) for fixture in fixtures
               for mode in ("auto", "gradient", "corner_color", "cross_lines")]
    report = {"evidence": "Synthetic images only; production image not supplied", "results": results}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(str(args.output.resolve()))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
