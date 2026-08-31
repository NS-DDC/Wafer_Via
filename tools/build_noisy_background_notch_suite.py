"""Build and validate wide/shallow notch images on varied noisy backgrounds.

The generated 2400x2400 images keep the source-space notch geometry at
105-110 px wide and 36-40 px deep.  Background noise is generated first and
the same field is used outside the wafer and inside the notch, so the notch is
a real border-connected exterior region rather than a painted inner pattern.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex.wafer_via_notch_standalone import (  # noqa: E402
    detect_wafer_notch,
    make_notch_overlay,
)


OUTPUT_DIR = ROOT / "image5" / "noisy_background_suite"
SAMPLE_DIR = ROOT / "codex" / "sample_img"
SIZE = 2400
WAFER_CENTER = np.asarray((1200.0, 1170.0), dtype=np.float64)
WAFER_RADIUS = 1050.0

CASES = (
    {
        "name": "charcoal_gaussian",
        "base": (24, 27, 31),
        "width": 105,
        "height": 36,
        "gaussian": 4.0,
        "low_frequency": 3.0,
        "gradient": (8.0, -5.0),
        "band": 2.0,
        "blobs": 20,
    },
    {
        "name": "cool_gray_gradient",
        "base": (150, 158, 166),
        "width": 106,
        "height": 37,
        "gaussian": 3.0,
        "low_frequency": 5.0,
        "gradient": (16.0, 10.0),
        "band": 2.5,
        "blobs": 28,
    },
    {
        "name": "warm_beige_banding",
        "base": (178, 202, 218),
        "width": 108,
        "height": 38,
        "gaussian": 3.5,
        "low_frequency": 4.0,
        "gradient": (-10.0, 7.0),
        "band": 6.0,
        "blobs": 24,
    },
    {
        "name": "pale_blue_mixed",
        "base": (218, 196, 166),
        "width": 110,
        "height": 40,
        "gaussian": 5.0,
        "low_frequency": 6.0,
        "gradient": (12.0, -8.0),
        "band": 4.0,
        "blobs": 38,
    },
    {
        "name": "sage_green_clouds",
        "base": (154, 190, 158),
        "width": 105,
        "height": 36,
        "gaussian": 4.0,
        "low_frequency": 9.0,
        "gradient": (-7.0, 12.0),
        "band": 2.0,
        "blobs": 34,
    },
    {
        "name": "lavender_speckle",
        "base": (208, 177, 203),
        "width": 106,
        "height": 37,
        "gaussian": 6.0,
        "low_frequency": 4.0,
        "gradient": (6.0, 8.0),
        "band": 3.0,
        "blobs": 52,
    },
    {
        "name": "dark_blue_vertical",
        "base": (82, 61, 45),
        "width": 108,
        "height": 38,
        "gaussian": 4.0,
        "low_frequency": 5.0,
        "gradient": (9.0, 5.0),
        "band": 8.0,
        "blobs": 30,
    },
    {
        "name": "bright_gray_heavy",
        "base": (205, 207, 210),
        "width": 110,
        "height": 40,
        "gaussian": 7.0,
        "low_frequency": 8.0,
        "gradient": (-14.0, 13.0),
        "band": 5.0,
        "blobs": 64,
    },
)


def _background_field(case: dict[str, object], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.asarray(case["base"], dtype=np.float32)
    image = np.broadcast_to(base, (SIZE, SIZE, 3)).copy()

    x = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)[None, :, None]
    y = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)[:, None, None]
    gx, gy = case["gradient"]
    image += x * float(gx) + y * float(gy)

    band = float(case["band"])
    if band:
        phase = np.linspace(0.0, 18.0 * math.pi, SIZE, dtype=np.float32)
        image += (np.sin(phase)[None, :, None] * band)

    low_size = 28
    low = rng.normal(0.0, 1.0, (low_size, low_size)).astype(np.float32)
    low = cv2.resize(low, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    image += low[:, :, None] * float(case["low_frequency"])

    gaussian = rng.normal(
        0.0, float(case["gaussian"]), (SIZE, SIZE, 1)
    ).astype(np.float32)
    channel_bias = rng.normal(0.0, 0.8, (SIZE, SIZE, 3)).astype(np.float32)
    image += gaussian + channel_bias
    image = np.clip(np.rint(image), 0, 255).astype(np.uint8)

    for _ in range(int(case["blobs"])):
        cx = int(rng.integers(0, SIZE))
        cy = int(rng.integers(0, SIZE))
        radius = int(rng.integers(5, 34))
        delta = rng.integers(-14, 15, size=3, dtype=np.int16)
        colour = np.clip(base.astype(np.int16) + delta, 0, 255)
        cv2.circle(
            image,
            (cx, cy),
            radius,
            tuple(int(value) for value in colour),
            -1,
            cv2.LINE_AA,
        )
    return image


def _wafer_texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = np.full((SIZE, SIZE, 3), (94, 121, 133), np.uint8)
    noise = rng.normal(0.0, 4.0, (SIZE, SIZE, 1)).astype(np.int16)
    texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in range(65, SIZE, 76):
        cv2.line(texture, (x, 0), (x, SIZE - 1), (62, 83, 72), 5, cv2.LINE_AA)
        cv2.line(texture, (x + 9, 0), (x + 9, SIZE - 1), (170, 187, 158), 3, cv2.LINE_AA)
    for y in range(55, SIZE, 68):
        cv2.line(texture, (0, y), (SIZE - 1, y), (101, 137, 112), 2, cv2.LINE_AA)
    return texture


def _draw_case(case: dict[str, object], index: int) -> np.ndarray:
    background = _background_field(case, 8200 + index)
    texture = _wafer_texture(9100 + index)
    image = background.copy()
    wafer_mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(
        wafer_mask,
        tuple(np.rint(WAFER_CENTER).astype(int)),
        int(round(WAFER_RADIUS)),
        255,
        -1,
        cv2.LINE_8,
    )
    image[wafer_mask != 0] = texture[wafer_mask != 0]
    cv2.circle(
        image,
        tuple(np.rint(WAFER_CENTER).astype(int)),
        int(round(WAFER_RADIUS)),
        (184, 204, 187),
        4,
        cv2.LINE_AA,
    )

    width = int(case["width"])
    depth = int(case["height"])
    notch_center_x = 1200.0 if width % 2 else 1199.5
    half_width = width * 0.5
    local_x = np.arange(
        int(math.floor(notch_center_x - half_width - 2)),
        int(math.ceil(notch_center_x + half_width + 2)) + 1,
    )
    normal_y = WAFER_CENTER[1] + np.sqrt(
        np.maximum(
            0.0,
            WAFER_RADIUS * WAFER_RADIUS
            - (local_x.astype(np.float64) - WAFER_CENTER[0]) ** 2,
        )
    )
    normalized = (local_x.astype(np.float64) - notch_center_x) / half_width
    raster_bias = 0.5 if width % 2 == 0 else 0.0
    boundary_y = normal_y - depth * np.sqrt(
        np.maximum(0.0, 1.0 - normalized * normalized)
    ) - raster_bias
    active_columns = 0
    maximum_rows = 0
    for position, x_value in enumerate(local_x):
        if abs(normalized[position]) > 1.0:
            continue
        first_y = int(math.ceil(boundary_y[position]))
        last_y = int(math.ceil(normal_y[position])) - 1
        if first_y <= last_y:
            image[first_y : last_y + 1, x_value] = background[
                first_y : last_y + 1, x_value
            ]
            active_columns += 1
            maximum_rows = max(maximum_rows, last_y - first_y + 1)
    if active_columns != width or maximum_rows != depth:
        raise RuntimeError(
            f"Raster mismatch for {case['name']}: "
            f"expected {width}x{depth}, got {active_columns}x{maximum_rows}"
        )
    return image


def _labelled_tile(
    image: np.ndarray,
    overlay: np.ndarray,
    case: dict[str, object],
    result,
) -> np.ndarray:
    cx = int(round(result.notch_point_px[0]))
    cy = int(round(result.notch_point_px[1]))
    half_x, half_y = 235, 165
    crop = overlay[
        max(0, cy - half_y) : min(SIZE, cy + half_y),
        max(0, cx - half_x) : min(SIZE, cx + half_x),
    ]
    crop = cv2.resize(crop, (940, 660), interpolation=cv2.INTER_NEAREST)
    tile = cv2.copyMakeBorder(
        crop, 92, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    detected_width = 2.0 * float(result.semicircle_radius_x_px)
    text1 = (
        f"{case['name']}  target={case['width']}x{case['height']}px  "
        f"detected={detected_width:.1f}x{result.semicircle_radius_y_px:.1f}px"
    )
    text2 = (
        f"angle={result.notch_angle_deg:.3f} deg  "
        f"score={result.semicircle_score:.3f}  "
        f"fit={result.semicircle_fit_residual_px:.2f}px"
    )
    for text_value, y_value in ((text1, 34), (text2, 72)):
        cv2.putText(
            tile, text_value, (16, y_value), cv2.FONT_HERSHEY_SIMPLEX,
            0.63, (255, 255, 255), 1, cv2.LINE_AA
        )
    return tile


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    for index, case in enumerate(CASES):
        image = _draw_case(case, index)
        input_path = OUTPUT_DIR / f"{index + 1:02d}_{case['name']}.jpg"
        cv2.imwrite(str(input_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        encoded = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        result = detect_wafer_notch(
            encoded,
            notch_roi_center_px=(1200, 2200),
            notch_roi_half_size_px=(240, 190),
            notch_semicircle_radius_range_px=(40, 70),
            notch_background_palette_size=5,
            notch_background_morph_px=8,
            failure_mode="error",
        )
        overlay = make_notch_overlay(encoded, result)
        overlay_path = OUTPUT_DIR / f"{index + 1:02d}_{case['name']}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(_labelled_tile(encoded, overlay, case, result))
        row = {
            "name": case["name"],
            "background_bgr": list(case["base"]),
            "target_width_px": int(case["width"]),
            "target_height_px": int(case["height"]),
            "detected_width_px": 2.0 * float(result.semicircle_radius_x_px),
            "detected_height_px": float(result.semicircle_radius_y_px),
            "angle_deg": float(result.notch_angle_deg),
            "score": float(result.semicircle_score),
            "fit_residual_px": float(result.semicircle_fit_residual_px),
            "circle_residual_px": float(result.circle_fit_residual_px),
            "palette_bgr": [list(colour) for colour in result.background_palette_bgr],
            "lab_threshold": float(result.background_distance_threshold_lab),
            "found": bool(result.found),
        }
        records.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del image, encoded, overlay, result
        gc.collect()

    sheet_rows = []
    for row_index in range(0, len(tiles), 2):
        sheet_rows.append(cv2.hconcat(tiles[row_index : row_index + 2]))
    sheet = cv2.vconcat(sheet_rows)
    cv2.imwrite(str(OUTPUT_DIR / "noisy_background_notch_results.png"), sheet)
    cv2.imwrite(
        str(SAMPLE_DIR / "noisy_background_notch_results.png"), sheet
    )
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
