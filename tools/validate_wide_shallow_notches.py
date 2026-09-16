"""Validate the exact source-image notch sizes and write 5000px overlays."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex.wafer_via_notch_standalone import (  # noqa: E402
    detect_wafer_notch,
    make_notch_background_debug_contact_sheet,
    make_notch_overlay,
)


IMAGE_DIR = ROOT / "image5"
SAMPLE_DIR = ROOT / "codex" / "sample_img"
VARIANTS = ("black", "gray", "pale_green", "pale_red")
TARGETS = {
    "black": (105, 36),
    "gray": (106, 37),
    "pale_green": (108, 38),
    "pale_red": (110, 40),
}


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    thumbnails = []
    source_tiles = []
    preview = None

    for name in VARIANTS:
        path = IMAGE_DIR / f"wafer_{name}_wide_shallow_10000x10000.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        source_crop = image[9250:9950, 4600:5400]
        source_crop = cv2.resize(
            source_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(str(IMAGE_DIR / f"debug_{name}_source_notch.png"), source_crop)
        target_width, target_height = TARGETS[name]
        source_center_x = 4996 if name == "black" else (4995 if name == "gray" else 4994)
        raw_crop = image[9500:9800, source_center_x - 250 : source_center_x + 250]
        tile = cv2.copyMakeBorder(
            raw_crop, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
        )
        source_label = f"{name}: raster={target_width}x{target_height}px (source 1:1)"
        cv2.putText(
            tile, source_label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.line(tile, (382, 40), (482, 40), (0, 220, 255), 2, cv2.LINE_8)
        cv2.putText(
            tile, "100 px", (400, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 220, 255), 1, cv2.LINE_AA
        )
        source_tiles.append(tile)
        result = detect_wafer_notch(
            image,
            notch_roi_center_px=(5000, 9650),
            notch_roi_half_size_px=(600, 600),
            notch_semicircle_radius_range_px=(30, 80),
            notch_background_morph_px=8,
            failure_mode="error",
        )
        overlay = make_notch_overlay(image, result, max_dimension=5000)
        overlay_path = IMAGE_DIR / f"angle_result_{name}_5000x5000.jpg"
        if not cv2.imwrite(
            str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise RuntimeError(f"Failed to write {overlay_path}")

        visual_scale = 5000.0 / max(image.shape[:2])
        cx = int(round(result.notch_point_px[0] * visual_scale))
        cy = int(round(result.notch_point_px[1] * visual_scale))
        half = 270
        zoom = overlay[
            max(0, cy - half) : min(overlay.shape[0], cy + half),
            max(0, cx - half) : min(overlay.shape[1], cx + half),
        ]
        thumb = cv2.resize(zoom, (1100, 1100), interpolation=cv2.INTER_AREA)
        detected_width = 2.0 * result.semicircle_radius_x_px
        detected_height = result.semicircle_radius_y_px
        cv2.rectangle(thumb, (0, 0), (1100, 92), (20, 20, 20), -1)
        label = (
            f"{name}  source target={target_width}x{target_height}px  "
            f"detected={detected_width:.1f}x{detected_height:.1f}px"
        )
        cv2.putText(
            thumb,
            label,
            (22, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            thumb,
            label,
            (22, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
        if name in {"black", "pale_green"}:
            debug = make_notch_background_debug_contact_sheet(
                image,
                notch_roi_center_px=(5000, 9650),
                notch_roi_half_size_px=(600, 600),
                notch_semicircle_radius_range_px=(30, 80),
                background_morph_px=8,
            )
            cv2.imwrite(
                str(IMAGE_DIR / f"debug_{name}_background_stages.png"), debug
            )
            if name == "pale_green":
                preview = zoom.copy()
                cv2.imwrite(
                    str(SAMPLE_DIR / "notch_roi_background_stages.png"), debug
                )

        results.append(
            {
                "variant": name,
                "found": bool(result.found),
                "angle_deg": float(result.notch_angle_deg),
                "correction_deg": float(result.correction_angle_deg),
                "detected_source_width_px": float(
                    2.0 * result.semicircle_radius_x_px
                ),
                "detected_source_height_px": float(
                    result.semicircle_radius_y_px
                ),
                "score": float(result.semicircle_score),
                "fit_residual_px": float(result.semicircle_fit_residual_px),
                "method": result.detection_method,
                "overlay_shape": list(overlay.shape),
            }
        )
        del image, overlay, zoom, result
        gc.collect()

    top = cv2.hconcat(thumbnails[:2])
    bottom = cv2.hconcat(thumbnails[2:])
    sheet = cv2.vconcat((top, bottom))
    cv2.imwrite(str(SAMPLE_DIR / "wide_shallow_notch_results.png"), sheet)
    cv2.imwrite(str(IMAGE_DIR / "wide_shallow_notch_contact_sheet.png"), sheet)
    source_sheet = cv2.vconcat(
        (cv2.hconcat(source_tiles[:2]), cv2.hconcat(source_tiles[2:]))
    )
    cv2.imwrite(
        str(SAMPLE_DIR / "wide_shallow_notch_source_1to1.png"), source_sheet
    )
    cv2.imwrite(
        str(IMAGE_DIR / "wide_shallow_notch_source_1to1.png"), source_sheet
    )
    if preview is not None:
        cv2.imwrite(str(SAMPLE_DIR / "notch_roi_semicircle_preview.png"), preview)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
