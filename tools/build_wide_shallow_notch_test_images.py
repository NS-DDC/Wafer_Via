"""Build wide, shallow notch variants from the four image5 wafer samples.

The original local notch silhouette is measured from the border-connected
background.  Each generated indentation uses a different requested size.  The
configured values are expressed in the original 10000x10000 source-image
coordinate system.  Therefore the 5000x5000 angle-result image shows each
notch at one-half of the configured width and height.  Existing files are
preserved; outputs use the ``*_wide_shallow_10000x10000.png`` suffix.
"""

from pathlib import Path
import math
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codex.wafer_via_notch_standalone import detect_wafer_notch


IMAGE_DIR = ROOT / "image5"
SOURCE_PATTERN = "wafer_*_10000x10000.png"
TARGET_SIZE_IN_SOURCE = {
    "black": (105.0, 36.0),
    "gray": (106.0, 37.0),
    "pale_green": (108.0, 38.0),
    "pale_red": (110.0, 40.0),
}


def _border_connected_background(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    height, width = image.shape[:2]
    corner = max(20, min(height, width) // 100)
    samples = np.concatenate(
        (
            image[:corner, :corner].reshape(-1, 3),
            image[:corner, width - corner :].reshape(-1, 3),
            image[height - corner :, :corner].reshape(-1, 3),
            image[height - corner :, width - corner :].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(samples, axis=0).astype(np.uint8)
    # Avoid a 10000x10000x3 int16 temporary (>570 MiB).  A per-channel range
    # is sufficient here because these controlled samples have a flat exterior.
    lower = np.clip(background.astype(np.int16) - 4, 0, 255).astype(np.uint8)
    upper = np.clip(background.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    candidate = cv2.inRange(image, lower, upper)
    # The controlled source samples have a flat background.  Keeping the
    # threshold mask directly avoids a 400 MiB connected-component label map.
    mask = candidate != 0
    return mask, tuple(int(value) for value in background)


def _top_background_y(mask: np.ndarray, xs: np.ndarray, center_y: float) -> np.ndarray:
    result = np.full(len(xs), np.nan, dtype=np.float64)
    minimum_y = max(0, int(math.floor(center_y)))
    for index, x in enumerate(xs):
        # Find the upper edge of the exterior connected to the bottom.  Using
        # the last non-background pixel ignores background-colored die noise
        # inside the wafer that would otherwise look like an early boundary.
        non_background = np.flatnonzero(~mask[minimum_y:, int(x)])
        if len(non_background):
            result[index] = float(non_background[-1] + minimum_y + 1)
        else:
            result[index] = float(minimum_y)
    return result


def build_variant(source_path: Path) -> tuple[Path, dict[str, float]]:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(source_path))
    height, width = image.shape[:2]
    notch = detect_wafer_notch(
        image,
        notch_roi_center_px=(width * 0.5, height * 0.965),
        notch_roi_half_size_px=(width * 0.06, height * 0.06),
        failure_mode="error",
    )
    exterior, background_bgr = _border_connected_background(image)
    center_x, center_y = notch.wafer_center_px
    radius = float(notch.wafer_radius_px)
    roi_half_width = max(100, int(round(width * 0.06)))
    xs = np.arange(
        max(0, int(round(center_x)) - roi_half_width),
        min(width, int(round(center_x)) + roi_half_width + 1),
        dtype=np.int32,
    )
    circle_y = center_y + np.sqrt(
        np.maximum(0.0, radius * radius - (xs.astype(np.float64) - center_x) ** 2)
    )
    old_boundary_y = _top_background_y(exterior, xs, center_y)
    intrusion = circle_y - old_boundary_y
    valid = np.isfinite(intrusion)
    maximum_depth = float(np.nanmax(intrusion[valid]))
    active = valid & (intrusion >= max(10.0, maximum_depth * 0.08))
    active_indices = np.flatnonzero(active)
    if len(active_indices) < 20:
        raise RuntimeError(f"Could not measure the original notch in {source_path.name}.")
    old_left = float(xs[active_indices[0]])
    old_right = float(xs[active_indices[-1]])
    old_width = old_right - old_left + 1.0
    old_half_width = old_width * 0.5
    variant_name = next(
        (name for name in TARGET_SIZE_IN_SOURCE if name in source_path.stem), None
    )
    if variant_name is None:
        raise ValueError(f"No target size is configured for {source_path.name}.")
    target_width, target_depth = TARGET_SIZE_IN_SOURCE[variant_name]
    source_to_result_scale = 5000.0 / float(max(height, width))
    target_half_width = target_width * 0.5
    notch_center_x = (old_left + old_right) * 0.5

    output = image.copy()
    # The source samples contain a decorative/deep semicircular structure
    # around the old notch.  Remove the whole structure so the only remaining
    # semicircle is the actual outer-edge notch being validated.
    edit_half_width = max(450.0, old_half_width * 2.25)
    local_left = max(0, int(math.floor(notch_center_x - edit_half_width)))
    local_right = min(width, int(math.ceil(notch_center_x + edit_half_width)) + 1)
    nominal_bottom = center_y + radius
    texture_shift = 650
    local_top = max(0, int(math.floor(nominal_bottom - texture_shift)))
    local_bottom = min(height, int(math.ceil(center_y + radius + 8.0)) + 1)

    x_values = np.arange(local_left, local_right, dtype=np.float64)
    dx = x_values - notch_center_x
    normal_y = center_y + np.sqrt(
        np.maximum(0.0, radius * radius - (x_values - center_x) ** 2)
    )
    normalized_x = dx / max(target_half_width, 1.0)
    ellipse = np.sqrt(np.maximum(0.0, 1.0 - normalized_x * normalized_x))
    target_boundary_y = normal_y - target_depth * ellipse
    inside_notch_width = np.abs(normalized_x) <= 1.0
    desired_boundary_y = np.where(
        inside_notch_width, target_boundary_y, normal_y
    )

    # Extend a normal grid strip vertically to the wafer edge.  A feathered
    # rectangle removes the old inner arch without introducing another curved
    # feature that the notch detector could accidentally fit.
    pixel_index = np.arange(len(x_values), dtype=np.float32)
    side_distance = np.minimum(pixel_index, pixel_index[::-1])
    horizontal_alpha = np.clip(side_distance / 48.0, 0.0, 1.0)
    background_array = np.asarray(background_bgr, dtype=np.float32)
    for y in range(local_top, local_bottom):
        desired_exterior = float(y) >= desired_boundary_y
        desired_wafer = ~desired_exterior
        source_y = max(0, y - texture_shift)
        vertical_alpha = float(np.clip((y - local_top) / 48.0, 0.0, 1.0))
        alpha = (horizontal_alpha * vertical_alpha)[:, None]
        row = output[y, local_left:local_right].astype(np.float32)
        texture = image[source_y, local_left:local_right].astype(np.float32)
        blended = row * (1.0 - alpha) + texture * alpha
        row[desired_wafer] = blended[desired_wafer]
        row[desired_exterior] = background_array
        output[y, local_left:local_right] = np.clip(
            np.rint(row), 0, 255
        ).astype(np.uint8)

    # Two-pixel antialiasing along the analytic semi-ellipse.
    for local_index, x in enumerate(range(local_left, local_right)):
        boundary_y = float(desired_boundary_y[local_index])
        first = max(local_top, int(math.floor(boundary_y)) - 1)
        last = min(local_bottom, int(math.ceil(boundary_y)) + 2)
        for y in range(first, last):
            exterior_weight = float(np.clip(y - boundary_y + 0.5, 0.0, 1.0))
            if 0.0 < exterior_weight < 1.0:
                value = (
                    output[y, x].astype(np.float32) * (1.0 - exterior_weight)
                    + background_array * exterior_weight
                )
                output[y, x] = np.clip(np.rint(value), 0, 255).astype(np.uint8)

    output_path = source_path.with_name(
        source_path.stem.replace("_10000x10000", "_wide_shallow_10000x10000")
        + source_path.suffix
    )
    if not cv2.imwrite(
        str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ):
        raise RuntimeError(f"Failed to write {output_path}.")
    return output_path, {
        "old_width_px": old_width,
        "old_depth_px": maximum_depth,
        "source_width_px": target_width,
        "source_height_px": target_depth,
        "result_width_px": target_width * source_to_result_scale,
        "result_height_px": target_depth * source_to_result_scale,
        "center_x_px": notch_center_x,
    }


def main() -> None:
    sources = sorted(
        path
        for path in IMAGE_DIR.glob(SOURCE_PATTERN)
        if "_wide_shallow_" not in path.name
    )
    if not sources:
        raise FileNotFoundError(f"No sources matched {IMAGE_DIR / SOURCE_PATTERN}")
    for source in sources:
        output, metrics = build_variant(source)
        print(output.name, metrics)


if __name__ == "__main__":
    main()
