"""Build wide, shallow notch variants from the four image5 wafer samples.

The original local notch silhouette is measured from the border-connected
background.  Each generated indentation uses a different requested size.  The
configured values are expressed in the 5000x5000 angle-result coordinate
system and doubled while drawing into the 10000x10000 source.  Existing files
are preserved; outputs use the ``*_wide_shallow_10000x10000.png`` suffix.
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
TARGET_SIZE_AT_5000 = {
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
    background = np.median(samples, axis=0).astype(np.int16)
    distance = np.linalg.norm(image.astype(np.int16) - background, axis=2)
    candidate = (distance <= 6.0).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    labels = np.unique(
        np.concatenate(
            (
                components[0, :],
                components[-1, :],
                components[:, 0],
                components[:, -1],
            )
        )
    )
    labels = labels[labels > 0]
    if not len(labels):
        raise RuntimeError("No background component reaches the image border.")
    label = int(max(labels, key=lambda value: stats[int(value), cv2.CC_STAT_AREA]))
    mask = components == label
    return mask, tuple(int(value) for value in background)


def _top_background_y(mask: np.ndarray, xs: np.ndarray, center_y: float) -> np.ndarray:
    result = np.full(len(xs), np.nan, dtype=np.float64)
    minimum_y = max(0, int(math.floor(center_y)))
    for index, x in enumerate(xs):
        values = np.flatnonzero(mask[minimum_y:, int(x)])
        if len(values):
            result[index] = float(values[0] + minimum_y)
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
        (name for name in TARGET_SIZE_AT_5000 if name in source_path.stem), None
    )
    if variant_name is None:
        raise ValueError(f"No target size is configured for {source_path.name}.")
    target_width_at_5000, target_height_at_5000 = TARGET_SIZE_AT_5000[variant_name]
    source_to_result_scale = 5000.0 / float(max(height, width))
    target_width = target_width_at_5000 / source_to_result_scale
    target_depth = target_height_at_5000 / source_to_result_scale
    target_half_width = target_width * 0.5
    notch_center_x = (old_left + old_right) * 0.5

    output = image.copy()
    output_exterior = exterior.copy()
    local_left = max(0, int(math.floor(notch_center_x - target_half_width - 4.0)))
    local_right = min(width, int(math.ceil(notch_center_x + target_half_width + 4.0)) + 1)
    local_top = max(0, int(math.floor(center_y + radius - maximum_depth - 16.0)))
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

    # Reconstruct pixels that belonged to the old deep notch but are wafer in
    # the new shallow notch.  Samples are taken from the corresponding old arc
    # using normalized horizontal position and inward distance.
    old_x = notch_center_x + normalized_x * old_half_width
    old_x_clipped = np.clip(old_x, xs[0], xs[-1])
    old_boundary = np.interp(old_x_clipped, xs.astype(np.float64), old_boundary_y)
    background_array = np.asarray(background_bgr, dtype=np.float32)
    for y in range(local_top, local_bottom):
        desired_exterior = inside_notch_width & (float(y) >= target_boundary_y)
        original_exterior = exterior[y, local_left:local_right]
        newly_wafer = (~desired_exterior) & original_exterior & inside_notch_width
        if np.any(newly_wafer):
            inward = np.maximum(2.0, target_boundary_y - float(y))
            source_y = np.clip(
                np.rint(old_boundary - inward).astype(np.int32), 0, height - 1
            )
            source_x = np.clip(np.rint(old_x).astype(np.int32), 0, width - 1)
            output[y, local_left:local_right][newly_wafer] = image[
                source_y[newly_wafer], source_x[newly_wafer]
            ]
        output[y, local_left:local_right][desired_exterior] = background_array
        output_exterior[y, local_left:local_right] = desired_exterior

    # Two-pixel antialiasing along the analytic semi-ellipse.
    for local_index, x in enumerate(range(local_left, local_right)):
        if not inside_notch_width[local_index]:
            continue
        boundary_y = float(target_boundary_y[local_index])
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
        "new_width_px": target_width,
        "new_depth_px": target_depth,
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
