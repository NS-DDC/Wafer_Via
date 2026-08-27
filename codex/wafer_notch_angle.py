"""Colour-independent wafer notch detection and angle alignment.

The detector fits the wafer's original outer circle, then measures where the
corner/background colour remains connected while penetrating inward through
that circle. Candidate depth, angular width, and area are scored together, so
the notch may be sharp, rounded, or a long shallow semicircle.

The angle reference is the vector from the fitted wafer centre to the midpoint
of the original outer circle across the notch opening. The deepest point stays
available as a diagnostic only. Image-space angles are clockwise: right=0,
down=90.
``correction_angle_deg`` is the OpenCV rotation that moves the notch to the
configured reference direction (bottom/90 degrees by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]

__all__ = [
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "make_notch_overlay",
    "make_notch_zoom",
]


@dataclass(frozen=True)
class NotchAngleResult:
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_point_px: Point
    notch_deepest_point_px: Point
    notch_angle_deg: float
    reference_angle_deg: float
    correction_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    notch_width_px: float
    confidence: float
    radial_noise_px: float
    candidate_arc_px: Tuple[Point, ...]
    wafer_contour_px: np.ndarray
    segmentation_threshold: float
    scale: float
    failure_mode: str


def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported image shape: {image.shape}")
    loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(str(image))
    return loaded


def _normalise_angle(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _background_distance_mask(image_bgr: np.ndarray):
    height, width = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch = max(3, int(round(min(height, width) * 0.008)))
    corner_pixels = np.concatenate(
        (
            lab[:patch, :patch].reshape(-1, 3),
            lab[:patch, -patch:].reshape(-1, 3),
            lab[-patch:, :patch].reshape(-1, 3),
            lab[-patch:, -patch:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(corner_pixels, axis=0)
    corner_distance = np.linalg.norm(corner_pixels - background, axis=1)
    threshold = max(4.0, float(np.percentile(corner_distance, 99.5) + 2.0))
    distance = np.linalg.norm(lab - background, axis=2)
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, threshold


def _select_wafer_contour(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    image_area = float(height * width)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    best = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if not 0.15 <= area_ratio <= 0.96:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        centre_error = math.hypot(cx - width / 2.0, cy - height / 2.0)
        centre_error /= max(1.0, math.hypot(width, height) / 2.0)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
        fill = area / max(math.pi * radius * radius, 1.0)
        score = 2.0 * area_ratio + circularity + fill - centre_error
        if best is None or score > best[0]:
            best = (score, contour)
    if best is None:
        raise RuntimeError(
            "Wafer boundary was not found; check that the image corners show background."
        )
    return best[1]


def _circular_gaussian(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    blurred = cv2.GaussianBlur(
        extended.reshape(1, -1), (kernel_size, 1), 0
    ).reshape(-1)
    return blurred[half:half + len(values)]


def _circular_median(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    windows = np.lib.stride_tricks.sliding_window_view(extended, kernel_size)
    return np.median(windows, axis=1).astype(np.float32)


def _radial_background_penetration(
    wafer_mask: np.ndarray,
    center: Point,
    radius: float,
    *,
    angle_samples: int,
    radial_inner_ratio: float,
):
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    radial_samples = max(96, int(round(radius * (1.0 - radial_inner_ratio) * 2.5)))
    radii = np.linspace(
        radius * float(radial_inner_ratio), radius + 2.0, radial_samples
    )
    map_x = (
        float(center[0]) + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        float(center[1]) + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    foreground = cv2.remap(
        wafer_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32) / 255.0

    # A die street or a single dark rim pixel must not look like outside
    # background. Majority voting across neighbouring angles keeps only a
    # continuous black region entering through the wafer edge.
    samples_per_degree = angle_samples / 360.0
    angular_kernel = max(3, int(round(0.5 * samples_per_degree)) | 1)
    foreground = cv2.blur(foreground, (1, angular_kernel)) >= 0.50
    inside = radii <= radius
    inside_radii = radii[inside][::-1]
    outer_to_inner = foreground[:, inside][:, ::-1]
    has_foreground = np.any(outer_to_inner, axis=1)
    first_foreground = np.argmax(outer_to_inner, axis=1)
    boundary_radius = np.where(
        has_foreground,
        inside_radii[first_foreground],
        radius * float(radial_inner_ratio),
    )
    penetration = (radius - boundary_radius).astype(np.float32)
    penetration = _circular_median(
        penetration, max(3, int(round(0.7 * samples_per_degree)) | 1)
    )
    return angles, penetration


def _robust_periodic_baseline(values: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Fit only slow circular/elliptical edge variation, rejecting notches."""

    columns = [np.ones_like(angles)]
    for order in range(1, 5):
        columns.extend((np.cos(order * angles), np.sin(order * angles)))
    design = np.column_stack(columns)
    keep = np.ones(len(values), dtype=bool)
    baseline = np.full_like(values, float(np.median(values)), dtype=np.float64)
    for _ in range(6):
        coefficients = np.linalg.lstsq(
            design[keep], values[keep], rcond=None
        )[0]
        baseline = design @ coefficients
        residual = values - baseline
        kept_residual = residual[keep]
        noise = 1.4826 * np.median(
            np.abs(kept_residual - np.median(kept_residual))
        )
        # Positive residuals are outside-black intrusions. Exclude them while
        # retaining broad low-frequency circle/ellipse fitting information.
        keep = residual < max(0.75, 2.5 * float(noise))
        if int(keep.sum()) < design.shape[1] * 3:
            break
    return baseline.astype(np.float32)


def _circular_candidate_groups(active: np.ndarray):
    if not np.any(active):
        return []
    if np.all(active):
        return [np.arange(len(active), dtype=np.int64)]
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    groups = []
    for start_value in starts:
        start = int(start_value)
        values = [start]
        index = (start + 1) % len(active)
        while active[index] and index != start:
            values.append(index)
            index = (index + 1) % len(active)
        groups.append(np.asarray(values, dtype=np.int64))
    return groups


def detect_wafer_notch(
    image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    max_dimension: int = 2048,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.002,
    min_wide_notch_deg: float = 2.0,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> NotchAngleResult:
    """Find a local inward deviation of the otherwise circular wafer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``. Detection measures
    how far the corner-background colour penetrates inside the fitted circle;
    it does not assume a V, U, or semicircle template.

    ``failure_mode="error"`` raises when no notch is reliable.
    ``failure_mode="zero"`` returns ``found=False`` and a zero correction.
    ``require_notch`` remains as a backwards-compatible alias.
    """

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source

    mask, segmentation_threshold = _background_distance_mask(work)
    contour = _select_wafer_contour(mask)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")

    angle_samples = max(720, int(angle_samples))
    angles, penetration = _radial_background_penetration(
        mask,
        (cx, cy),
        float(radius),
        angle_samples=angle_samples,
        radial_inner_ratio=radial_inner_ratio,
    )
    # ``baseline_window_deg`` is retained for API compatibility. A global
    # robust periodic fit replaces the old local closing window, so a long
    # semicircular notch is not absorbed into the baseline.
    _ = baseline_window_deg
    baseline = _robust_periodic_baseline(penetration, angles)
    deficit = penetration - baseline
    median_deficit = float(np.median(deficit))
    radial_noise = float(
        1.4826 * np.median(np.abs(deficit - median_deficit))
    )
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(radius) * float(min_notch_depth_ratio))
    )
    candidate_threshold = max(
        0.60, depth_limit * 0.45, 3.0 * radial_noise
    )
    groups = _circular_candidate_groups(deficit >= candidate_threshold)
    degree_step = 360.0 / angle_samples
    candidates = []
    for indices in groups:
        width_deg = float(len(indices) * degree_step)
        if width_deg > 90.0:
            continue
        values = np.maximum(deficit[indices], 0.0)
        peak = float(values.max())
        area = float(values.sum() * degree_step)
        score = area * math.sqrt(max(peak, 0.0))
        candidates.append((score, peak, area, width_deg, indices))
    if candidates:
        _, peak_depth, candidate_area, notch_width_deg, candidate_indices = max(
            candidates, key=lambda item: item[0]
        )
        peak_index = int(candidate_indices[np.argmax(deficit[candidate_indices])])
    else:
        peak_index = int(np.argmax(deficit))
        peak_depth = float(deficit[peak_index])
        candidate_area = 0.0
        notch_width_deg = degree_step
        candidate_indices = np.asarray((peak_index,), dtype=np.int64)

    strong_notch = peak_depth >= max(
        depth_limit, 0.75, 6.0 * radial_noise
    )
    wide_shallow_notch = bool(
        peak_depth >= max(depth_limit * 0.50, 0.75, 4.0 * radial_noise)
        and notch_width_deg >= float(min_wide_notch_deg)
        and candidate_area >= depth_limit * max(1.5, float(min_wide_notch_deg))
    )
    found = bool(strong_notch or wide_shallow_notch)

    # The requested reference is the angular midpoint of the separated notch
    # region, not the depth-weighted apex. Unwrap group indices so a notch
    # crossing 0/360 degrees is handled correctly.
    unwrapped = np.unwrap(
        candidate_indices.astype(np.float64) * 2.0 * math.pi / angle_samples
    )
    notch_center_index = float(np.mean(unwrapped) * angle_samples / (2.0 * math.pi))
    notch_angle_rad = float(
        (notch_center_index % angle_samples) * 2.0 * math.pi / angle_samples
    )
    notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
    deepest_angle = float(angles[peak_index])
    notch_radius = float(radius - penetration[peak_index])
    notch_deepest_point = (
        float(cx + math.cos(deepest_angle) * notch_radius),
        float(cy + math.sin(deepest_angle) * notch_radius),
    )
    # This is the user-confirmed red point: the notch centre direction at the
    # fitted wafer outer circle, i.e. where the circle would be without a cut.
    notch_point = (
        float(cx + math.cos(notch_angle_rad) * radius),
        float(cy + math.sin(notch_angle_rad) * radius),
    )

    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(notch_width_deg) / 2.0)
    )
    snr = peak_depth / max(0.25, radial_noise)
    depth_score = (peak_depth - depth_limit) / max(depth_limit * 2.0, 1.0)
    area_score = candidate_area / max(depth_limit * 6.0, 1.0)
    confidence = float(np.clip(
        0.50 * min(1.0, snr / 10.0)
        + 0.30 * np.clip(depth_score, 0.0, 1.0)
        + 0.20 * np.clip(area_score, 0.0, 1.0),
        0.0,
        1.0,
    ))
    if not found:
        confidence = 0.0
        if mode == "error":
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                "Use failure_mode='zero' to return angle 0 instead."
            )
        notch_angle_deg = float(reference_angle_deg) % 360.0
        notch_angle_rad = math.radians(notch_angle_deg)
        notch_point = (
            float(cx + math.cos(notch_angle_rad) * radius),
            float(cy + math.sin(notch_angle_rad) * radius),
        )
        notch_deepest_point = notch_point
        candidate_indices = np.asarray((), dtype=np.int64)

    inv_scale = 1.0 / scale
    full_center = (float(cx * inv_scale), float(cy * inv_scale))
    full_radius = float(radius * inv_scale)
    full_notch_point = (
        float(notch_point[0] * inv_scale),
        float(notch_point[1] * inv_scale),
    )
    full_deepest_point = (
        float(notch_deepest_point[0] * inv_scale),
        float(notch_deepest_point[1] * inv_scale),
    )
    arc = tuple(
        (
            float((cx + math.cos(angles[index]) * (radius - penetration[index])) * inv_scale),
            float((cy + math.sin(angles[index]) * (radius - penetration[index])) * inv_scale),
        )
        for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
    )
    contour_full = np.rint(contour.astype(np.float64) * inv_scale).astype(np.int32)
    return NotchAngleResult(
        found=found,
        wafer_center_px=full_center,
        wafer_radius_px=full_radius,
        notch_point_px=full_notch_point,
        notch_deepest_point_px=full_deepest_point,
        notch_angle_deg=notch_angle_deg,
        reference_angle_deg=float(reference_angle_deg),
        correction_angle_deg=_normalise_angle(
            notch_angle_deg - float(reference_angle_deg)
        ),
        notch_depth_px=float(peak_depth * inv_scale),
        notch_width_deg=notch_width_deg,
        notch_width_px=float(notch_width_px * inv_scale),
        confidence=confidence,
        radial_noise_px=float(radial_noise * inv_scale),
        candidate_arc_px=arc,
        wafer_contour_px=contour_full,
        segmentation_threshold=float(segmentation_threshold),
        scale=scale,
        failure_mode=mode,
    )


def align_wafer_by_notch(
    image: ImageInput,
    result: Optional[NotchAngleResult] = None,
    *,
    reference_angle_deg: float = 90.0,
    failure_mode: Literal["error", "zero"] = "error",
    interpolation: int = cv2.INTER_CUBIC,
    border_value: Tuple[int, int, int] = (0, 0, 0),
):
    """Return ``(aligned_image, matrix, inverse_matrix, notch_result)``."""

    source = _load_bgr(image)
    if result is None:
        result = detect_wafer_notch(
            source,
            reference_angle_deg=reference_angle_deg,
            failure_mode=failure_mode,
        )
    height, width = source.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        result.wafer_center_px, result.correction_angle_deg, 1.0
    )
    aligned = cv2.warpAffine(
        source,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    inverse = cv2.invertAffineTransform(matrix)
    return aligned, matrix, inverse, result


def make_notch_overlay(
    image: ImageInput,
    result: NotchAngleResult,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Visualise the user-confirmed outer reference and deepest diagnostic."""

    overlay = _load_bgr(image).copy()
    center = tuple(int(round(v)) for v in result.wafer_center_px)
    notch = tuple(int(round(v)) for v in result.notch_point_px)
    deepest = tuple(int(round(v)) for v in result.notch_deepest_point_px)
    radius = int(round(result.wafer_radius_px))
    cv2.circle(overlay, center, radius, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.drawContours(
        overlay, [result.wafer_contour_px], -1, (120, 120, 120), 1, cv2.LINE_AA
    )
    if result.candidate_arc_px:
        arc = np.rint(np.asarray(result.candidate_arc_px)).astype(np.int32)
        cv2.polylines(overlay, [arc], False, (0, 255, 255), max(2, thickness), cv2.LINE_AA)
    cv2.arrowedLine(
        overlay, center, notch, (0, 220, 0), max(2, thickness), cv2.LINE_AA, tipLength=0.025
    )
    cv2.circle(overlay, center, max(5, thickness * 3), (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, deepest, max(4, thickness * 2), (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, notch, max(6, thickness * 4), (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, notch, max(10, thickness * 6), (255, 255, 255), thickness, cv2.LINE_AA)
    text = (
        f"notch={result.notch_angle_deg:.3f} deg  "
        f"correction={result.correction_angle_deg:+.3f} deg  "
        f"depth={result.notch_depth_px:.1f}px  conf={result.confidence:.2f}"
    )
    cv2.putText(
        overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return overlay


def make_notch_zoom(
    image: ImageInput,
    result: NotchAngleResult,
    *,
    size_px: Optional[int] = None,
    scale: float = 4.0,
) -> np.ndarray:
    """Return an enlarged annotated crop around the selected notch point."""

    overlay = make_notch_overlay(image, result, thickness=2)
    height, width = overlay.shape[:2]
    crop_size = int(size_px or max(80, round(result.wafer_radius_px * 0.13)))
    # Centre the crop between the outer reference and inner apex so both remain
    # visible even when the outer reference is very close to the image border.
    cx = int(round((result.notch_point_px[0] + result.notch_deepest_point_px[0]) / 2.0))
    cy = int(round((result.notch_point_px[1] + result.notch_deepest_point_px[1]) / 2.0))
    x0, x1 = max(0, cx - crop_size), min(width, cx + crop_size)
    y0, y1 = max(0, cy - crop_size), min(height, cy + crop_size)
    crop = overlay[y0:y1, x0:x1]
    return cv2.resize(
        crop, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST
    )
