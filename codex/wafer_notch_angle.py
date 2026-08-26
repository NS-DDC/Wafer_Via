"""Colour-independent wafer notch detection and angle alignment.

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
from typing import Any, Optional, Tuple, Union

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


def _radial_profile(
    filled_mask: np.ndarray,
    center: Point,
    radius: float,
    *,
    angle_samples: int,
    radial_inner_ratio: float,
):
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    radial_samples = max(160, int(round(radius * 0.35)))
    radii = np.linspace(
        radius * float(radial_inner_ratio), radius + 2.0, radial_samples
    )
    map_x = (
        float(center[0]) + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        float(center[1]) + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    samples = cv2.remap(
        filled_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    indices = np.where(
        samples, np.arange(radial_samples, dtype=np.int32)[None, :], -1
    ).max(axis=1)
    profile = np.where(
        indices >= 0,
        radii[np.maximum(indices, 0)],
        radius * float(radial_inner_ratio),
    ).astype(np.float32)
    return angles, profile


def detect_wafer_notch(
    image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    max_dimension: int = 2048,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.006,
    require_notch: bool = True,
) -> NotchAngleResult:
    """Find a local inward deviation of the otherwise circular wafer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.
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
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)

    angle_samples = max(720, int(angle_samples))
    angles, profile = _radial_profile(
        filled,
        (cx, cy),
        float(radius),
        angle_samples=angle_samples,
        radial_inner_ratio=radial_inner_ratio,
    )
    samples_per_degree = angle_samples / 360.0
    baseline_kernel = max(
        5, int(round(float(baseline_window_deg) * samples_per_degree)) | 1
    )
    extended = np.concatenate(
        (profile[-baseline_kernel:], profile, profile[:baseline_kernel])
    ).reshape(1, -1)
    baseline = cv2.dilate(
        extended, np.ones((1, baseline_kernel), np.uint8)
    ).reshape(-1)[baseline_kernel:baseline_kernel + angle_samples]
    smooth_kernel = max(5, int(round(4.0 * samples_per_degree)) | 1)
    baseline = _circular_gaussian(baseline, smooth_kernel)
    deficit = baseline - profile
    deficit = _circular_gaussian(
        deficit, max(3, int(round(0.8 * samples_per_degree)) | 1)
    )

    peak_index = int(np.argmax(deficit))
    peak_depth = float(deficit[peak_index])
    median_deficit = float(np.median(deficit))
    radial_noise = float(
        1.4826 * np.median(np.abs(deficit - median_deficit))
    )
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(4.0, float(radius) * float(min_notch_depth_ratio))
    )
    found = bool(
        peak_depth >= depth_limit
        and peak_depth >= median_deficit + max(3.0, 5.0 * radial_noise)
    )

    width_threshold = max(depth_limit * 0.75, peak_depth * 0.30)
    active = deficit >= width_threshold
    left = peak_index
    right = peak_index
    while active[(left - 1) % angle_samples] and peak_index - left < angle_samples:
        left -= 1
    while active[(right + 1) % angle_samples] and right - peak_index < angle_samples:
        right += 1
    candidate_indices = np.arange(left, right + 1, dtype=np.int64) % angle_samples
    # The requested reference is the angular midpoint of the separated notch
    # region, not the depth-weighted apex. Keep the unwrapped left/right values
    # so a notch crossing 0/360 degrees is handled correctly.
    notch_center_index = (float(left) + float(right)) / 2.0
    notch_angle_rad = float(
        (notch_center_index % angle_samples) * 2.0 * math.pi / angle_samples
    )
    notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
    notch_index = int(round(notch_angle_deg / 360.0 * angle_samples)) % angle_samples
    notch_radius = float(profile[notch_index])
    notch_deepest_point = (
        float(cx + math.cos(notch_angle_rad) * notch_radius),
        float(cy + math.sin(notch_angle_rad) * notch_radius),
    )
    # This is the user-confirmed red point: the notch centre direction at the
    # fitted wafer outer circle, i.e. where the circle would be without a cut.
    notch_point = (
        float(cx + math.cos(notch_angle_rad) * radius),
        float(cy + math.sin(notch_angle_rad) * radius),
    )

    notch_width_deg = float(len(candidate_indices) / samples_per_degree)
    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(notch_width_deg) / 2.0)
    )
    snr = peak_depth / max(1.0, radial_noise)
    depth_score = (peak_depth - depth_limit) / max(depth_limit * 2.0, 1.0)
    confidence = float(np.clip(0.55 * min(1.0, snr / 10.0) + 0.45 * np.clip(depth_score, 0.0, 1.0), 0.0, 1.0))
    if not found:
        confidence = min(confidence, 0.25)
        if require_notch:
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"required={depth_limit / scale:.2f}px."
            )

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
            float((cx + math.cos(angles[index]) * profile[index]) * inv_scale),
            float((cy + math.sin(angles[index]) * profile[index]) * inv_scale),
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
    )


def align_wafer_by_notch(
    image: ImageInput,
    result: Optional[NotchAngleResult] = None,
    *,
    reference_angle_deg: float = 90.0,
    interpolation: int = cv2.INTER_CUBIC,
    border_value: Tuple[int, int, int] = (0, 0, 0),
):
    """Return ``(aligned_image, matrix, inverse_matrix, notch_result)``."""

    source = _load_bgr(image)
    if result is None:
        result = detect_wafer_notch(
            source, reference_angle_deg=reference_angle_deg
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
