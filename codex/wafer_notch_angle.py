"""Geometry-first wafer notch detection and angle alignment.

The detector does not classify wafer/background colours. It tracks the
strongest continuous colour edge near the expected outer circumference, fits
the circle while excluding the bottom search sector, and measures a local
inward deviation inside that sector. Candidate depth, angular width, edge
support, and area are scored together.

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
    detection_method: str
    search_center_angle_deg: float
    search_half_width_deg: float
    edge_support: float
    circle_fit_residual_px: float


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


def _lab_edge_strength(image_bgr: np.ndarray):
    """Return colour-transition strength without choosing either side's colour."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5, 5), 0).astype(np.float32)
    squared = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel_index in range(3):
        channel = lab[:, :, channel_index]
        gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        squared += gx * gx + gy * gy
    edge = np.sqrt(squared).astype(np.float32)
    normaliser = float(np.percentile(edge, 99.5))
    if not np.isfinite(normaliser) or normaliser <= 1e-6:
        raise RuntimeError("Wafer outer edge was not found: image has no usable colour edge.")
    edge = np.clip(edge / normaliser, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    return edge.astype(np.float32), normaliser


def _angle_distance_deg(angles_deg: np.ndarray, centre_deg: float) -> np.ndarray:
    return np.abs((angles_deg - float(centre_deg) + 180.0) % 360.0 - 180.0)


def _polar_sample(
    image: np.ndarray,
    center: Point,
    radii: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    map_x = (
        float(center[0]) + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        float(center[1]) + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _initial_outer_radius(
    edge: np.ndarray,
    center: Point,
    angles: np.ndarray,
) -> Tuple[float, float]:
    """Find the outer circular edge supported by many unrelated angles."""

    height, width = edge.shape
    max_radius = min(
        float(center[0]),
        float(center[1]),
        float(width - 1) - float(center[0]),
        float(height - 1) - float(center[1]),
    )
    if max_radius < min(height, width) * 0.25:
        raise RuntimeError("Wafer centre hint leaves too little room for an outer circle.")
    radii = np.linspace(max_radius * 0.55, max_radius * 0.995, max(128, int(max_radius * 0.50)))
    polar = _polar_sample(edge, center, radii, angles)
    # A true circumference is present at the same radius over many angles.
    # The 65th percentile rejects isolated die/street edges.
    radial_score = np.percentile(polar, 65.0, axis=0).astype(np.float32)
    radial_score = cv2.GaussianBlur(radial_score.reshape(1, -1), (11, 1), 0).reshape(-1)
    outer_bias = 0.70 + 0.30 * (radii - radii[0]) / max(1e-6, radii[-1] - radii[0])
    scored = radial_score * outer_bias
    index = int(np.argmax(scored))
    return float(radii[index]), float(radial_score[index])


def _track_outer_edge(
    edge: np.ndarray,
    center: Point,
    radius: float,
    angles: np.ndarray,
    *,
    inward_px: float,
    outward_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track a continuous edge close to the predicted circumference."""

    height, width = edge.shape
    max_corner_radius = math.hypot(width, height)
    radial_count = max(80, int(round(inward_px + outward_px)) * 2)
    radii = np.linspace(
        max(2.0, float(radius) - float(inward_px)),
        min(max_corner_radius, float(radius) + float(outward_px)),
        radial_count,
    )
    polar = _polar_sample(edge, center, radii, angles).astype(np.float32)
    samples_per_degree = len(angles) / 360.0
    angular_kernel = max(3, int(round(0.45 * samples_per_degree)) | 1)
    polar = cv2.GaussianBlur(polar, (3, angular_kernel), 0)

    # Only a weak distance prior is used. At a notch the outer-circle edge is
    # absent, so the actual inner arc must still be allowed to win.
    distance = np.abs(radii - float(radius))
    prior = np.exp(-distance / max(2.0, float(inward_px) * 0.55))
    scored = polar * (0.82 + 0.18 * prior[None, :])
    indices = np.argmax(scored, axis=1)
    boundary = radii[indices].astype(np.float32)
    support = polar[np.arange(len(angles)), indices].astype(np.float32)
    boundary = _circular_median(
        boundary, max(3, int(round(0.25 * samples_per_degree)) | 1)
    )
    return boundary, support


def _fit_circle_from_radial_profile(
    boundary: np.ndarray,
    support: np.ndarray,
    angles: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit radius and centre offset from r(theta)=R+dx*cos+dy*sin."""

    design = np.column_stack((np.ones(len(angles)), np.cos(angles), np.sin(angles)))
    base = valid & np.isfinite(boundary) & np.isfinite(support)
    if int(base.sum()) < 60:
        raise RuntimeError("Wafer circle fit has insufficient supported edge angles.")
    support_floor = float(np.percentile(support[base], 35.0))
    base &= support >= support_floor
    keep = base.copy()
    coefficients = np.asarray((float(np.median(boundary[base])), 0.0, 0.0))
    residual_noise = float("inf")
    for _ in range(7):
        weights = np.clip(support[keep], 0.03, 1.0)
        weighted_design = design[keep] * np.sqrt(weights)[:, None]
        weighted_values = boundary[keep] * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        residual = boundary - design @ coefficients
        centre = float(np.median(residual[keep]))
        residual_noise = float(1.4826 * np.median(np.abs(residual[keep] - centre)))
        new_keep = base & (np.abs(residual - centre) <= max(1.25, 3.0 * residual_noise))
        if int(new_keep.sum()) < 60 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        residual_noise,
    )


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
    max_dimension: int = 3072,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.001,
    min_wide_notch_deg: float = 2.0,
    search_center_angle_deg: float = 90.0,
    search_half_width_deg: float = 45.0,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> NotchAngleResult:
    """Find a local inward deviation of the wafer's geometric outer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.

    No foreground/background colour is assumed. LAB colour-gradient magnitude
    supplies edge evidence only. The circle is fitted outside the configured
    bottom search sector, and the notch is the supported inward edge deviation
    within that sector. Hints are full-resolution image coordinates.

    ``failure_mode="error"`` raises when no notch is reliable.
    ``failure_mode="zero"`` returns ``found=False`` and a zero correction.
    ``require_notch`` remains as a backwards-compatible alias.
    """

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 2.0 <= float(search_half_width_deg) <= 120.0:
        raise ValueError("search_half_width_deg must be between 2 and 120 degrees.")
    if not 0.50 <= float(radial_inner_ratio) < 1.0:
        raise ValueError("radial_inner_ratio must be in [0.50, 1.0).")

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source

    edge, edge_normaliser = _lab_edge_strength(work)
    work_height, work_width = work.shape[:2]

    angle_samples = max(720, int(angle_samples))
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    angles_deg = np.degrees(angles)
    search_distance = _angle_distance_deg(angles_deg, search_center_angle_deg)
    search_mask = search_distance <= float(search_half_width_deg)
    fit_mask = search_distance >= float(search_half_width_deg) + 5.0

    if wafer_center_hint_px is None:
        center = (work_width / 2.0, work_height / 2.0)
    else:
        center = (
            float(wafer_center_hint_px[0]) * scale,
            float(wafer_center_hint_px[1]) * scale,
        )
    if wafer_radius_hint_px is None:
        radius, _ = _initial_outer_radius(edge, center, angles)
    else:
        radius = float(wafer_radius_hint_px) * scale
    if radius <= min(work_height, work_width) * 0.20:
        raise RuntimeError("Estimated wafer radius is implausibly small.")

    circle_fit_noise = float("inf")
    # Re-centre using the first harmonic of the tracked radius. The notch
    # sector is excluded, so a wide or deep notch cannot pull the fitted circle.
    for _ in range(4):
        fit_window = max(12.0, radius * 0.08)
        boundary, support = _track_outer_edge(
            edge,
            center,
            radius,
            angles,
            inward_px=fit_window,
            outward_px=fit_window,
        )
        fitted_radius, offset_x, offset_y, circle_fit_noise = _fit_circle_from_radial_profile(
            boundary, support, angles, fit_mask
        )
        max_step = max(3.0, radius * 0.06)
        offset_length = math.hypot(offset_x, offset_y)
        if offset_length > max_step:
            factor = max_step / offset_length
            offset_x *= factor
            offset_y *= factor
        center = (center[0] + offset_x, center[1] + offset_y)
        radius = float(fitted_radius)
        if not np.isfinite(radius) or radius <= min(work_height, work_width) * 0.20:
            raise RuntimeError("Wafer circle fit produced an invalid radius.")
        if abs(offset_x) + abs(offset_y) < 0.20:
            break
    cx, cy = center
    if not (-0.10 * work_width <= cx <= 1.10 * work_width and -0.10 * work_height <= cy <= 1.10 * work_height):
        raise RuntimeError("Estimated wafer centre is outside the image.")

    inward_range = max(8.0, radius * (1.0 - float(radial_inner_ratio)))
    outward_range = max(5.0, radius * 0.018)
    boundary, support = _track_outer_edge(
        edge,
        center,
        radius,
        angles,
        inward_px=inward_range,
        outward_px=outward_range,
    )

    raw_depth = float(radius) - boundary
    supported_fit = fit_mask & (support >= np.percentile(support[fit_mask], 30.0))
    baseline_shift = float(np.median(raw_depth[supported_fit]))
    deficit = raw_depth - baseline_shift
    samples_per_degree = angle_samples / 360.0
    deficit = _circular_gaussian(
        deficit, max(3, int(round(0.30 * samples_per_degree)) | 1)
    )
    fit_residual = deficit[supported_fit]
    fit_residual_median = float(np.median(fit_residual))
    radial_noise = float(
        1.4826 * np.median(np.abs(fit_residual - fit_residual_median))
    )

    # ``baseline_window_deg`` remains accepted so old copy/paste calls do not
    # break. Geometry fitting outside the search sector replaces that baseline.
    _ = baseline_window_deg
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(radius) * float(min_notch_depth_ratio))
    )
    candidate_threshold = max(
        0.50, depth_limit * 0.40, 2.5 * radial_noise
    )
    # Do not reject a notch because its edge is weak: low contrast is exactly
    # the difficult production case. Edge support contributes to confidence,
    # while geometry (depth/width/area) decides whether the depression exists.
    active = search_mask & (deficit >= candidate_threshold)
    # The notch edge can be interrupted by glare, die streets, or texture.
    # Join only short angular gaps; this preserves one physical depression
    # without requiring every ray to contain a strong gradient.
    bridge_kernel = max(3, int(round(0.80 * samples_per_degree)) | 1)
    half_bridge = bridge_kernel // 2
    extended_active = np.concatenate(
        (active[-half_bridge:], active, active[:half_bridge])
    ).astype(np.uint8).reshape(1, -1)
    extended_active = cv2.morphologyEx(
        extended_active,
        cv2.MORPH_CLOSE,
        np.ones((1, bridge_kernel), np.uint8),
    ).reshape(-1)
    active = extended_active[half_bridge:half_bridge + angle_samples].astype(bool)
    active &= search_mask
    groups = _circular_candidate_groups(active)
    degree_step = 360.0 / angle_samples
    candidates = []
    for indices in groups:
        width_deg = float(len(indices) * degree_step)
        if width_deg > 2.0 * float(search_half_width_deg):
            continue
        values = np.maximum(deficit[indices], 0.0)
        peak = float(values.max())
        area = float(values.sum() * degree_step)
        candidate_support = float(np.mean(support[indices]))
        score = area * math.sqrt(max(peak, 0.0)) * max(candidate_support, 0.02)
        candidates.append((score, peak, area, width_deg, candidate_support, indices))
    if candidates:
        _, peak_depth, candidate_area, notch_width_deg, candidate_support, candidate_indices = max(
            candidates, key=lambda item: item[0]
        )
        peak_index = int(candidate_indices[np.argmax(deficit[candidate_indices])])
    else:
        search_indices = np.flatnonzero(search_mask)
        peak_index = int(search_indices[np.argmax(deficit[search_indices])])
        peak_depth = float(deficit[peak_index])
        candidate_area = 0.0
        notch_width_deg = degree_step
        candidate_support = float(support[peak_index])
        candidate_indices = np.asarray((peak_index,), dtype=np.int64)

    minimum_width = degree_step
    strong_notch = bool(
        peak_depth >= max(depth_limit, 0.70, 4.5 * radial_noise)
        and notch_width_deg >= minimum_width
    )
    wide_shallow_notch = bool(
        peak_depth >= max(depth_limit * 0.45, 0.60, 3.0 * radial_noise)
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
    notch_radius = float(boundary[peak_index])
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
        0.35 * min(1.0, snr / 8.0)
        + 0.25 * np.clip(depth_score, 0.0, 1.0)
        + 0.20 * np.clip(area_score, 0.0, 1.0)
        + 0.20 * np.clip(candidate_support, 0.0, 1.0),
        0.0,
        1.0,
    ))
    if not found:
        confidence = 0.0
        if mode == "error":
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                f"search={float(search_center_angle_deg):.1f}+/-{float(search_half_width_deg):.1f}deg. "
                "Use failure_mode='zero' to return angle 0, or provide wafer centre/radius hints."
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
            float((cx + math.cos(angles[index]) * boundary[index]) * inv_scale),
            float((cy + math.sin(angles[index]) * boundary[index]) * inv_scale),
        )
        for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
    )
    contour_stride = max(1, angle_samples // 1440)
    contour_indices = np.arange(0, angle_samples, contour_stride, dtype=np.int64)
    contour_points = np.column_stack((
        cx + np.cos(angles[contour_indices]) * boundary[contour_indices],
        cy + np.sin(angles[contour_indices]) * boundary[contour_indices],
    ))
    contour_full = np.rint(contour_points * inv_scale).astype(np.int32).reshape(-1, 1, 2)
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
        segmentation_threshold=float(edge_normaliser),
        scale=scale,
        failure_mode=mode,
        detection_method="geometry_edge_bottom_sector",
        search_center_angle_deg=float(search_center_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=float(candidate_support),
        circle_fit_residual_px=float(circle_fit_noise * inv_scale),
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
    for boundary_angle in (
        result.search_center_angle_deg - result.search_half_width_deg,
        result.search_center_angle_deg + result.search_half_width_deg,
    ):
        angle_rad = math.radians(boundary_angle)
        endpoint = (
            int(round(center[0] + math.cos(angle_rad) * radius)),
            int(round(center[1] + math.sin(angle_rad) * radius)),
        )
        cv2.line(overlay, center, endpoint, (255, 0, 255), 1, cv2.LINE_AA)
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
        f"found={result.found}  notch={result.notch_angle_deg:.3f} deg  "
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
    diagnostic_text = (
        f"method={result.detection_method}  edge={result.edge_support:.3f}  "
        f"circle_residual={result.circle_fit_residual_px:.2f}px"
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
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
