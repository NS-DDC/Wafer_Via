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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]

__all__ = [
    "NotchAngleResult",
    "AlignedNotchGuideResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "make_notch_overlay",
    "make_notch_zoom",
    "draw_aligned_wafer_notch_guide",
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
    candidate_arc_px: Tuple[Point, ...] = field(repr=False)
    wafer_contour_px: np.ndarray
    segmentation_threshold: float
    scale: float
    failure_mode: str
    detection_method: str
    search_center_angle_deg: float
    search_half_width_deg: float
    edge_support: float
    circle_fit_residual_px: float
    roi_center_px: Optional[Point] = None
    roi_bounds_px: Optional[Tuple[float, float, float, float]] = None
    semicircle_center_px: Optional[Point] = None
    semicircle_radius_px: Optional[float] = None
    semicircle_radius_x_px: Optional[float] = None
    semicircle_radius_y_px: Optional[float] = None
    semicircle_shape: str = "none"
    semicircle_score: float = 0.0
    semicircle_fit_residual_px: float = 0.0
    background_segmentation_used: bool = False
    background_palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
    background_distance_threshold_lab: float = 0.0


@dataclass(frozen=True)
class AlignedNotchGuideResult:
    """V5-style geometry and a writable overlay for one aligned wafer image.

    Every point is expressed in the input ``aligned_image`` coordinate system.
    ``overlay_image`` is a full-resolution BGR copy, so callers may draw their
    own ground-truth marks on it with ordinary OpenCV functions.
    """

    overlay_image: np.ndarray = field(repr=False)
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_center_px: Optional[Point]
    notch_point_px: Optional[Point]
    notch_left_px: Optional[Point]
    notch_right_px: Optional[Point]
    notch_angle_deg: Optional[float]
    reference_angle_deg: float
    residual_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    effective_depth_threshold_px: float
    candidate_arc_px: Tuple[Point, ...]
    wafer_contour_px: np.ndarray = field(repr=False)
    search_center_angle_deg: float
    search_half_width_deg: float
    detection_method: str


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


def _lab_edge_strength_from_lab(lab_image: np.ndarray):
    """Return colour-transition strength from a reusable raw LAB image."""

    lab = np.asarray(lab_image)
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_image must have shape (height, width, 3).")
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


def _lab_edge_strength(image_bgr: np.ndarray):
    """Return colour-transition strength without choosing either side's colour."""

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return _lab_edge_strength_from_lab(lab)


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


@dataclass(frozen=True)
class _LocalSemicircleCandidate:
    center: Point
    radius: float
    score: float
    edge_support: float
    arc_coverage: float
    arc_points: Tuple[Point, ...]
    roi_bounds: Tuple[int, int, int, int]
    fit_residual: float = 0.0
    radius_x: Optional[float] = None
    radius_y: Optional[float] = None
    shape: str = "semicircle"


@dataclass(frozen=True)
class _RoiBackgroundGeometry:
    palette_lab: np.ndarray = field(repr=False)
    distance_threshold_lab: float
    sample_mask: np.ndarray = field(repr=False)
    background_like_mask: np.ndarray = field(repr=False)
    exterior_background_mask: np.ndarray = field(repr=False)
    wafer_mask: np.ndarray = field(repr=False)
    wafer_contour: np.ndarray = field(repr=False)
    wafer_center: Point
    wafer_radius: float
    wafer_circle_residual: float
    roi_bounds: Tuple[int, int, int, int]
    outward_unit: Point


def _robust_circle_from_points(
    points: np.ndarray,
    *,
    minimum_points: int = 30,
) -> Tuple[Point, float, float, np.ndarray]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < int(minimum_points):
        raise RuntimeError("Circle fit has insufficient boundary points.")
    center = np.median(values, axis=0)
    radius = float(np.median(np.linalg.norm(values - center, axis=1)))
    keep = np.ones(len(values), dtype=bool)
    residual = np.zeros(len(values), dtype=np.float64)
    noise = float("inf")
    for _ in range(10):
        selected = values[keep]
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        targets = np.sum(selected * selected, axis=1)
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        center = coefficients[:2]
        radius_squared = float(coefficients[2] + center @ center)
        if radius_squared <= 0.0:
            raise RuntimeError("Circle fit produced a non-positive radius.")
        radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(values - center, axis=1) - radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        new_keep = np.abs(residual - median) <= max(1.25, 3.0 * noise)
        if int(new_keep.sum()) < int(minimum_points) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    fit_residual = float(np.median(np.abs(residual[keep])))
    return (float(center[0]), float(center[1])), float(radius), fit_residual, keep


def _learn_background_from_notch_roi(
    image_bgr: np.ndarray,
    roi_center: Point,
    roi_half_size: Point,
    center_hint: Point,
    *,
    palette_size: int = 3,
    outer_band_fraction: float = 0.28,
    distance_threshold_lab: Optional[float] = None,
    noise_margin_lab: float = 4.0,
    morph_size_px: float = 24.0,
    lab_image: Optional[np.ndarray] = None,
) -> _RoiBackgroundGeometry:
    """Learn exterior colour in the outward ROI band and segment the wafer."""

    height, width = image_bgr.shape[:2]
    x0 = max(0, int(math.floor(roi_center[0] - roi_half_size[0])))
    y0 = max(0, int(math.floor(roi_center[1] - roi_half_size[1])))
    x1 = min(width, int(math.ceil(roi_center[0] + roi_half_size[0])) + 1)
    y1 = min(height, int(math.ceil(roi_center[1] + roi_half_size[1])) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")
    if not 1 <= int(palette_size) <= 8:
        raise ValueError("notch_background_palette_size must be between 1 and 8.")
    if not 0.10 <= float(outer_band_fraction) <= 0.60:
        raise ValueError("notch_background_outer_band_fraction must be in [0.10, 0.60].")

    outward = np.asarray(roi_center, dtype=np.float64) - np.asarray(
        center_hint, dtype=np.float64
    )
    outward_length = float(np.linalg.norm(outward))
    if outward_length <= 1e-6:
        outward = np.asarray((0.0, 1.0), dtype=np.float64)
    else:
        outward /= outward_length

    roi_height, roi_width = y1 - y0, x1 - x0
    local_y, local_x = np.indices((roi_height, roi_width), dtype=np.float32)
    global_x = local_x + float(x0)
    global_y = local_y + float(y0)
    projection = (
        (global_x - float(roi_center[0])) * float(outward[0])
        + (global_y - float(roi_center[1])) * float(outward[1])
    )
    quantile = 1.0 - float(outer_band_fraction)
    projection_threshold = float(np.quantile(projection, quantile))
    sample_local = projection >= projection_threshold
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y0:y1, x0:x1] = sample_local.astype(np.uint8) * 255

    if lab_image is None:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    else:
        lab = np.asarray(lab_image)
        if lab.shape != image_bgr.shape:
            raise ValueError("lab_image shape must match image_bgr shape.")
        if lab.ndim != 3 or lab.shape[2] != 3:
            raise ValueError("lab_image must have shape (height, width, 3).")
    lab = lab.astype(np.float32, copy=False)
    samples = lab[y0:y1, x0:x1][sample_local].reshape(-1, 3)
    if len(samples) < 64:
        raise RuntimeError("The outward notch ROI band has too few background pixels.")
    stride = max(1, len(samples) // 30000)
    samples_for_fit = samples[::stride].astype(np.float32)
    distinct = np.unique(samples_for_fit.astype(np.uint8), axis=0)
    cluster_count = min(int(palette_size), len(distinct), len(samples_for_fit))
    if cluster_count <= 1:
        palette = np.median(samples_for_fit, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros((len(samples_for_fit), 1), dtype=np.int32)
    else:
        cv2.setRNGSeed(1907)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.20,
        )
        _, labels, palette = cv2.kmeans(
            samples_for_fit,
            cluster_count,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
    assigned = palette[labels.reshape(-1)]
    sample_residual = np.linalg.norm(samples_for_fit - assigned, axis=1)
    automatic_threshold = max(
        8.0,
        float(np.percentile(sample_residual, 98.0)) + float(noise_margin_lab),
    )
    threshold = (
        automatic_threshold
        if distance_threshold_lab is None
        else float(distance_threshold_lab)
    )
    if threshold <= 0.0:
        raise ValueError("notch_background_distance_threshold_lab must be positive.")

    nearest_distance = np.full((height, width), np.inf, dtype=np.float32)
    for colour in palette:
        delta = lab - colour.reshape(1, 1, 3)
        distance = np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)
        np.minimum(nearest_distance, distance, out=nearest_distance)
    background_like = (nearest_distance <= threshold).astype(np.uint8) * 255
    morph_size = max(3, int(round(float(morph_size_px))) | 1)
    morph_size = min(morph_size, max(3, (min(height, width) // 12) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    background_like = cv2.morphologyEx(
        background_like, cv2.MORPH_CLOSE, kernel
    )

    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        (background_like > 0).astype(np.uint8), 8
    )
    border_labels = np.unique(np.concatenate((
        components[0, :],
        components[-1, :],
        components[:, 0],
        components[:, -1],
    )))
    border_labels = border_labels[border_labels > 0]
    if not len(border_labels):
        raise RuntimeError("ROI background colour did not connect to the image border.")
    exterior_label = int(max(
        border_labels, key=lambda label: int(stats[int(label), cv2.CC_STAT_AREA])
    ))
    exterior_background = (components == exterior_label).astype(np.uint8) * 255

    foreground = (exterior_background == 0).astype(np.uint8)
    open_size = max(3, int(round(float(morph_size_px) * 0.45)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    fg_count, fg_components, fg_stats, fg_centroids = cv2.connectedComponentsWithStats(
        foreground, 8
    )
    center_x = int(np.clip(round(center_hint[0]), 0, width - 1))
    center_y = int(np.clip(round(center_hint[1]), 0, height - 1))
    wafer_label = int(fg_components[center_y, center_x])
    if wafer_label <= 0:
        candidates = []
        for label in range(1, fg_count):
            area = int(fg_stats[label, cv2.CC_STAT_AREA])
            centroid = fg_centroids[label]
            distance_to_hint = float(np.linalg.norm(centroid - np.asarray(center_hint)))
            candidates.append((area / max(1.0, 1.0 + distance_to_hint), label))
        if not candidates:
            raise RuntimeError("Wafer component was not found after ROI background segmentation.")
        wafer_label = int(max(candidates)[1])
    wafer_mask = (fg_components == wafer_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Wafer contour was not found after ROI background segmentation.")
    wafer_contour = max(contours, key=cv2.contourArea)
    contour_values = wafer_contour.reshape(-1, 2)
    border_clear = (
        (contour_values[:, 0] > 1)
        & (contour_values[:, 0] < width - 2)
        & (contour_values[:, 1] > 1)
        & (contour_values[:, 1] < height - 2)
    )
    fit_points = contour_values[border_clear]
    if len(fit_points) > 6000:
        fit_points = fit_points[::max(1, len(fit_points) // 6000)]
    wafer_center, wafer_radius, circle_residual, _ = _robust_circle_from_points(
        fit_points, minimum_points=100
    )
    if wafer_radius <= min(height, width) * 0.20:
        raise RuntimeError("Background-segmented wafer radius is implausibly small.")

    return _RoiBackgroundGeometry(
        palette_lab=palette.astype(np.float32),
        distance_threshold_lab=float(threshold),
        sample_mask=sample_mask,
        background_like_mask=background_like,
        exterior_background_mask=exterior_background,
        wafer_mask=wafer_mask,
        wafer_contour=wafer_contour,
        wafer_center=wafer_center,
        wafer_radius=float(wafer_radius),
        wafer_circle_residual=float(circle_residual),
        roi_bounds=(x0, y0, x1, y1),
        outward_unit=(float(outward[0]), float(outward[1])),
    )


def _sample_fitted_arc_support(
    edge: np.ndarray,
    arc_points: np.ndarray,
) -> Tuple[float, float, float]:
    """Measure local edge support and left/right balance along a fitted arc."""

    height, width = edge.shape
    values = []
    for point in np.asarray(arc_points, dtype=np.float64).reshape(-1, 2):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0, x1 = max(0, x - 2), min(width, x + 3)
        y0, y1 = max(0, y - 2), min(height, y + 3)
        values.append(
            0.0 if x0 >= x1 or y0 >= y1 else float(np.max(edge[y0:y1, x0:x1]))
        )
    support = np.asarray(values, dtype=np.float64)
    if not len(support):
        return 0.0, 0.0, 0.0
    edge_support = float(np.mean(support))
    support_floor = max(0.08, float(np.percentile(support, 35.0)) * 0.70)
    coverage = float(np.mean(support >= support_floor))
    midpoint = len(support) // 2
    left = float(np.mean(support[:midpoint])) if midpoint else edge_support
    right = (
        float(np.mean(support[midpoint + 1 :]))
        if midpoint + 1 < len(support) else edge_support
    )
    symmetry = min(left, right) / max(1e-6, max(left, right))
    return edge_support, coverage, float(symmetry)


def _fit_semiellipse_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a shallow/wide semi-ellipse to the exterior intrusion boundary."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi].astype(np.float64)
    if len(points) < 24:
        return None

    wafer_center = np.asarray(geometry.wafer_center, dtype=np.float64)
    outward = np.asarray(geometry.outward_unit, dtype=np.float64)
    outward /= max(1e-9, float(np.linalg.norm(outward)))
    tangent = np.asarray((-outward[1], outward[0]), dtype=np.float64)
    vectors = points - wafer_center
    radial_distance = np.linalg.norm(vectors, axis=1)
    depth = float(geometry.wafer_radius) - radial_distance
    tangential = vectors @ tangent
    outward_projection = vectors @ outward
    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_half_width = max(3.0, minimum_half_size * 0.035)
        max_half_width = max(min_half_width + 2.0, minimum_half_size * 0.55)
    else:
        min_half_width, max_half_width = (
            float(radius_range[0]), float(radius_range[1])
        )
        if min_half_width <= 0.0 or max_half_width <= min_half_width:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    noise_floor = max(0.75, float(geometry.wafer_circle_residual) * 1.35)
    usable = (
        (depth >= noise_floor)
        & (depth <= max(4.0, minimum_half_size * 0.75))
        & (outward_projection >= float(geometry.wafer_radius) - 2.2 * minimum_half_size)
        & (np.abs(tangential - float(np.dot(np.asarray(roi_center) - wafer_center, tangent)))
           <= float(roi_half_size[0]) * 1.10)
    )
    tangential = tangential[usable]
    depth = depth[usable]
    if len(depth) < 24:
        return None

    # Keep the deepest exterior-boundary sample in each tangential pixel bin.
    bins = np.rint(tangential).astype(np.int32)
    unique_bins = np.unique(bins)
    fitted_t = []
    fitted_d = []
    for value in unique_bins:
        selected_depth = depth[bins == value]
        fitted_t.append(float(value))
        fitted_d.append(float(np.max(selected_depth)))
    fitted_t = np.asarray(fitted_t, dtype=np.float64)
    fitted_d = np.asarray(fitted_d, dtype=np.float64)
    if len(fitted_t) < 18:
        return None

    order = np.argsort(fitted_t)
    fitted_t, fitted_d = fitted_t[order], fitted_d[order]
    expected_t = float(np.dot(np.asarray(roi_center) - wafer_center, tangent))
    central = np.abs(fitted_t - expected_t) <= float(roi_half_size[0]) * 0.75
    if not np.any(central):
        return None
    local_peak = float(np.max(fitted_d[central]))
    strong_threshold = max(noise_floor * 1.55, local_peak * 0.12)
    strong_indices = np.flatnonzero(central & (fitted_d >= strong_threshold))
    if len(strong_indices) < 8:
        return None
    split_at = np.flatnonzero(np.diff(fitted_t[strong_indices]) > 3.5) + 1
    groups = np.split(strong_indices, split_at)
    groups = [group for group in groups if len(group) >= 8]
    if not groups:
        return None
    group = max(
        groups,
        key=lambda values: float(np.sum(fitted_d[values]))
        * math.exp(
            -0.5
            * (
                (float(np.mean(fitted_t[values])) - expected_t)
                / max(3.0, float(roi_half_size[0]) * 0.30)
            )
            ** 2
        ),
    )
    lower = float(fitted_t[group[0]] - 4.0)
    upper = float(fitted_t[group[-1]] + 4.0)
    selected_group = (fitted_t >= lower) & (fitted_t <= upper)
    fitted_t, fitted_d = fitted_t[selected_group], fitted_d[selected_group]
    if len(fitted_t) < 18:
        return None

    # Semi-ellipse linearisation: d^2 = A*t^2 + B*t + C.  Robust iterations
    # reject texture/noise points while retaining the broad, shallow arc.
    origin_t = float(np.median(fitted_t))
    x = fitted_t - origin_t
    keep = np.ones(len(x), dtype=bool)
    half_width = depth_axis = center_offset = fit_residual = 0.0
    for _ in range(10):
        if int(keep.sum()) < 18:
            return None
        design = np.column_stack((x[keep] * x[keep], x[keep], np.ones(int(keep.sum()))))
        coefficients = np.linalg.lstsq(
            design, fitted_d[keep] * fitted_d[keep], rcond=None
        )[0]
        quadratic, linear, constant = (float(value) for value in coefficients)
        if quadratic >= -1e-8:
            return None
        center_local = -linear / (2.0 * quadratic)
        depth_squared = constant - quadratic * center_local * center_local
        if depth_squared <= 0.0:
            return None
        depth_axis = math.sqrt(depth_squared)
        half_width_squared = -depth_squared / quadratic
        if half_width_squared <= 0.0:
            return None
        half_width = math.sqrt(half_width_squared)
        center_offset = origin_t + center_local
        normalized = (fitted_t - center_offset) / max(half_width, 1e-6)
        predicted = depth_axis * np.sqrt(
            np.maximum(0.0, 1.0 - normalized * normalized)
        )
        residual = fitted_d - predicted
        valid_span = np.abs(normalized) <= 1.08
        current = keep & valid_span
        if int(current.sum()) < 18:
            return None
        median = float(np.median(residual[current]))
        noise = float(
            1.4826 * np.median(np.abs(residual[current] - median))
        )
        new_keep = valid_span & (
            np.abs(residual - median) <= max(1.15, 2.8 * noise)
        )
        if int(new_keep.sum()) < 18:
            return None
        fit_residual = float(np.median(np.abs(residual[new_keep])))
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if not (min_half_width * 0.70 <= half_width <= max_half_width * 1.25):
        return None
    aspect = depth_axis / max(half_width, 1e-6)
    if not 0.12 <= aspect <= 1.60:
        return None

    direction_vector = outward * float(geometry.wafer_radius) + tangent * center_offset
    direction_length = float(np.linalg.norm(direction_vector))
    if direction_length <= 1e-6:
        return None
    direction = direction_vector / direction_length
    arc_tangent = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    inward = -direction
    baseline_center = wafer_center + direction * float(geometry.wafer_radius)
    unit = np.linspace(-1.0, 1.0, 97, dtype=np.float64)
    arc_values = (
        baseline_center[None, :]
        + arc_tangent[None, :] * (unit * half_width)[:, None]
        + inward[None, :] * (
            depth_axis * np.sqrt(np.maximum(0.0, 1.0 - unit * unit))
        )[:, None]
    )
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    edge_support, arc_coverage, symmetry = _sample_fitted_arc_support(
        boundary_edge, arc_values
    )
    apex = baseline_center + inward * depth_axis
    center_distance = float(np.linalg.norm(apex - np.asarray(roi_center)))
    center_prior = math.exp(
        -0.5 * (center_distance / max(4.0, minimum_half_size * 0.28)) ** 2
    )
    fit_quality = math.exp(
        -fit_residual / max(1.0, depth_axis * 0.10)
    )
    score = float(np.clip(
        0.26 * edge_support
        + 0.18 * arc_coverage
        + 0.10 * symmetry
        + 0.18 * center_prior
        + 0.28 * fit_quality,
        0.0,
        1.0,
    ))
    stride = max(1, len(arc_values) // 48)
    return _LocalSemicircleCandidate(
        center=(float(baseline_center[0]), float(baseline_center[1])),
        radius=float(half_width),
        score=score,
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=tuple(
            (float(point[0]), float(point[1])) for point in arc_values[::stride]
        ),
        roi_bounds=geometry.roi_bounds,
        fit_residual=float(fit_residual),
        radius_x=float(half_width),
        radius_y=float(depth_axis),
        shape="semiellipse",
    )


def _fit_circle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit the exterior-background intrusion contour around the expected notch."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi]
    if len(points) < 20:
        return None

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
    relative = points.astype(np.float64) - np.asarray(roi_center, dtype=np.float64)
    distances = np.linalg.norm(relative, axis=1)
    inward = -np.asarray(geometry.outward_unit, dtype=np.float64)
    inward_projection = relative @ inward
    usable = (
        (distances >= min_radius)
        & (distances <= max_radius)
        & (inward_projection >= -max(2.0, min_radius * 0.25))
    )
    points = points[usable]
    distances = distances[usable]
    if len(points) < 20:
        return None

    bin_width = max(1.0, minimum_half_size / 240.0)
    bins = np.arange(min_radius, max_radius + 2.0 * bin_width, bin_width)
    histogram, edges = np.histogram(distances, bins=bins)
    if not np.any(histogram):
        return None
    smooth_histogram = cv2.GaussianBlur(
        histogram.astype(np.float32).reshape(1, -1), (5, 1), 0
    ).reshape(-1)
    peak_indices = np.argsort(smooth_histogram)[::-1][:12]
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    best: Optional[_LocalSemicircleCandidate] = None
    for peak_index in peak_indices:
        peak_radius = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
        band = max(2.0, peak_radius * 0.08)
        selected = points[np.abs(distances - peak_radius) <= band]
        if len(selected) < 20:
            continue
        try:
            center, radius, fit_residual, _ = _robust_circle_from_points(
                selected, minimum_points=18
            )
        except RuntimeError:
            continue
        center_distance = float(np.linalg.norm(
            np.asarray(center) - np.asarray(roi_center)
        ))
        if center_distance > minimum_half_size * 0.45:
            continue
        if not min_radius * 0.70 <= radius <= max_radius * 1.20:
            continue
        inward_angle = math.atan2(
            geometry.wafer_center[1] - center[1],
            geometry.wafer_center[0] - center[0],
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            boundary_edge, center, radius, inward_angle
        )
        center_prior = math.exp(
            -0.5 * (center_distance / max(4.0, minimum_half_size * 0.20)) ** 2
        )
        fit_quality = math.exp(
            -fit_residual / max(1.25, radius * 0.055)
        )
        score = float(np.clip(
            0.24 * edge_support
            + 0.16 * arc_coverage
            + 0.10 * symmetry
            + 0.25 * center_prior
            + 0.25 * fit_quality,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=center,
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=geometry.roi_bounds,
            fit_residual=fit_residual,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _fit_semicircle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a semi-ellipse first, then retain the historical circle fallback."""

    candidate = _fit_semiellipse_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )
    if candidate is not None:
        return candidate
    return _fit_circle_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )


def _normalise_roi_half_size(
    value: Union[float, Tuple[float, float]],
    *,
    scale: float,
) -> Point:
    if isinstance(value, (int, float, np.integer, np.floating)):
        half_width = half_height = float(value)
    else:
        if len(value) != 2:
            raise ValueError("notch_roi_half_size_px must be a number or (half_width, half_height).")
        half_width, half_height = float(value[0]), float(value[1])
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("notch_roi_half_size_px values must be positive.")
    return half_width * float(scale), half_height * float(scale)


def _sample_semicircle_support(
    edge: np.ndarray,
    center: Point,
    radius: float,
    inward_angle_rad: float,
) -> Tuple[float, float, float, Tuple[Point, ...]]:
    """Measure the inward-facing half of a local U-shaped notch circle."""

    arc_angles = np.linspace(
        float(inward_angle_rad) - math.radians(100.0),
        float(inward_angle_rad) + math.radians(100.0),
        161,
        dtype=np.float64,
    )
    band = max(1.0, min(5.0, float(radius) * 0.10))
    radial_offsets = np.linspace(-band, band, 7, dtype=np.float64)
    radii = np.maximum(1.0, float(radius) + radial_offsets)
    sampled = _polar_sample(edge, center, radii, arc_angles)
    supported = sampled.max(axis=1)
    edge_support = float(np.mean(supported))
    support_floor = max(0.08, float(np.percentile(supported, 35.0)) * 0.70)
    arc_coverage = float(np.mean(supported >= support_floor))
    midpoint = len(supported) // 2
    left_support = float(np.mean(supported[:midpoint]))
    right_support = float(np.mean(supported[midpoint + 1:]))
    symmetry = min(left_support, right_support) / max(
        1e-6, max(left_support, right_support)
    )
    stride = max(1, len(arc_angles) // 48)
    arc_points = tuple(
        (
            float(center[0] + math.cos(float(angle)) * float(radius)),
            float(center[1] + math.sin(float(angle)) * float(radius)),
        )
        for angle in arc_angles[::stride]
    )
    return edge_support, arc_coverage, float(symmetry), arc_points


def _refine_semicircle_candidate(
    edge: np.ndarray,
    wafer_center: Point,
    candidate: _LocalSemicircleCandidate,
) -> _LocalSemicircleCandidate:
    """Robustly re-fit the actual inward arc after coarse Hough detection."""

    initial_center = np.asarray(candidate.center, dtype=np.float64)
    initial_radius = float(candidate.radius)
    inward_angle = math.atan2(
        float(wafer_center[1]) - float(initial_center[1]),
        float(wafer_center[0]) - float(initial_center[0]),
    )
    arc_angles = np.linspace(
        inward_angle - math.radians(100.0),
        inward_angle + math.radians(100.0),
        241,
        dtype=np.float64,
    )
    radii = np.linspace(
        max(2.0, initial_radius * 0.65), initial_radius * 1.35, 101
    )
    polar = _polar_sample(edge, candidate.center, radii, arc_angles)
    indices = np.argmax(polar, axis=1)
    support = polar[np.arange(len(arc_angles)), indices]
    selected_radii = radii[indices]
    support_floor = max(0.08, float(np.percentile(support, 35.0)))
    keep = (
        (support >= support_floor)
        & (np.abs(selected_radii - initial_radius) <= initial_radius * 0.30)
    )
    points = np.column_stack((
        initial_center[0] + np.cos(arc_angles) * selected_radii,
        initial_center[1] + np.sin(arc_angles) * selected_radii,
    ))
    if int(keep.sum()) < 30:
        return candidate

    fitted_center = initial_center.copy()
    fitted_radius = initial_radius
    residual = np.zeros(len(points), dtype=np.float64)
    for _ in range(8):
        selected = points[keep]
        weights = np.clip(support[keep], 0.03, 1.0)
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        values = np.sum(selected * selected, axis=1)
        root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None], values * root_weights, rcond=None
        )[0]
        fitted_center = coefficients[:2]
        radius_squared = float(coefficients[2] + fitted_center @ fitted_center)
        if radius_squared <= 0.0:
            return candidate
        fitted_radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(points - fitted_center, axis=1) - fitted_radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        refined_keep = keep & (np.abs(residual - median) <= max(1.25, 2.5 * noise))
        if int(refined_keep.sum()) < 30 or np.array_equal(refined_keep, keep):
            break
        keep = refined_keep

    center_shift = float(np.linalg.norm(fitted_center - initial_center))
    fit_residual = float(np.median(np.abs(residual[keep])))
    reliable = bool(
        center_shift <= initial_radius * 0.45
        and initial_radius * 0.55 <= fitted_radius <= initial_radius * 1.55
        and fit_residual <= max(1.5, fitted_radius * 0.06)
    )
    if not reliable:
        return replace(candidate, fit_residual=fit_residual)

    refined_inward_angle = math.atan2(
        float(wafer_center[1]) - float(fitted_center[1]),
        float(wafer_center[0]) - float(fitted_center[0]),
    )
    edge_support, arc_coverage, _, arc_points = _sample_semicircle_support(
        edge,
        (float(fitted_center[0]), float(fitted_center[1])),
        fitted_radius,
        refined_inward_angle,
    )
    return replace(
        candidate,
        center=(float(fitted_center[0]), float(fitted_center[1])),
        radius=float(fitted_radius),
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=arc_points,
        fit_residual=fit_residual,
    )


def _detect_semicircle_in_roi(
    edge: np.ndarray,
    wafer_center: Point,
    wafer_radius: float,
    roi_center: Point,
    roi_half_size: Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Find a small inward-facing semicircle only inside a user ROI."""

    height, width = edge.shape
    x0 = max(0, int(math.floor(float(roi_center[0]) - float(roi_half_size[0]))))
    y0 = max(0, int(math.floor(float(roi_center[1]) - float(roi_half_size[1]))))
    x1 = min(width, int(math.ceil(float(roi_center[0]) + float(roi_half_size[0]))) + 1)
    y1 = min(height, int(math.ceil(float(roi_center[1]) + float(roi_half_size[1]))) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
        if min_radius <= 0.0 or max_radius <= min_radius:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    crop = np.clip(edge[y0:y1, x0:x1] * 255.0, 0.0, 255.0).astype(np.uint8)
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(4.0, min_radius * 0.70),
        param1=40.0,
        param2=max(6.0, min(crop.shape) * 0.018),
        minRadius=max(2, int(math.floor(min_radius))),
        maxRadius=max(3, int(math.ceil(max_radius))),
    )
    if circles is None:
        return None

    wafer_center_array = np.asarray(wafer_center, dtype=np.float64)
    roi_center_array = np.asarray(roi_center, dtype=np.float64)
    # The caller supplies this coordinate precisely because the hardware keeps
    # the notch in a stable area. Give proximity real weight after the arc has
    # passed the geometric filters; otherwise a stronger decorative/internal
    # circle elsewhere in a large ROI can still win.
    roi_scale = max(4.0, 0.20 * minimum_half_size)
    candidate_pool: List[_LocalSemicircleCandidate] = []
    for local_x, local_y, candidate_radius in circles[0]:
        center = np.asarray(
            (float(local_x) + float(x0), float(local_y) + float(y0)),
            dtype=np.float64,
        )
        radius = float(candidate_radius)
        center_radius = float(np.linalg.norm(center - wafer_center_array))
        ring_error = abs(center_radius - float(wafer_radius))
        if ring_error > max(radius * 2.0, minimum_half_size * 0.20):
            continue

        inward_angle = math.atan2(
            float(wafer_center[1]) - float(center[1]),
            float(wafer_center[0]) - float(center[0]),
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            edge, (float(center[0]), float(center[1])), radius, inward_angle
        )
        hint_distance = float(np.linalg.norm(center - roi_center_array))
        center_prior = math.exp(-0.5 * (hint_distance / roi_scale) ** 2)
        ring_prior = math.exp(
            -0.5 * (ring_error / max(2.0, radius * 0.90)) ** 2
        )
        score = float(np.clip(
            0.32 * edge_support
            + 0.16 * arc_coverage
            + 0.12 * symmetry
            + 0.32 * center_prior
            + 0.08 * ring_prior,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=(float(center[0]), float(center[1])),
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=(x0, y0, x1, y1),
        )
        candidate_pool.append(candidate)
    if not candidate_pool:
        return None

    # Refine several strong coarse candidates. Decorative circles can have a
    # slightly higher Hough score, while the true notch wins decisively after
    # its inward arc is fitted with a low radial residual.
    best: Optional[_LocalSemicircleCandidate] = None
    for coarse in sorted(
        candidate_pool, key=lambda item: item.score, reverse=True
    )[:20]:
        refined = _refine_semicircle_candidate(edge, wafer_center, coarse)
        fit_quality = (
            math.exp(
                -float(refined.fit_residual)
                / max(1.5, float(refined.radius) * 0.06)
            )
            if refined.fit_residual > 0.0
            else 0.0
        )
        refined = replace(
            refined,
            score=float(np.clip(0.80 * coarse.score + 0.20 * fit_quality, 0.0, 1.0)),
        )
        if best is None or refined.score > best.score:
            best = refined
    return best


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
    notch_roi_center_px: Optional[Point] = None,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    notch_semicircle_min_score: float = 0.55,
    notch_use_roi_background: bool = True,
    notch_background_palette_size: int = 3,
    notch_background_outer_band_fraction: float = 0.28,
    notch_background_distance_threshold_lab: Optional[float] = None,
    notch_background_noise_margin_lab: float = 4.0,
    notch_background_morph_px: float = 24.0,
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> NotchAngleResult:
    """Find a local inward deviation of the wafer's geometric outer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.

    Without an ROI, LAB colour-gradient magnitude supplies edge evidence and
    no foreground/background colour is assumed. With
    ``notch_roi_center_px=(x, y)`` the default ROI mode learns the wafer-exterior
    background palette from the outward part of that crop. Only the
    border-connected background is retained; its boundary supplies both a
    noise-resistant wafer silhouette and the local inward semicircle or
    wide/shallow semi-ellipse.
    ``notch_roi_half_size_px`` is a scalar or ``(half_width, half_height)`` in
    full-resolution pixels. ``notch_semicircle_radius_range_px`` and
    ``notch_background_morph_px`` are also full-resolution pixel values.

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
    if not 0.0 <= float(notch_semicircle_min_score) <= 1.0:
        raise ValueError("notch_semicircle_min_score must be between 0 and 1.")
    if float(notch_background_noise_margin_lab) < 0.0:
        raise ValueError("notch_background_noise_margin_lab must be non-negative.")
    if float(notch_background_morph_px) <= 0.0:
        raise ValueError("notch_background_morph_px must be positive.")

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source

    # Convert once. Edge extraction uses a blurred copy while ROI background
    # learning uses the unblurred LAB values for colour-distance segmentation.
    work_lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    edge, edge_normaliser = _lab_edge_strength_from_lab(work_lab)
    work_height, work_width = work.shape[:2]

    angle_samples = max(720, int(angle_samples))
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    angles_deg = np.degrees(angles)
    if wafer_center_hint_px is None:
        center = (work_width / 2.0, work_height / 2.0)
    else:
        center = (
            float(wafer_center_hint_px[0]) * scale,
            float(wafer_center_hint_px[1]) * scale,
        )
    roi_center: Optional[Point] = None
    roi_half_size: Optional[Point] = None
    semicircle_radius_range: Optional[Tuple[float, float]] = None
    background_geometry: Optional[_RoiBackgroundGeometry] = None
    effective_search_center_angle_deg = float(search_center_angle_deg) % 360.0
    if notch_roi_center_px is not None:
        roi_center = (
            float(notch_roi_center_px[0]) * scale,
            float(notch_roi_center_px[1]) * scale,
        )
        if not (
            -0.05 * work_width <= roi_center[0] <= 1.05 * work_width
            and -0.05 * work_height <= roi_center[1] <= 1.05 * work_height
        ):
            raise ValueError("notch_roi_center_px lies outside the input image.")
        roi_half_size = _normalise_roi_half_size(
            notch_roi_half_size_px, scale=scale
        )
        effective_search_center_angle_deg = float(
            math.degrees(
                math.atan2(roi_center[1] - center[1], roi_center[0] - center[0])
            )
            % 360.0
        )
        if notch_semicircle_radius_range_px is not None:
            semicircle_radius_range = (
                float(notch_semicircle_radius_range_px[0]) * scale,
                float(notch_semicircle_radius_range_px[1]) * scale,
            )
        if bool(notch_use_roi_background):
            background_geometry = _learn_background_from_notch_roi(
                work,
                roi_center,
                roi_half_size,
                center,
                palette_size=notch_background_palette_size,
                outer_band_fraction=notch_background_outer_band_fraction,
                distance_threshold_lab=notch_background_distance_threshold_lab,
                noise_margin_lab=notch_background_noise_margin_lab,
                morph_size_px=float(notch_background_morph_px) * scale,
                lab_image=work_lab,
            )
            if wafer_center_hint_px is None:
                center = background_geometry.wafer_center
    del work_lab
    search_distance = _angle_distance_deg(
        angles_deg, effective_search_center_angle_deg
    )
    search_mask = search_distance <= float(search_half_width_deg)
    fit_mask = search_distance >= float(search_half_width_deg) + 5.0

    if wafer_radius_hint_px is None:
        if background_geometry is not None:
            radius = float(background_geometry.wafer_radius)
        else:
            radius, _ = _initial_outer_radius(edge, center, angles)
    else:
        radius = float(wafer_radius_hint_px) * scale
    if radius <= min(work_height, work_width) * 0.20:
        raise RuntimeError("Estimated wafer radius is implausibly small.")

    circle_fit_noise = (
        float(background_geometry.wafer_circle_residual)
        if background_geometry is not None else float("inf")
    )
    # Re-centre using the first harmonic of the tracked radius. The notch
    # sector is excluded, so a wide or deep notch cannot pull the fitted circle.
    for _ in range(0 if background_geometry is not None else 4):
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

    if roi_center is not None:
        effective_search_center_angle_deg = float(
            math.degrees(math.atan2(roi_center[1] - cy, roi_center[0] - cx))
            % 360.0
        )
        search_distance = _angle_distance_deg(
            angles_deg, effective_search_center_angle_deg
        )
        search_mask = search_distance <= float(search_half_width_deg)

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
    detection_method = "geometry_edge_bottom_sector"
    local_arc: Optional[Tuple[Point, ...]] = None
    semicircle_candidate: Optional[_LocalSemicircleCandidate] = None
    if roi_center is not None and roi_half_size is not None:
        if background_geometry is not None:
            semicircle_candidate = _fit_semicircle_from_background_boundary(
                background_geometry,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "roi_background_connected_notch_arc"
        else:
            semicircle_candidate = _detect_semicircle_in_roi(
                edge,
                center,
                radius,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "geometry_edge_manual_roi_semicircle"
        if semicircle_candidate is None:
            found = False
            confidence = 0.0
            candidate_support = 0.0
            peak_depth = 0.0
            notch_width_deg = 0.0
            notch_width_px = 0.0
            candidate_indices = np.asarray((), dtype=np.int64)
        else:
            local_arc = semicircle_candidate.arc_points
            semicircle_center = np.asarray(
                semicircle_candidate.center, dtype=np.float64
            )
            wafer_center_array = np.asarray(center, dtype=np.float64)
            outward_vector = semicircle_center - wafer_center_array
            outward_length = float(np.linalg.norm(outward_vector))
            if outward_length <= 1e-6:
                found = False
                outward_unit = np.asarray((0.0, 1.0), dtype=np.float64)
            else:
                outward_unit = outward_vector / outward_length
            inward_unit = -outward_unit
            notch_half_width = float(
                semicircle_candidate.radius_x
                if semicircle_candidate.radius_x is not None
                else semicircle_candidate.radius
            )
            notch_height = float(
                semicircle_candidate.radius_y
                if semicircle_candidate.radius_y is not None
                else semicircle_candidate.radius
            )
            deepest = semicircle_center + inward_unit * float(
                notch_height
            )
            notch_angle_rad = float(
                math.atan2(float(outward_unit[1]), float(outward_unit[0]))
                % (2.0 * math.pi)
            )
            notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
            notch_deepest_point = (float(deepest[0]), float(deepest[1]))
            notch_point = (
                float(cx + math.cos(notch_angle_rad) * radius),
                float(cy + math.sin(notch_angle_rad) * radius),
            )
            peak_depth = max(
                0.0,
                float(radius)
                - float(np.linalg.norm(deepest - wafer_center_array)),
            )
            notch_width_px = float(2.0 * notch_half_width)
            notch_width_deg = float(math.degrees(
                2.0 * math.asin(
                    min(1.0, notch_half_width / max(radius, 1e-6))
                )
            ))
            candidate_support = float(semicircle_candidate.edge_support)
            found = bool(
                outward_length > 1e-6
                and semicircle_candidate.score >= float(notch_semicircle_min_score)
                and semicircle_candidate.arc_coverage >= 0.55
                and peak_depth >= max(0.60, depth_limit * 0.40)
            )
            confidence = (
                float(semicircle_candidate.score) if found else 0.0
            )
            candidate_indices = np.asarray((), dtype=np.int64)
    if not found:
        confidence = 0.0
        if mode == "error":
            roi_message = ""
            if roi_center is not None:
                roi_message = (
                    f" roi_center=({roi_center[0] / scale:.1f},"
                    f"{roi_center[1] / scale:.1f}),"
                    f" semicircle_score={0.0 if semicircle_candidate is None else semicircle_candidate.score:.3f}."
                )
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                f"search={effective_search_center_angle_deg:.1f}+/-{float(search_half_width_deg):.1f}deg."
                f"{roi_message} Use failure_mode='zero' to return angle 0, or correct the ROI/wafer hints."
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
    if local_arc is not None:
        arc = tuple(
            (float(point[0] * inv_scale), float(point[1] * inv_scale))
            for point in local_arc
        )
    else:
        arc = tuple(
            (
                float((cx + math.cos(angles[index]) * boundary[index]) * inv_scale),
                float((cy + math.sin(angles[index]) * boundary[index]) * inv_scale),
            )
            for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
        )
    if background_geometry is not None:
        contour_full = np.rint(
            background_geometry.wafer_contour.astype(np.float64) * inv_scale
        ).astype(np.int32)
    else:
        contour_stride = max(1, angle_samples // 1440)
        contour_indices = np.arange(0, angle_samples, contour_stride, dtype=np.int64)
        contour_points = np.column_stack((
            cx + np.cos(angles[contour_indices]) * boundary[contour_indices],
            cy + np.sin(angles[contour_indices]) * boundary[contour_indices],
        ))
        contour_full = np.rint(contour_points * inv_scale).astype(np.int32).reshape(-1, 1, 2)
    if background_geometry is None:
        palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
        background_threshold = 0.0
    else:
        palette_lab_u8 = np.clip(
            np.rint(background_geometry.palette_lab), 0, 255
        ).astype(np.uint8).reshape(1, -1, 3)
        converted_palette = cv2.cvtColor(palette_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        palette_bgr = tuple(
            tuple(int(value) for value in colour) for colour in converted_palette
        )
        background_threshold = float(background_geometry.distance_threshold_lab)
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
        detection_method=detection_method,
        search_center_angle_deg=effective_search_center_angle_deg,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=float(candidate_support),
        circle_fit_residual_px=float(circle_fit_noise * inv_scale),
        roi_center_px=(
            None
            if roi_center is None
            else (float(roi_center[0] * inv_scale), float(roi_center[1] * inv_scale))
        ),
        roi_bounds_px=(
            None
            if semicircle_candidate is None
            else tuple(float(value * inv_scale) for value in semicircle_candidate.roi_bounds)
        ),
        semicircle_center_px=(
            None
            if semicircle_candidate is None
            else (
                float(semicircle_candidate.center[0] * inv_scale),
                float(semicircle_candidate.center[1] * inv_scale),
            )
        ),
        semicircle_radius_px=(
            None
            if semicircle_candidate is None
            else float(semicircle_candidate.radius * inv_scale)
        ),
        semicircle_radius_x_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_x
                    if semicircle_candidate.radius_x is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_radius_y_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_y
                    if semicircle_candidate.radius_y is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_shape=(
            "none" if semicircle_candidate is None else semicircle_candidate.shape
        ),
        semicircle_score=(
            0.0 if semicircle_candidate is None else float(semicircle_candidate.score)
        ),
        semicircle_fit_residual_px=(
            0.0
            if semicircle_candidate is None
            else float(semicircle_candidate.fit_residual * inv_scale)
        ),
        background_segmentation_used=background_geometry is not None,
        background_palette_bgr=palette_bgr,
        background_distance_threshold_lab=background_threshold,
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


def _transform_result_for_visual(
    result: NotchAngleResult,
    *,
    scale: float = 1.0,
    offset: Point = (0.0, 0.0),
) -> NotchAngleResult:
    def transform_point(point: Point) -> Point:
        return (
            float(point[0]) * float(scale) + float(offset[0]),
            float(point[1]) * float(scale) + float(offset[1]),
        )

    def transform_optional_point(point: Optional[Point]) -> Optional[Point]:
        return None if point is None else transform_point(point)

    contour = result.wafer_contour_px.astype(np.float64) * float(scale)
    contour[:, :, 0] += float(offset[0])
    contour[:, :, 1] += float(offset[1])
    return replace(
        result,
        wafer_center_px=transform_point(result.wafer_center_px),
        wafer_radius_px=float(result.wafer_radius_px) * float(scale),
        notch_point_px=transform_point(result.notch_point_px),
        notch_deepest_point_px=transform_point(result.notch_deepest_point_px),
        notch_depth_px=float(result.notch_depth_px) * float(scale),
        notch_width_px=float(result.notch_width_px) * float(scale),
        radial_noise_px=float(result.radial_noise_px) * float(scale),
        candidate_arc_px=tuple(transform_point(point) for point in result.candidate_arc_px),
        wafer_contour_px=np.rint(contour).astype(np.int32),
        circle_fit_residual_px=float(result.circle_fit_residual_px) * float(scale),
        roi_center_px=transform_optional_point(result.roi_center_px),
        roi_bounds_px=(
            None
            if result.roi_bounds_px is None
            else (
                float(result.roi_bounds_px[0]) * float(scale) + float(offset[0]),
                float(result.roi_bounds_px[1]) * float(scale) + float(offset[1]),
                float(result.roi_bounds_px[2]) * float(scale) + float(offset[0]),
                float(result.roi_bounds_px[3]) * float(scale) + float(offset[1]),
            )
        ),
        semicircle_center_px=transform_optional_point(result.semicircle_center_px),
        semicircle_radius_px=(
            None
            if result.semicircle_radius_px is None
            else float(result.semicircle_radius_px) * float(scale)
        ),
        semicircle_radius_x_px=(
            None
            if result.semicircle_radius_x_px is None
            else float(result.semicircle_radius_x_px) * float(scale)
        ),
        semicircle_radius_y_px=(
            None
            if result.semicircle_radius_y_px is None
            else float(result.semicircle_radius_y_px) * float(scale)
        ),
        semicircle_fit_residual_px=(
            float(result.semicircle_fit_residual_px) * float(scale)
        ),
    )


def make_notch_overlay(
    image: ImageInput,
    result: NotchAngleResult,
    *,
    thickness: int = 2,
    max_dimension: Optional[int] = None,
) -> np.ndarray:
    """Visualise the user-confirmed outer reference and deepest diagnostic."""

    source = _load_bgr(image)
    if max_dimension is not None and int(max_dimension) > 0:
        visual_scale = min(
            1.0,
            float(max_dimension) / max(source.shape[0], source.shape[1]),
        )
    else:
        visual_scale = 1.0
    if visual_scale < 1.0:
        source = cv2.resize(
            source,
            None,
            fx=visual_scale,
            fy=visual_scale,
            interpolation=cv2.INTER_AREA,
        )
        result = _transform_result_for_visual(result, scale=visual_scale)
        thickness = max(1, int(round(float(thickness) * visual_scale)))
    overlay = source.copy()
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
    if result.roi_bounds_px is not None:
        x0, y0, x1, y1 = (int(round(value)) for value in result.roi_bounds_px)
        cv2.rectangle(
            overlay, (x0, y0), (x1, y1), (255, 0, 255),
            max(2, thickness), cv2.LINE_AA
        )
    if result.roi_center_px is not None:
        roi_center = tuple(int(round(value)) for value in result.roi_center_px)
        cv2.drawMarker(
            overlay, roi_center, (255, 0, 255), cv2.MARKER_CROSS,
            max(12, thickness * 8), max(2, thickness), cv2.LINE_AA
        )
    if result.semicircle_center_px is not None and result.semicircle_radius_px is not None:
        local_center = tuple(
            int(round(value)) for value in result.semicircle_center_px
        )
        if result.candidate_arc_px:
            fitted_arc = np.rint(
                np.asarray(result.candidate_arc_px)
            ).astype(np.int32)
            cv2.polylines(
                overlay, [fitted_arc], False, (255, 180, 0),
                max(2, thickness), cv2.LINE_AA
            )
        cv2.circle(
            overlay, local_center, max(4, thickness * 2),
            (255, 180, 0), -1, cv2.LINE_AA
        )
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
        f"circle_residual={result.circle_fit_residual_px:.2f}px  "
        f"arc={result.semicircle_shape}:{result.semicircle_score:.3f}  "
        f"arc_fit={result.semicircle_fit_residual_px:.2f}px"
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overlay, diagnostic_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    if result.background_segmentation_used:
        palette_text = "/".join(
            f"{colour[0]},{colour[1]},{colour[2]}"
            for colour in result.background_palette_bgr
        )
        background_text = (
            f"ROI exterior BGR={palette_text}  "
            f"LAB distance<={result.background_distance_threshold_lab:.1f}"
        )
        cv2.putText(
            overlay, background_text, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            overlay, background_text, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
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

    source = _load_bgr(image)
    height, width = source.shape[:2]
    crop_size = int(size_px or max(80, round(result.wafer_radius_px * 0.13)))
    # Centre the crop between the outer reference and inner apex so both remain
    # visible even when the outer reference is very close to the image border.
    cx = int(round((result.notch_point_px[0] + result.notch_deepest_point_px[0]) / 2.0))
    cy = int(round((result.notch_point_px[1] + result.notch_deepest_point_px[1]) / 2.0))
    x0, x1 = max(0, cx - crop_size), min(width, cx + crop_size)
    y0, y1 = max(0, cy - crop_size), min(height, cy + crop_size)
    crop = source[y0:y1, x0:x1]
    local_result = _transform_result_for_visual(
        result,
        offset=(-float(x0), -float(y0)),
    )
    crop = make_notch_overlay(crop, local_result, thickness=2)
    return cv2.resize(
        crop, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST
    )


def make_notch_background_debug_contact_sheet(
    image: ImageInput,
    *,
    notch_roi_center_px: Point,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    max_dimension: int = 1536,
    background_palette_size: int = 3,
    background_outer_band_fraction: float = 0.28,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 4.0,
    background_morph_px: float = 24.0,
) -> np.ndarray:
    """Return six labelled stages of the ROI-background notch pipeline."""

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    analysis_scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if analysis_scale < 1.0:
        work = cv2.resize(
            source, None, fx=analysis_scale, fy=analysis_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        work = source
    work_height, work_width = work.shape[:2]
    roi_center = (
        float(notch_roi_center_px[0]) * analysis_scale,
        float(notch_roi_center_px[1]) * analysis_scale,
    )
    roi_half_size = _normalise_roi_half_size(
        notch_roi_half_size_px, scale=analysis_scale
    )
    if wafer_center_hint_px is None:
        center_hint = (work_width / 2.0, work_height / 2.0)
    else:
        center_hint = (
            float(wafer_center_hint_px[0]) * analysis_scale,
            float(wafer_center_hint_px[1]) * analysis_scale,
        )
    radius_range = (
        None
        if notch_semicircle_radius_range_px is None
        else (
            float(notch_semicircle_radius_range_px[0]) * analysis_scale,
            float(notch_semicircle_radius_range_px[1]) * analysis_scale,
        )
    )
    geometry = _learn_background_from_notch_roi(
        work,
        roi_center,
        roi_half_size,
        center_hint,
        palette_size=background_palette_size,
        outer_band_fraction=background_outer_band_fraction,
        distance_threshold_lab=background_distance_threshold_lab,
        noise_margin_lab=background_noise_margin_lab,
        morph_size_px=float(background_morph_px) * analysis_scale,
    )
    candidate = _fit_semicircle_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )
    result = detect_wafer_notch(
        work,
        max_dimension=max_dimension,
        wafer_center_hint_px=(
            None if wafer_center_hint_px is None else center_hint
        ),
        wafer_radius_hint_px=(
            None
            if wafer_radius_hint_px is None
            else float(wafer_radius_hint_px) * analysis_scale
        ),
        notch_roi_center_px=roi_center,
        notch_roi_half_size_px=roi_half_size,
        notch_semicircle_radius_range_px=radius_range,
        notch_background_palette_size=background_palette_size,
        notch_background_outer_band_fraction=background_outer_band_fraction,
        notch_background_distance_threshold_lab=background_distance_threshold_lab,
        notch_background_noise_margin_lab=background_noise_margin_lab,
        notch_background_morph_px=float(background_morph_px) * analysis_scale,
        failure_mode="zero",
    )
    final_overlay = make_notch_overlay(work, result)

    x0, y0, x1, y1 = geometry.roi_bounds
    roi_source = work[y0:y1, x0:x1].copy()
    sample_local = geometry.sample_mask[y0:y1, x0:x1] > 0
    tint = np.full_like(roi_source, (255, 0, 255))
    roi_source[sample_local] = cv2.addWeighted(
        roi_source[sample_local], 0.45, tint[sample_local], 0.55, 0.0
    )

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    nearest_distance = np.full((work_height, work_width), np.inf, dtype=np.float32)
    for colour in geometry.palette_lab:
        delta = lab - colour.reshape(1, 1, 3)
        np.minimum(
            nearest_distance,
            np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32),
            out=nearest_distance,
        )
    distance_roi = nearest_distance[y0:y1, x0:x1]
    distance_u8 = np.clip(
        distance_roi / max(1.0, geometry.distance_threshold_lab * 2.0) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    distance_colour = cv2.applyColorMap(distance_u8, cv2.COLORMAP_TURBO)
    background_like = cv2.cvtColor(
        geometry.background_like_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR
    )
    exterior = cv2.cvtColor(
        geometry.exterior_background_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR
    )

    silhouette = cv2.cvtColor(geometry.wafer_mask, cv2.COLOR_GRAY2BGR)
    circle_center = tuple(int(round(value)) for value in geometry.wafer_center)
    cv2.circle(
        silhouette, circle_center, int(round(geometry.wafer_radius)),
        (255, 255, 0), 3, cv2.LINE_AA
    )
    cv2.rectangle(silhouette, (x0, y0), (x1, y1), (255, 0, 255), 3, cv2.LINE_AA)
    final_roi = final_overlay[y0:y1, x0:x1]
    if candidate is not None:
        local_center = (
            int(round(candidate.center[0] - x0)),
            int(round(candidate.center[1] - y0)),
        )
        cv2.drawMarker(
            final_roi, local_center, (255, 180, 0), cv2.MARKER_CROSS,
            18, 2, cv2.LINE_AA
        )

    panel_width, panel_height = 440, 320

    def panel(image_value: np.ndarray, label: str) -> np.ndarray:
        canvas = np.full((panel_height, panel_width, 3), 24, dtype=np.uint8)
        available_height = panel_height - 40
        resize_scale = min(
            panel_width / max(1, image_value.shape[1]),
            available_height / max(1, image_value.shape[0]),
        )
        resized = cv2.resize(
            image_value,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA if resize_scale < 1.0 else cv2.INTER_NEAREST,
        )
        left = (panel_width - resized.shape[1]) // 2
        top = 34 + (available_height - resized.shape[0]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.putText(
            canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (245, 245, 245), 1, cv2.LINE_AA
        )
        return canvas

    panels = [
        panel(roi_source, "1  ROI: outward background sample"),
        panel(distance_colour, "2  LAB distance to background"),
        panel(background_like, "3  Background-like mask"),
        panel(exterior, "4  Border-connected background"),
        panel(silhouette, "5  Wafer mask + robust circle"),
        panel(final_roi, "6  Notch arc + final angle"),
    ]
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def draw_aligned_wafer_notch_guide(
    aligned_image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    search_half_width_deg: float = 70.0,
    bg_threshold: int = 20,
    wafer_morph_kernel: int = 25,
    silhouette_open_kernel: int = 3,
    angle_samples: int = 14400,
    radial_samples: int = 200,
    min_notch_depth_px: float = 4.0,
    noise_margin_px: float = 3.0,
    min_notch_span_deg: float = 0.06,
    smooth_deg: float = 0.25,
    failure_mode: Literal["error", "zero"] = "zero",
    thickness: Optional[int] = None,
    draw_text: bool = True,
) -> AlignedNotchGuideResult:
    """Detect and draw V5 wafer/notch geometry on an already aligned image.

    This diagnostic deliberately mirrors the proven V5 approach: the largest
    non-black component is closed/opened, its minimum enclosing circle supplies
    the wafer ring, and a dense radial scan searches only the lower sector for
    an inward notch. It does not rotate the image or build a die map.

    The returned ``overlay_image`` has the same shape as the input and is a
    writable BGR copy. All returned coordinates refer to that image. The green
    line is the requested alignment reference, the red line is the measured
    notch direction, and the small orange arc between them is the residual
    angle after alignment.

    ``failure_mode="zero"`` still returns the wafer ring/search lines when the
    notch is absent. ``failure_mode="error"`` raises ``RuntimeError`` instead.
    """

    mode = str(failure_mode).strip().lower()
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 1.0 <= float(search_half_width_deg) <= 170.0:
        raise ValueError("search_half_width_deg must be between 1 and 170 degrees.")
    if int(angle_samples) < 720:
        raise ValueError("angle_samples must be at least 720.")
    if int(radial_samples) < 32:
        raise ValueError("radial_samples must be at least 32.")
    if float(min_notch_depth_px) < 0.0 or float(noise_margin_px) < 0.0:
        raise ValueError("notch depth thresholds must be non-negative.")

    source = _load_bgr(aligned_image)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # V5 detect_wafer(): largest threshold component after a 25x25 close/open,
    # followed by minEnclosingCircle. Kernel sizes remain configurable only so
    # production images can be tuned without changing the implementation.
    _, wafer_mask = cv2.threshold(
        gray, int(bg_threshold), 255, cv2.THRESH_BINARY
    )
    morph_size = max(1, int(wafer_morph_kernel))
    if morph_size > 1:
        morph_size |= 1
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_CLOSE, morph_kernel
        )
        wafer_mask = cv2.morphologyEx(
            wafer_mask, cv2.MORPH_OPEN, morph_kernel
        )
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "V5 wafer ring was not found. Adjust bg_threshold for this image."
        )
    wafer_contour = max(contours, key=cv2.contourArea)
    (wafer_cx, wafer_cy), wafer_radius = cv2.minEnclosingCircle(wafer_contour)
    wafer_cx = float(round(wafer_cx))
    wafer_cy = float(round(wafer_cy))
    wafer_radius = float(round(wafer_radius))
    if wafer_radius <= 0.0:
        raise RuntimeError("V5 wafer ring radius is invalid.")

    # V5 _wafer_silhouette(): optional light opening and largest connected
    # contour. Unlike the closed mask above, this keeps the notch concavity.
    _, silhouette_mask = cv2.threshold(
        gray, int(bg_threshold), 255, cv2.THRESH_BINARY
    )
    open_size = max(1, int(silhouette_open_kernel))
    if open_size >= 3:
        open_size |= 1
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        silhouette_mask = cv2.morphologyEx(
            silhouette_mask, cv2.MORPH_OPEN, open_kernel
        )
    silhouette_contours, _ = cv2.findContours(
        silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    silhouette = np.zeros_like(gray, dtype=np.uint8)
    if silhouette_contours:
        silhouette_contour = max(silhouette_contours, key=cv2.contourArea)
        cv2.drawContours(silhouette, [silhouette_contour], -1, 1, -1)

    sample_count = int(angle_samples)
    angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    radii_axis = np.linspace(
        wafer_radius * 0.93,
        wafer_radius * 1.015,
        int(radial_samples),
    )
    xs = (
        wafer_cx + radii_axis[None, :] * np.cos(angles)[:, None]
    ).astype(np.int32)
    ys = (
        wafer_cy + radii_axis[None, :] * np.sin(angles)[:, None]
    ).astype(np.int32)
    np.clip(xs, 0, width - 1, out=xs)
    np.clip(ys, 0, height - 1, out=ys)
    on_wafer = silhouette[ys, xs] > 0
    last_indices = np.where(
        on_wafer.any(axis=1),
        on_wafer.shape[1] - 1 - np.argmax(on_wafer[:, ::-1], axis=1),
        0,
    )
    boundary_radii = radii_axis[last_indices]
    depth = np.median(boundary_radii) - boundary_radii

    smooth_window = max(
        3, int(round(float(smooth_deg) / 360.0 * sample_count))
    )
    if smooth_window >= 3:
        smooth_kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        padded = np.concatenate(
            (depth[-smooth_window:], depth, depth[:smooth_window])
        )
        depth = np.convolve(padded, smooth_kernel, mode="same")[
            smooth_window:smooth_window + sample_count
        ]

    angles_deg = np.degrees(angles)
    distance_from_reference = np.abs(
        (angles_deg - float(reference_angle_deg) + 180.0) % 360.0 - 180.0
    )
    in_sector = distance_from_reference <= float(search_half_width_deg)
    active = (depth > float(min_notch_depth_px)) & in_sector
    groups = _circular_candidate_groups(active)
    degree_step = 360.0 / sample_count
    groups = [
        group for group in groups
        if len(group) * degree_step >= float(min_notch_span_deg)
    ]
    candidate_indices = (
        max(groups, key=lambda group: float(depth[group].sum()))
        if groups else np.asarray((), dtype=np.int64)
    )
    outside_depth = depth[~in_sector]
    noise_floor = (
        float(np.percentile(outside_depth, 99.5))
        if outside_depth.size else 0.0
    )
    effective_threshold = max(
        float(min_notch_depth_px), noise_floor + float(noise_margin_px)
    )
    found = bool(
        candidate_indices.size
        and float(np.max(depth[candidate_indices])) >= effective_threshold
    )

    notch_center: Optional[Point] = None
    notch_point: Optional[Point] = None
    notch_left: Optional[Point] = None
    notch_right: Optional[Point] = None
    notch_angle: Optional[float] = None
    residual_angle = 0.0
    notch_depth = 0.0
    notch_width = 0.0
    candidate_arc: Tuple[Point, ...] = ()

    if found:
        candidate_depth = depth[candidate_indices]
        candidate_angles = angles[candidate_indices]
        weight_sum = float(candidate_depth.sum())
        notch_angle = float(
            math.degrees(
                math.atan2(
                    float((np.sin(candidate_angles) * candidate_depth).sum()),
                    float((np.cos(candidate_angles) * candidate_depth).sum()),
                )
            ) % 360.0
        )
        residual_angle = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
        notch_radius_values = boundary_radii[candidate_indices]
        boundary_x = wafer_cx + notch_radius_values * np.cos(candidate_angles)
        boundary_y = wafer_cy + notch_radius_values * np.sin(candidate_angles)
        notch_center = (
            float((boundary_x * candidate_depth).sum() / weight_sum),
            float((boundary_y * candidate_depth).sum() / weight_sum),
        )
        notch_angle_rad = math.radians(notch_angle)
        notch_point = (
            float(wafer_cx + wafer_radius * math.cos(notch_angle_rad)),
            float(wafer_cy + wafer_radius * math.sin(notch_angle_rad)),
        )
        notch_left = (float(boundary_x[0]), float(boundary_y[0]))
        notch_right = (float(boundary_x[-1]), float(boundary_y[-1]))
        notch_depth = float(np.max(candidate_depth))
        notch_width = float(len(candidate_indices) * degree_step)
        arc_stride = max(1, len(candidate_indices) // 256)
        candidate_arc = tuple(
            (float(boundary_x[index]), float(boundary_y[index]))
            for index in range(0, len(candidate_indices), arc_stride)
        )
    elif mode == "error":
        raise RuntimeError(
            "V5 notch was not found in the aligned image: "
            f"effective_depth_threshold={effective_threshold:.2f}px, "
            f"search={float(reference_angle_deg):.1f}+/-"
            f"{float(search_half_width_deg):.1f}deg."
        )

    overlay = source.copy()
    line_width = (
        max(1, int(thickness))
        if thickness is not None
        else max(2, int(round(max(height, width) / 3000.0)))
    )
    center_int = (int(round(wafer_cx)), int(round(wafer_cy)))
    radius_int = int(round(wafer_radius))

    # Actual threshold contour (gray) and ideal V5 enclosing ring (cyan).
    cv2.drawContours(
        overlay, [wafer_contour], -1, (150, 150, 150),
        max(1, line_width // 2), cv2.LINE_AA
    )
    cv2.circle(
        overlay, center_int, radius_int, (255, 255, 0), line_width, cv2.LINE_AA
    )
    cv2.drawMarker(
        overlay, center_int, (255, 0, 0), cv2.MARKER_CROSS,
        max(12, line_width * 8), max(1, line_width), cv2.LINE_AA
    )

    def ring_point(angle_deg: float, ratio: float = 1.0) -> Tuple[int, int]:
        angle_rad = math.radians(float(angle_deg))
        return (
            int(round(wafer_cx + wafer_radius * ratio * math.cos(angle_rad))),
            int(round(wafer_cy + wafer_radius * ratio * math.sin(angle_rad))),
        )

    # Search-sector limits are magenta; aligned reference is green.
    for search_angle in (
        float(reference_angle_deg) - float(search_half_width_deg),
        float(reference_angle_deg) + float(search_half_width_deg),
    ):
        cv2.line(
            overlay, center_int, ring_point(search_angle),
            (255, 0, 255), max(1, line_width // 2), cv2.LINE_AA
        )
    reference_endpoint = ring_point(reference_angle_deg)
    cv2.arrowedLine(
        overlay, center_int, reference_endpoint, (0, 220, 0),
        line_width, cv2.LINE_AA, tipLength=0.018
    )

    if found and notch_angle is not None and notch_point is not None:
        if candidate_arc:
            arc_points = np.rint(np.asarray(candidate_arc)).astype(np.int32)
            cv2.polylines(
                overlay, [arc_points], False, (0, 255, 255),
                max(2, line_width * 2), cv2.LINE_AA
            )
        for boundary_point in (notch_left, notch_right):
            if boundary_point is not None:
                point_int = tuple(int(round(value)) for value in boundary_point)
                cv2.line(
                    overlay, center_int, point_int, (0, 165, 255),
                    max(1, line_width // 2), cv2.LINE_AA
                )
                cv2.circle(
                    overlay, point_int, max(4, line_width * 2),
                    (255, 255, 255), -1, cv2.LINE_AA
                )
        notch_endpoint = tuple(int(round(value)) for value in notch_point)
        cv2.arrowedLine(
            overlay, center_int, notch_endpoint, (0, 0, 255),
            max(2, line_width * 2), cv2.LINE_AA, tipLength=0.018
        )
        if notch_center is not None:
            cv2.circle(
                overlay,
                tuple(int(round(value)) for value in notch_center),
                max(5, line_width * 3), (0, 128, 255), -1, cv2.LINE_AA
            )
        cv2.circle(
            overlay, notch_endpoint, max(7, line_width * 4),
            (0, 0, 255), -1, cv2.LINE_AA
        )

        # Draw the signed shortest arc from the green reference to the red
        # detected direction. This is the residual angle in the aligned image.
        arc_count = max(8, int(abs(residual_angle) * 2.0) + 2)
        guide_angles = np.linspace(
            float(reference_angle_deg),
            float(reference_angle_deg) + residual_angle,
            arc_count,
        )
        guide_radius = wafer_radius * 0.24
        guide_arc = np.rint(np.column_stack((
            wafer_cx + guide_radius * np.cos(np.radians(guide_angles)),
            wafer_cy + guide_radius * np.sin(np.radians(guide_angles)),
        ))).astype(np.int32)
        cv2.polylines(
            overlay, [guide_arc], False, (0, 128, 255),
            max(2, line_width), cv2.LINE_AA
        )

    if draw_text:
        if found and notch_angle is not None:
            summary = (
                f"V5 found=True  notch={notch_angle:.4f} deg  "
                f"aligned residual={residual_angle:+.4f} deg"
            )
            detail = (
                f"center=({wafer_cx:.1f},{wafer_cy:.1f})  "
                f"radius={wafer_radius:.1f}px  depth={notch_depth:.2f}px  "
                f"width={notch_width:.3f}deg"
            )
        else:
            summary = "V5 found=False  residual=+0.0000 deg"
            detail = (
                f"center=({wafer_cx:.1f},{wafer_cy:.1f})  "
                f"radius={wafer_radius:.1f}px  "
                f"threshold={effective_threshold:.2f}px"
            )
        font_scale = max(0.55, min(1.2, max(height, width) / 9000.0))
        text_x = max(12, line_width * 6)
        text_y = max(34, line_width * 17)
        for row, text_value in enumerate((summary, detail)):
            y = text_y + row * int(round(34 * font_scale))
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), max(3, line_width * 3), cv2.LINE_AA
            )
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), max(1, line_width), cv2.LINE_AA
            )

    return AlignedNotchGuideResult(
        overlay_image=overlay,
        found=found,
        wafer_center_px=(wafer_cx, wafer_cy),
        wafer_radius_px=wafer_radius,
        notch_center_px=notch_center,
        notch_point_px=notch_point,
        notch_left_px=notch_left,
        notch_right_px=notch_right,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg) % 360.0,
        residual_angle_deg=float(residual_angle),
        notch_depth_px=notch_depth,
        notch_width_deg=notch_width,
        effective_depth_threshold_px=float(effective_threshold),
        candidate_arc_px=candidate_arc,
        wafer_contour_px=wafer_contour.copy(),
        search_center_angle_deg=float(reference_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        detection_method="v5_silhouette_radial_aligned",
    )
