"""V5 wafer-ring/notch guide with border-adaptive background segmentation.

This file is intentionally separate from the existing notch implementation.
It keeps the V5 geometry (largest contour -> minEnclosingCircle -> bottom-sector
radial scan), but learns the outside colour from the image border instead of
assuming that the background is black. It is self-contained and can be copied
as one file.

The public function does not rotate the image or build a die map. It accepts an
already aligned wafer image and returns a writable full-resolution overlay plus
the measured ring, notch points, and residual angle.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]

__all__ = [
    "AdaptiveBackgroundNotchGuideResult",
    "draw_aligned_wafer_notch_guide",
]


@dataclass(frozen=True)
class AdaptiveBackgroundNotchGuideResult:
    overlay_image: np.ndarray = field(repr=False)
    found: bool
    wafer_center_px: Point
    wafer_radius_px: float
    notch_center_px: Optional[Point]
    notch_deepest_point_px: Optional[Point]
    notch_point_px: Optional[Point]
    notch_left_px: Optional[Point]
    notch_right_px: Optional[Point]
    notch_angle_deg: Optional[float]
    reference_angle_deg: float
    residual_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    effective_depth_threshold_px: float
    candidate_arc_px: Tuple[Point, ...] = field(repr=False)
    wafer_contour_px: np.ndarray = field(repr=False)
    background_palette_bgr: Tuple[Tuple[int, int, int], ...]
    segmentation_threshold_lab: float
    analysis_scale: float
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


def _odd_kernel(value: float, *, minimum: int = 1) -> int:
    size = max(int(minimum), int(round(float(value))))
    return size if size % 2 else size + 1


def _border_pixels(image: np.ndarray, band_px: int) -> np.ndarray:
    height, width = image.shape[:2]
    band = max(2, min(int(band_px), height // 4, width // 4))
    return np.concatenate((
        image[:band, :, :].reshape(-1, 3),
        image[-band:, :, :].reshape(-1, 3),
        image[band:-band, :band, :].reshape(-1, 3),
        image[band:-band, -band:, :].reshape(-1, 3),
    ), axis=0)


def _learn_background_palette(
    image_bgr: np.ndarray,
    *,
    border_band_px: int,
    palette_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return LAB palette, BGR palette, and border residual percentile."""

    border_bgr = _border_pixels(image_bgr, border_band_px)
    if len(border_bgr) > 40000:
        step = int(math.ceil(len(border_bgr) / 40000.0))
        border_bgr = border_bgr[::step]
    border_lab = cv2.cvtColor(
        border_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    unique_count = len(np.unique(border_bgr, axis=0))
    cluster_count = max(1, min(int(palette_size), unique_count, len(border_lab)))
    if cluster_count == 1:
        palette_lab = np.median(border_lab, axis=0, keepdims=True).astype(np.float32)
    else:
        cv2.setRNGSeed(20260828)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.05,
        )
        _, _, palette_lab = cv2.kmeans(
            border_lab,
            cluster_count,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        palette_lab = palette_lab.astype(np.float32)

    border_min_sq = np.full(len(border_lab), np.inf, dtype=np.float32)
    for colour in palette_lab:
        delta = border_lab - colour[None, :]
        border_min_sq = np.minimum(
            border_min_sq, np.sum(delta * delta, axis=1)
        )
    border_residual = float(np.percentile(np.sqrt(border_min_sq), 99.5))

    palette_u8 = np.clip(np.rint(palette_lab), 0, 255).astype(np.uint8)
    palette_bgr = cv2.cvtColor(
        palette_u8.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR
    ).reshape(-1, 3)
    return palette_lab, palette_bgr, border_residual


def _background_difference_mask(
    image_bgr: np.ndarray,
    palette_lab: np.ndarray,
    threshold_lab: float,
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    minimum_sq = np.full(lab.shape[:2], np.inf, dtype=np.float32)
    for colour in palette_lab:
        delta = lab - colour.reshape(1, 1, 3)
        distance_sq = (
            delta[:, :, 0] * delta[:, :, 0]
            + delta[:, :, 1] * delta[:, :, 1]
            + delta[:, :, 2] * delta[:, :, 2]
        )
        np.minimum(minimum_sq, distance_sq, out=minimum_sq)
    return (minimum_sq > float(threshold_lab) ** 2).astype(np.uint8) * 255


def _candidate_groups(active: np.ndarray):
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


def draw_aligned_wafer_notch_guide(
    aligned_image: ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    search_center_angle_deg: Optional[float] = None,
    search_half_width_deg: float = 70.0,
    max_analysis_dimension: int = 3072,
    border_band_px: int = 16,
    background_palette_size: int = 3,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 6.0,
    min_background_distance_lab: float = 8.0,
    wafer_morph_kernel: int = 25,
    silhouette_open_kernel: int = 3,
    angle_samples: int = 14400,
    radial_samples: int = 200,
    min_notch_depth_px: Optional[float] = 4.0,
    min_notch_depth_ratio: float = 0.001,
    noise_margin_px: float = 3.0,
    min_notch_span_deg: float = 0.06,
    smooth_deg: float = 0.25,
    wafer_center_hint_px: Optional[Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    failure_mode: Literal["error", "zero"] = "zero",
    thickness: Optional[int] = None,
    draw_text: bool = True,
) -> AdaptiveBackgroundNotchGuideResult:
    """Draw V5 ring/notch geometry after learning background from the border.

    The input may be a path or a memory BGR ndarray. Returned coordinates and
    ``overlay_image`` always use the original input resolution. The image is
    not rotated. A large image is downscaled only for analysis, then all points
    are converted back before drawing.

    Use ``failure_mode="zero"`` to keep the ring/search overlay when no notch is
    found, or ``"error"`` to raise ``RuntimeError``.
    """

    mode = str(failure_mode).strip().lower()
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    if not 1.0 <= float(search_half_width_deg) <= 170.0:
        raise ValueError("search_half_width_deg must be between 1 and 170 degrees.")
    if int(angle_samples) < 720 or int(radial_samples) < 32:
        raise ValueError("angle_samples >= 720 and radial_samples >= 32 are required.")
    if int(background_palette_size) < 1:
        raise ValueError("background_palette_size must be at least 1.")

    source = _load_bgr(aligned_image)
    full_height, full_width = source.shape[:2]
    analysis_limit = max(512, int(max_analysis_dimension))
    scale = min(1.0, analysis_limit / float(max(full_height, full_width)))
    if scale < 1.0:
        work = cv2.resize(
            source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        work = source
    height, width = work.shape[:2]

    scaled_border_band = max(2, int(round(float(border_band_px) * scale)))
    palette_lab, palette_bgr, border_residual = _learn_background_palette(
        work,
        border_band_px=scaled_border_band,
        palette_size=int(background_palette_size),
    )
    if background_distance_threshold_lab is None:
        segmentation_threshold = max(
            float(min_background_distance_lab),
            border_residual + float(background_noise_margin_lab),
        )
    else:
        segmentation_threshold = float(background_distance_threshold_lab)
    if segmentation_threshold <= 0.0:
        raise ValueError("background distance threshold must be positive.")

    raw_mask = _background_difference_mask(
        work, palette_lab, segmentation_threshold
    )
    morph_size = _odd_kernel(float(wafer_morph_kernel) * scale)
    wafer_mask = raw_mask.copy()
    if morph_size >= 3:
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
            "Adaptive-background wafer ring was not found. "
            "Adjust background_distance_threshold_lab or border_band_px."
        )
    wafer_contour = max(contours, key=cv2.contourArea)
    (wafer_cx, wafer_cy), wafer_radius = cv2.minEnclosingCircle(wafer_contour)
    if wafer_center_hint_px is not None:
        wafer_cx = float(wafer_center_hint_px[0]) * scale
        wafer_cy = float(wafer_center_hint_px[1]) * scale
    if wafer_radius_hint_px is not None:
        wafer_radius = float(wafer_radius_hint_px) * scale
    if wafer_radius <= min(height, width) * 0.20:
        raise RuntimeError("Adaptive-background wafer radius is implausibly small.")

    open_size = _odd_kernel(float(silhouette_open_kernel) * scale)
    silhouette_mask = raw_mask.copy()
    if open_size >= 3:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        silhouette_mask = cv2.morphologyEx(
            silhouette_mask, cv2.MORPH_OPEN, open_kernel
        )
    silhouette_contours, _ = cv2.findContours(
        silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not silhouette_contours:
        raise RuntimeError("Adaptive-background wafer silhouette was not found.")
    silhouette_contour = max(silhouette_contours, key=cv2.contourArea)
    silhouette = np.zeros((height, width), dtype=np.uint8)
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
    smooth_kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    padded = np.concatenate((depth[-smooth_window:], depth, depth[:smooth_window]))
    depth = np.convolve(padded, smooth_kernel, mode="same")[
        smooth_window:smooth_window + sample_count
    ]

    angles_deg = np.degrees(angles)
    search_center = (
        float(reference_angle_deg)
        if search_center_angle_deg is None
        else float(search_center_angle_deg)
    )
    in_sector = np.abs(
        (angles_deg - search_center + 180.0) % 360.0 - 180.0
    ) <= float(search_half_width_deg)
    scaled_min_depth = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(wafer_radius) * float(min_notch_depth_ratio))
    )
    scaled_noise_margin = float(noise_margin_px) * scale
    active = (depth > scaled_min_depth) & in_sector
    degree_step = 360.0 / sample_count
    groups = [
        group for group in _candidate_groups(active)
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
        scaled_min_depth, noise_floor + scaled_noise_margin
    )
    found = bool(
        candidate_indices.size
        and float(np.max(depth[candidate_indices])) >= effective_threshold
    )

    notch_center_work: Optional[Point] = None
    notch_deepest_work: Optional[Point] = None
    notch_point_work: Optional[Point] = None
    notch_left_work: Optional[Point] = None
    notch_right_work: Optional[Point] = None
    notch_angle: Optional[float] = None
    residual_angle = 0.0
    notch_depth_work = 0.0
    notch_width = 0.0
    candidate_arc_work: Tuple[Point, ...] = ()

    if found:
        candidate_depth = depth[candidate_indices]
        candidate_angles = angles[candidate_indices]
        weight_sum = float(candidate_depth.sum())
        notch_angle = float(math.degrees(math.atan2(
            float((np.sin(candidate_angles) * candidate_depth).sum()),
            float((np.cos(candidate_angles) * candidate_depth).sum()),
        )) % 360.0)
        residual_angle = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
        notch_radii = boundary_radii[candidate_indices]
        boundary_x = wafer_cx + notch_radii * np.cos(candidate_angles)
        boundary_y = wafer_cy + notch_radii * np.sin(candidate_angles)
        notch_center_work = (
            float((boundary_x * candidate_depth).sum() / weight_sum),
            float((boundary_y * candidate_depth).sum() / weight_sum),
        )
        peak_local_index = int(np.argmax(candidate_depth))
        notch_deepest_work = (
            float(boundary_x[peak_local_index]),
            float(boundary_y[peak_local_index]),
        )
        notch_angle_rad = math.radians(notch_angle)
        notch_point_work = (
            float(wafer_cx + wafer_radius * math.cos(notch_angle_rad)),
            float(wafer_cy + wafer_radius * math.sin(notch_angle_rad)),
        )
        notch_left_work = (float(boundary_x[0]), float(boundary_y[0]))
        notch_right_work = (float(boundary_x[-1]), float(boundary_y[-1]))
        notch_depth_work = float(np.max(candidate_depth))
        notch_width = float(len(candidate_indices) * degree_step)
        arc_stride = max(1, len(candidate_indices) // 256)
        candidate_arc_work = tuple(
            (float(boundary_x[index]), float(boundary_y[index]))
            for index in range(0, len(candidate_indices), arc_stride)
        )
    elif mode == "error":
        raise RuntimeError(
            "Adaptive-background V5 notch was not found: "
            f"effective_depth_threshold={effective_threshold / scale:.2f}px, "
            f"search={search_center:.1f}+/-"
            f"{float(search_half_width_deg):.1f}deg."
        )

    inv_scale = 1.0 / scale

    def full_point(point: Optional[Point]) -> Optional[Point]:
        if point is None:
            return None
        return float(point[0] * inv_scale), float(point[1] * inv_scale)

    full_center = (float(wafer_cx * inv_scale), float(wafer_cy * inv_scale))
    full_radius = float(wafer_radius * inv_scale)
    notch_center = full_point(notch_center_work)
    notch_deepest = full_point(notch_deepest_work)
    notch_point = full_point(notch_point_work)
    notch_left = full_point(notch_left_work)
    notch_right = full_point(notch_right_work)
    candidate_arc = tuple(
        (float(point[0] * inv_scale), float(point[1] * inv_scale))
        for point in candidate_arc_work
    )
    contour_full = np.rint(
        wafer_contour.astype(np.float64) * inv_scale
    ).astype(np.int32)

    overlay = source.copy()
    line_width = (
        max(1, int(thickness)) if thickness is not None
        else max(2, int(round(max(full_height, full_width) / 3000.0)))
    )
    center_int = tuple(int(round(value)) for value in full_center)
    radius_int = int(round(full_radius))

    cv2.drawContours(
        overlay, [contour_full], -1, (150, 150, 150),
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
            int(round(full_center[0] + full_radius * ratio * math.cos(angle_rad))),
            int(round(full_center[1] + full_radius * ratio * math.sin(angle_rad))),
        )

    for search_angle in (
        search_center - float(search_half_width_deg),
        search_center + float(search_half_width_deg),
    ):
        cv2.line(
            overlay, center_int, ring_point(search_angle),
            (255, 0, 255), max(1, line_width // 2), cv2.LINE_AA
        )
    cv2.arrowedLine(
        overlay, center_int, ring_point(reference_angle_deg), (0, 220, 0),
        line_width, cv2.LINE_AA, tipLength=0.018
    )

    if found and notch_angle is not None and notch_point is not None:
        if candidate_arc:
            cv2.polylines(
                overlay,
                [np.rint(np.asarray(candidate_arc)).astype(np.int32)],
                False,
                (0, 255, 255),
                max(2, line_width * 2),
                cv2.LINE_AA,
            )
        for boundary_point in (notch_left, notch_right):
            if boundary_point is None:
                continue
            point_int = tuple(int(round(value)) for value in boundary_point)
            cv2.line(
                overlay, center_int, point_int, (0, 165, 255),
                max(1, line_width // 2), cv2.LINE_AA
            )
            cv2.circle(
                overlay, point_int, max(4, line_width * 2),
                (255, 255, 255), -1, cv2.LINE_AA
            )
        notch_int = tuple(int(round(value)) for value in notch_point)
        cv2.arrowedLine(
            overlay, center_int, notch_int, (0, 0, 255),
            max(2, line_width * 2), cv2.LINE_AA, tipLength=0.018
        )
        if notch_center is not None:
            cv2.circle(
                overlay,
                tuple(int(round(value)) for value in notch_center),
                max(5, line_width * 3),
                (0, 128, 255),
                -1,
                cv2.LINE_AA,
            )
        cv2.circle(
            overlay, notch_int, max(7, line_width * 4),
            (0, 0, 255), -1, cv2.LINE_AA
        )
        guide_angles = np.linspace(
            float(reference_angle_deg),
            float(reference_angle_deg) + residual_angle,
            max(8, int(abs(residual_angle) * 2.0) + 2),
        )
        guide_radius = full_radius * 0.24
        guide_arc = np.rint(np.column_stack((
            full_center[0] + guide_radius * np.cos(np.radians(guide_angles)),
            full_center[1] + guide_radius * np.sin(np.radians(guide_angles)),
        ))).astype(np.int32)
        cv2.polylines(
            overlay, [guide_arc], False, (0, 128, 255),
            max(2, line_width), cv2.LINE_AA
        )

    if draw_text:
        palette_text = "/".join(
            f"{int(colour[0])},{int(colour[1])},{int(colour[2])}"
            for colour in palette_bgr
        )
        if found and notch_angle is not None:
            summary = (
                f"adaptive V5 found=True  notch={notch_angle:.4f} deg  "
                f"aligned residual={residual_angle:+.4f} deg"
            )
            detail = (
                f"center=({full_center[0]:.1f},{full_center[1]:.1f})  "
                f"radius={full_radius:.1f}px  "
                f"depth={notch_depth_work * inv_scale:.2f}px  "
                f"width={notch_width:.3f}deg"
            )
        else:
            summary = "adaptive V5 found=False  residual=+0.0000 deg"
            detail = (
                f"center=({full_center[0]:.1f},{full_center[1]:.1f})  "
                f"radius={full_radius:.1f}px"
            )
        background_text = (
            f"border BGR={palette_text}  LAB threshold={segmentation_threshold:.2f}"
        )
        font_scale = max(
            0.55, min(1.2, max(full_height, full_width) / 9000.0)
        )
        text_x = max(12, line_width * 6)
        text_y = max(34, line_width * 17)
        for row, text_value in enumerate((summary, detail, background_text)):
            y = text_y + row * int(round(34 * font_scale))
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), max(3, line_width * 3), cv2.LINE_AA
            )
            cv2.putText(
                overlay, text_value, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), max(1, line_width), cv2.LINE_AA
            )

    return AdaptiveBackgroundNotchGuideResult(
        overlay_image=overlay,
        found=found,
        wafer_center_px=full_center,
        wafer_radius_px=full_radius,
        notch_center_px=notch_center,
        notch_deepest_point_px=notch_deepest,
        notch_point_px=notch_point,
        notch_left_px=notch_left,
        notch_right_px=notch_right,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg) % 360.0,
        residual_angle_deg=float(residual_angle),
        notch_depth_px=float(notch_depth_work * inv_scale),
        notch_width_deg=float(notch_width),
        effective_depth_threshold_px=float(effective_threshold * inv_scale),
        candidate_arc_px=candidate_arc,
        wafer_contour_px=contour_full,
        background_palette_bgr=tuple(
            tuple(int(value) for value in colour) for colour in palette_bgr
        ),
        segmentation_threshold_lab=float(segmentation_threshold),
        analysis_scale=float(scale),
        search_center_angle_deg=search_center % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        detection_method="v5_border_adaptive_silhouette_radial_aligned",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw adaptive-background V5 wafer/notch diagnostics."
    )
    parser.add_argument("image", help="Aligned wafer image path")
    parser.add_argument("--output", required=True, help="Output overlay path")
    parser.add_argument("--reference-angle", type=float, default=90.0)
    parser.add_argument("--failure-mode", choices=("error", "zero"), default="zero")
    args = parser.parse_args()

    result = draw_aligned_wafer_notch_guide(
        args.image,
        reference_angle_deg=args.reference_angle,
        failure_mode=args.failure_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result.overlay_image):
        raise RuntimeError(f"Failed to write overlay: {output}")
    print(f"found={result.found}")
    print(f"wafer_center_px={result.wafer_center_px}")
    print(f"wafer_radius_px={result.wafer_radius_px:.3f}")
    print(f"notch_point_px={result.notch_point_px}")
    print(f"notch_angle_deg={result.notch_angle_deg}")
    print(f"residual_angle_deg={result.residual_angle_deg:+.6f}")
    print(f"overlay={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
