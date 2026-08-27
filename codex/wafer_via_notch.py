"""Wafer die-map pipeline whose only angle source is the wafer notch.

YOLO points are used for the centre corner and X/Y pitch only. No YOLO pair,
projection, FFT, or die-render angle estimate is called by this pipeline.
The notch detector follows outside/background penetration through the fitted
wafer circle and therefore does not require one fixed notch shape.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from . import wafer_via as _base
    from .wafer_via import *  # noqa: F401,F403
    from .wafer_notch_angle import (
        NotchAngleResult,
        align_wafer_by_notch,
        detect_wafer_notch,
        make_notch_overlay,
        make_notch_zoom,
    )
except ImportError:
    import wafer_via as _base  # type: ignore[no-redef]
    from wafer_via import *  # type: ignore[no-redef]  # noqa: F401,F403
    from wafer_notch_angle import (  # type: ignore[no-redef]
        NotchAngleResult,
        align_wafer_by_notch,
        detect_wafer_notch,
        make_notch_overlay,
        make_notch_zoom,
    )


ImageInput = Union[str, Path, np.ndarray]
Point = Tuple[float, float]

__all__ = list(_base.__all__) + [
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "make_notch_overlay",
    "make_notch_zoom",
    "estimate_grid_from_yolo_notch",
    "build_die_map_from_yolo",
    "build_die_map",
]


def estimate_grid_from_yolo_notch(
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    notch_correction_angle_deg: float,
    notch_confidence: float,
    reference_point_clip: Optional[Point] = None,
    detection_format: _base.DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: _base.RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    axis_tolerance: float = 0.18,
    perpendicular_tolerance_px: float = 5.0,
) -> _base.GridEstimate:
    """Estimate centre and pitch while taking angle only from the notch."""

    if not 0.0 <= float(refine_min_confidence) <= 1.0:
        raise ValueError("refine_min_confidence must be between 0.0 and 1.0.")
    image = _base._load_bgr(clip_image)
    height, width = image.shape[:2]
    points = _base.parse_yolo_points(
        detections,
        (width, height),
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
    )
    if len(points) < 3:
        raise ValueError(
            f"At least three YOLO cross-points are required; received {len(points)}."
        )
    raw_points = list(points)
    refinement_confidences = [0.0] * len(points)
    if refine:
        refined_points = []
        refinement_confidences = []
        for point in raw_points:
            candidate, candidate_confidence = _base.refine_cross_point(
                image,
                point,
                search_radius=refine_radius,
                max_street_width=refine_max_street_width,
                mode=refine_mode,
                corner_patch_ratio=refine_corner_patch_ratio,
                corner_reference_weight=refine_corner_reference_weight,
                noise_kernel=refine_noise_kernel,
            )
            value = float(candidate_confidence)
            refinement_confidences.append(value)
            refined_points.append(
                candidate if value >= float(refine_min_confidence) else point
            )
        points = refined_points

    array = np.asarray(points, dtype=np.float64)
    selection_reference = np.asarray(
        reference_point_clip
        if reference_point_clip is not None
        else (width / 2.0, height / 2.0),
        dtype=np.float64,
    ).reshape(-1)
    if selection_reference.size != 2 or not np.all(np.isfinite(selection_reference)):
        raise ValueError("reference_point_clip must contain two finite coordinates.")
    center_index = int(
        np.argmin(np.linalg.norm(array - selection_reference, axis=1))
    )
    center = array[center_index]

    # This is the essential change: axes come only from the notch correction.
    angle = math.radians(float(notch_correction_angle_deg))
    axis_x = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    axis_y = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    delta = array - center
    delta[center_index] = 0.0
    side_index, side_vector = _base._select_axis_neighbour(
        delta,
        axis_x,
        axis_y,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    below_index, below_vector = _base._select_axis_neighbour(
        delta,
        axis_y,
        axis_x,
        prefer_positive=True,
        axis_tolerance=axis_tolerance,
        perpendicular_tolerance_px=perpendicular_tolerance_px,
    )
    if side_index == below_index:
        raise ValueError("The same YOLO point was selected for both pitch axes.")

    pitch_x = float(np.linalg.norm(side_vector))
    pitch_y = float(np.linalg.norm(below_vector))
    angle_x = _base._fold_grid_angle(
        math.degrees(math.atan2(float(side_vector[1]), float(side_vector[0])))
    )
    angle_y = _base._fold_grid_angle(
        math.degrees(math.atan2(float(-below_vector[0]), float(below_vector[1])))
    )
    return _base.GridEstimate(
        points_clip=tuple(_base._point(point) for point in array),
        center_corner_clip=_base._point(center),
        side_corner_clip=_base._point(array[side_index]),
        below_corner_clip=_base._point(array[below_index]),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        angle_deg=float(notch_correction_angle_deg),
        angle_x_deg=float(angle_x),
        angle_y_deg=float(angle_y),
        angle_confidence=float(notch_confidence),
        refined=bool(refine),
        raw_points_clip=tuple(_base._point(point) for point in raw_points),
        refinement_confidences=tuple(refinement_confidences),
        refinement_mode=refine_mode if refine else "none",
        center_corner_raw_clip=_base._point(raw_points[center_index]),
        side_corner_raw_clip=_base._point(raw_points[side_index]),
        below_corner_raw_clip=_base._point(raw_points[below_index]),
        angle_mode="notch",
        robust_angle_deg=float(notch_correction_angle_deg),
        local_angle_deg=float((angle_x + angle_y) / 2.0),
        angle_pairs_clip=(),
        angle_pairs_raw_clip=(),
        angle_pair_axes=(),
        angle_pair_angles_deg=(),
        angle_pair_residuals_deg=(),
        angle_candidate_count=0,
    )


def build_die_map_from_yolo(
    wafer_image: ImageInput,
    clip_image: ImageInput,
    detections: Union[str, Path, np.ndarray, Sequence[Any]],
    *,
    clip_origin: Optional[Point] = None,
    detection_format: _base.DetectionFormat = "auto",
    normalized: Optional[bool] = None,
    confidence_threshold: float = 0.25,
    refine: bool = True,
    refine_radius: int = 18,
    refine_mode: _base.RefinementMode = "auto",
    refine_max_street_width: Optional[int] = None,
    refine_corner_patch_ratio: float = 0.22,
    refine_corner_reference_weight: float = 0.70,
    refine_noise_kernel: int = 5,
    refine_min_confidence: float = 0.15,
    pitch_size: Optional[Tuple[float, float]] = None,
    pixel_per_unit: float = 32.0,
    include_edge: bool = True,
    edge_margin: float = 1.0,
    edge_mode: str = "circle",
    notch_reference_angle_deg: float = 90.0,
    notch_max_dimension: int = 2048,
    notch_angle_samples: int = 3600,
    notch_baseline_window_deg: float = 10.0,
    notch_min_depth_px: Optional[float] = None,
    notch_min_depth_ratio: float = 0.002,
    notch_min_wide_deg: float = 2.0,
    notch_failure_mode: Literal["error", "zero"] = "error",
    return_aligned_image: bool = True,
    alignment_interpolation: int = cv2.INTER_CUBIC,
    alignment_border_value: Tuple[int, int, int] = (0, 0, 0),
) -> _base.WaferDieMap:
    """Build a die map with the wafer notch as the sole angle source."""

    wafer = _base._load_bgr(wafer_image)
    clip = _base._load_bgr(clip_image)
    full_height, full_width = wafer.shape[:2]
    clip_height, clip_width = clip.shape[:2]
    notch = detect_wafer_notch(
        wafer,
        reference_angle_deg=notch_reference_angle_deg,
        max_dimension=notch_max_dimension,
        angle_samples=notch_angle_samples,
        baseline_window_deg=notch_baseline_window_deg,
        min_notch_depth_px=notch_min_depth_px,
        min_notch_depth_ratio=notch_min_depth_ratio,
        min_wide_notch_deg=notch_min_wide_deg,
        failure_mode=notch_failure_mode,
    )
    if clip_origin is None:
        clip_origin = (
            (full_width - clip_width) / 2.0,
            (full_height - clip_height) / 2.0,
        )
    wafer_center_clip = (
        notch.wafer_center_px[0] - float(clip_origin[0]),
        notch.wafer_center_px[1] - float(clip_origin[1]),
    )
    estimate = estimate_grid_from_yolo_notch(
        clip,
        detections,
        notch_correction_angle_deg=notch.correction_angle_deg,
        notch_confidence=notch.confidence,
        reference_point_clip=wafer_center_clip,
        detection_format=detection_format,
        normalized=normalized,
        confidence_threshold=confidence_threshold,
        refine=refine,
        refine_radius=refine_radius,
        refine_mode=refine_mode,
        refine_max_street_width=refine_max_street_width,
        refine_corner_patch_ratio=refine_corner_patch_ratio,
        refine_corner_reference_weight=refine_corner_reference_weight,
        refine_noise_kernel=refine_noise_kernel,
        refine_min_confidence=refine_min_confidence,
    )
    bx, by, bw, bh = cv2.boundingRect(notch.wafer_contour_px)
    boundary = _base.WaferBoundary(
        center_px=notch.wafer_center_px,
        radius_px=notch.wafer_radius_px,
        contour_px=notch.wafer_contour_px,
        area_px=float(cv2.contourArea(notch.wafer_contour_px)),
        bbox_px=(int(bx), int(by), int(bx + bw), int(by + bh)),
        method="notch_background_circle",
    )
    origin_full = (
        float(clip_origin[0]) + estimate.center_corner_clip[0],
        float(clip_origin[1]) + estimate.center_corner_clip[1],
    )
    if pitch_size is None:
        map_pitch_x, map_pitch_y = estimate.pitch_x, estimate.pitch_y
        pitch_source = "detected"
    else:
        pitch_values = np.asarray(pitch_size, dtype=np.float64).reshape(-1)
        if (
            pitch_values.size != 2
            or not np.all(np.isfinite(pitch_values))
            or np.any(pitch_values <= 0.0)
        ):
            raise ValueError(
                "pitch_size must be a positive finite (pitch_x, pitch_y) pair."
            )
        map_pitch_x, map_pitch_y = float(pitch_values[0]), float(pitch_values[1])
        pitch_source = "manual"

    die_map = _base.generate_die_map(
        boundary,
        (full_height, full_width),
        origin_full,
        map_pitch_x,
        map_pitch_y,
        notch.correction_angle_deg,
        pixel_per_unit=pixel_per_unit,
        include_edge=include_edge,
        edge_margin=edge_margin,
        edge_mode=edge_mode,
        angle_confidence=notch.confidence,
        grid_estimate=estimate,
    )

    def pair_to_full(pair):
        return (
            (
                float(clip_origin[0]) + float(pair[0][0]),
                float(clip_origin[1]) + float(pair[0][1]),
            ),
            (
                float(clip_origin[0]) + float(pair[1][0]),
                float(clip_origin[1]) + float(pair[1][1]),
            ),
        )

    die_map.pitch_x_points_full = pair_to_full(estimate.pitch_x_points_clip)
    die_map.pitch_y_points_full = pair_to_full(estimate.pitch_y_points_clip)
    die_map.pitch_x_points_raw_full = pair_to_full(
        estimate.pitch_x_points_raw_clip
    )
    die_map.pitch_y_points_raw_full = pair_to_full(
        estimate.pitch_y_points_raw_clip
    )
    die_map.angle_pairs_full = ()
    die_map.angle_pairs_raw_full = ()
    die_map.detected_pitch_x = float(estimate.pitch_x)
    die_map.detected_pitch_y = float(estimate.pitch_y)
    die_map.pitch_source = pitch_source
    matrix = cv2.getRotationMatrix2D(
        notch.wafer_center_px, notch.correction_angle_deg, 1.0
    )
    inverse = cv2.invertAffineTransform(matrix)
    die_map.original_to_aligned_matrix = matrix
    die_map.aligned_to_original_matrix = inverse
    if return_aligned_image:
        die_map.aligned_image = cv2.warpAffine(
            wafer,
            matrix,
            (full_width, full_height),
            flags=alignment_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=alignment_border_value,
        )
    die_map.angle_align_method = "notch" if notch.found else "notch_zero_fallback"
    die_map.notch_result = notch
    die_map.notch_point_px = notch.notch_point_px
    die_map.notch_deepest_point_px = notch.notch_deepest_point_px
    die_map.notch_angle_deg = notch.notch_angle_deg
    die_map.notch_reference_angle_deg = notch.reference_angle_deg
    die_map.notch_depth_px = notch.notch_depth_px
    die_map.notch_width_px = notch.notch_width_px
    return die_map


build_die_map = build_die_map_from_yolo
