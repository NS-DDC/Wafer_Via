"""Wafer_Via variant with the V5 full-wafer ``die_render`` angle option.

The existing :mod:`wafer_via` module is intentionally unchanged. This module
keeps its YOLO centre/pitch/wafer-map pipeline and optionally replaces only the
final angle with the projection + FFT method adapted from
NS-DDC/Wafer_Map_Die_V5.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np

try:  # Package import
    from . import wafer_via as _base
    from .wafer_via import *  # noqa: F401,F403 - compatible public API
except ImportError:  # Same-folder copy/import
    import wafer_via as _base  # type: ignore[no-redef]
    from wafer_via import *  # type: ignore[no-redef]  # noqa: F401,F403


AngleAlignMethod = Literal["die_render", "yolo"]

DEFAULT_DIE_RENDER_SEARCH_DEG = 6.0
DEFAULT_DIE_RENDER_COARSE_STEP = 0.15
DEFAULT_DIE_RENDER_FINE_STEP = 0.02
DEFAULT_DIE_RENDER_ROI_RATIO = 0.55
DEFAULT_DIE_RENDER_MAX_DIM = 1400
DEFAULT_DIE_RENDER_FFT_MAX_DIM = 1024
DEFAULT_DIE_RENDER_AGREE_TOL_DEG = 0.40
DEFAULT_DIE_RENDER_FULL_SCAN_DEG = 44.0
DEFAULT_DIE_RENDER_MAX_ITER = 3
DEFAULT_DIE_RENDER_MIN_ANGLE_DEG = 0.01

__all__ = list(_base.__all__) + [
    "measure_wafer_angle_die_render",
    "build_die_map_from_yolo",
    "build_die_map",
]


def _projection_score(
    image_bgr: np.ndarray,
    wafer_cx: float,
    wafer_cy: float,
    wafer_r: float,
    *,
    roi_ratio: float,
    max_dim: int,
):
    """Return a rotation score based on X/Y projection periodicity."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    half = max(16, int(round(float(wafer_r) * float(roi_ratio))))
    x0, x1 = max(0, int(round(wafer_cx)) - half), min(width, int(round(wafer_cx)) + half)
    y0, y1 = max(0, int(round(wafer_cy)) - half), min(height, int(round(wafer_cy)) + half)
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None

    roi = gray[y0:y1, x0:x1]
    scale = min(1.0, float(max_dim) / float(max(roi.shape[:2])))
    scaled_width = max(8, int(round(roi.shape[1] * scale)))
    scaled_height = max(8, int(round(roi.shape[0] * scale)))
    if scale < 1.0:
        roi = cv2.resize(
            roi, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )

    local_cx = (float(wafer_cx) - x0) * scale
    local_cy = (float(wafer_cy) - y0) * scale
    yy, xx = np.ogrid[:scaled_height, :scaled_width]
    radius_scaled = half * scale
    circle_mask = (
        (xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= radius_scaled ** 2
    )

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    grid = binary.astype(np.float32)
    grid[~circle_mask] = 0.0
    if float(grid.sum()) < 1.0:
        return None

    inner_mask = (
        (xx - local_cx) ** 2
        + (yy - local_cy) ** 2
        <= (radius_scaled * 0.92) ** 2
    ).astype(np.float32)
    rotation_center = (scaled_width / 2.0, scaled_height / 2.0)

    def score(angle_deg: float) -> float:
        matrix = cv2.getRotationMatrix2D(rotation_center, float(angle_deg), 1.0)
        rotated = cv2.warpAffine(
            grid, matrix, (scaled_width, scaled_height), flags=cv2.INTER_LINEAR
        )
        rotated *= inner_mask
        return float(rotated.sum(axis=0).var() + rotated.sum(axis=1).var())

    return score


def _search_peak(
    score,
    center: float,
    search_deg: float,
    coarse_step: float,
    fine_step: float,
) -> Tuple[float, float]:
    coarse = np.arange(
        center - search_deg, center + search_deg + 1e-9, coarse_step
    )
    coarse_scores = np.asarray([score(angle) for angle in coarse])
    coarse_best = float(coarse[int(np.argmax(coarse_scores))])

    fine = np.arange(
        coarse_best - coarse_step,
        coarse_best + coarse_step + 1e-9,
        fine_step,
    )
    fine_scores = np.asarray([score(angle) for angle in fine])
    best_index = int(np.argmax(fine_scores))
    best_angle = float(fine[best_index])
    best_score = float(fine_scores[best_index])
    if 0 < best_index < len(fine) - 1:
        before = float(fine_scores[best_index - 1])
        current = float(fine_scores[best_index])
        after = float(fine_scores[best_index + 1])
        denominator = before - 2.0 * current + after
        if abs(denominator) > 1e-9:
            best_angle += 0.5 * (before - after) / denominator * fine_step
    return best_angle, best_score


def _measure_fft_angle(
    image_bgr: np.ndarray,
    wafer_cx: float,
    wafer_cy: float,
    wafer_r: float,
    *,
    roi_ratio: float,
    max_dim: int,
) -> Optional[float]:
    """Estimate the grid angle independently from the 2-D FFT spectrum."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    half = max(16, int(round(float(wafer_r) * float(roi_ratio))))
    x0, x1 = max(0, int(round(wafer_cx)) - half), min(width, int(round(wafer_cx)) + half)
    y0, y1 = max(0, int(round(wafer_cy)) - half), min(height, int(round(wafer_cy)) + half)
    roi = gray[y0:y1, x0:x1]
    if min(roi.shape[:2]) < 16:
        return None
    scale = min(1.0, float(max_dim) / float(max(roi.shape[:2])))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    size = min(roi.shape[:2])
    offset_y = (roi.shape[0] - size) // 2
    offset_x = (roi.shape[1] - size) // 2
    square = roi[offset_y:offset_y + size, offset_x:offset_x + size].astype(np.float32)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    spectrum = np.fft.fftshift(
        np.fft.fft2((square - float(square.mean())) * window)
    )
    magnitude = spectrum.real ** 2 + spectrum.imag ** 2

    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    dx = (xx - center).astype(np.float64)
    dy = (yy - center).astype(np.float64)
    radius = np.sqrt(dx * dx + dy * dy)
    radius_min = max(4.0, size * 0.012)
    radius_max = size * 0.45
    band = (radius >= radius_min) & (radius <= radius_max)
    if int(band.sum()) < 16:
        return None

    radial_energy = np.bincount(
        radius.astype(np.int32)[band].ravel(),
        weights=magnitude[band].ravel(),
        minlength=int(radius_max) + 2,
    )
    if radial_energy.size == 0 or float(radial_energy.max()) <= 0.0:
        return None
    peak_radius = max(float(np.argmax(radial_energy)), radius_min + 1.0)
    annulus = (
        (np.abs(radius - peak_radius) <= max(2.0, peak_radius * 0.45))
        & (radius >= radius_min)
    )
    if int(annulus.sum()) < 16:
        annulus = band

    phase = np.arctan2(dy[annulus], dx[annulus])
    weights = magnitude[annulus].astype(np.float64)
    if float(weights.sum()) <= 0.0:
        return None
    vector = np.sum(weights * np.exp(1j * 4.0 * phase))
    tilt = float(np.degrees(np.angle(vector)) / 4.0)
    return float((tilt + 45.0) % 90.0 - 45.0)


def measure_wafer_angle_die_render(
    image_bgr: np.ndarray,
    wafer_cx: float,
    wafer_cy: float,
    wafer_r: float,
    *,
    roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
    max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM,
    fft_max_dim: int = DEFAULT_DIE_RENDER_FFT_MAX_DIM,
    search_deg: float = DEFAULT_DIE_RENDER_SEARCH_DEG,
    coarse_step: float = DEFAULT_DIE_RENDER_COARSE_STEP,
    fine_step: float = DEFAULT_DIE_RENDER_FINE_STEP,
    agree_tol_deg: float = DEFAULT_DIE_RENDER_AGREE_TOL_DEG,
    full_scan_deg: float = DEFAULT_DIE_RENDER_FULL_SCAN_DEG,
) -> Dict[str, Any]:
    """Measure the full-wafer grid angle using V5 projection + FFT cues."""

    score = _projection_score(
        image_bgr,
        wafer_cx,
        wafer_cy,
        wafer_r,
        roi_ratio=roi_ratio,
        max_dim=max_dim,
    )
    fft_angle = _measure_fft_angle(
        image_bgr,
        wafer_cx,
        wafer_cy,
        wafer_r,
        roi_ratio=roi_ratio,
        max_dim=fft_max_dim,
    )
    if score is None:
        if fft_angle is None:
            return {
                "angle": 0.0,
                "confidence": 0.0,
                "agree": False,
                "projection": None,
                "fft": None,
                "candidates": [],
            }
        return {
            "angle": float(fft_angle),
            "confidence": 0.45,
            "agree": False,
            "projection": None,
            "fft": float(fft_angle),
            "candidates": [float(fft_angle)],
        }

    projection_angle, projection_score = _search_peak(
        score, 0.0, search_deg, coarse_step, fine_step
    )
    if (
        fft_angle is not None
        and abs(projection_angle - fft_angle) <= agree_tol_deg
    ):
        return {
            "angle": float(projection_angle),
            "confidence": 0.97,
            "agree": True,
            "projection": float(projection_angle),
            "fft": float(fft_angle),
            "candidates": [float(projection_angle), float(fft_angle)],
        }

    candidates = [(projection_angle, projection_score)]
    if fft_angle is not None:
        candidates.append(
            _search_peak(
                score,
                float(fft_angle),
                max(coarse_step * 3.0, 1.0),
                coarse_step,
                fine_step,
            )
        )
    candidates.append(
        _search_peak(
            score,
            0.0,
            full_scan_deg,
            max(coarse_step * 2.0, 0.3),
            fine_step,
        )
    )
    best_angle, _ = max(candidates, key=lambda candidate: candidate[1])
    agree = bool(
        fft_angle is not None and abs(best_angle - fft_angle) <= agree_tol_deg
    )
    return {
        "angle": float(best_angle),
        "confidence": 0.90 if agree else 0.60,
        "agree": agree,
        "projection": float(projection_angle),
        "fft": None if fft_angle is None else float(fft_angle),
        "candidates": [float(candidate[0]) for candidate in candidates],
    }


def _measure_iterative(
    image_bgr: np.ndarray,
    wafer_center: Tuple[float, float],
    wafer_radius: float,
    *,
    max_iter: int,
    min_angle_deg: float,
    measure_kwargs: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    total_angle = 0.0
    first_info: Optional[Dict[str, Any]] = None
    deltas = []
    for _ in range(max(1, int(max_iter))):
        if abs(total_angle) > 1e-12:
            aligned, _, _ = _base.align_wafer_image(
                image_bgr, wafer_center, total_angle
            )
        else:
            aligned = image_bgr
        info = measure_wafer_angle_die_render(
            aligned,
            wafer_center[0],
            wafer_center[1],
            wafer_radius,
            **measure_kwargs,
        )
        if first_info is None:
            first_info = dict(info)
        delta = float(info.get("angle") or 0.0)
        deltas.append(delta)
        if float(info.get("confidence") or 0.0) <= 0.0 or abs(delta) < min_angle_deg:
            break
        total_angle += delta
    result = first_info or {
        "angle": 0.0,
        "confidence": 0.0,
        "agree": False,
    }
    result["total_angle"] = float(total_angle)
    result["iteration_deltas"] = tuple(float(value) for value in deltas)
    result["final_residual"] = float(deltas[-1]) if deltas else 0.0
    return float(total_angle), result


def _copy_geometry_diagnostics(source: Any, target: Any) -> None:
    for name in (
        "pitch_x_points_full",
        "pitch_y_points_full",
        "pitch_x_points_raw_full",
        "pitch_y_points_raw_full",
        "detected_pitch_x",
        "detected_pitch_y",
        "pitch_source",
    ):
        setattr(target, name, getattr(source, name))


def build_die_map_from_yolo(
    wafer_image: Any,
    clip_image: Any,
    detections: Any,
    *,
    angle_align_method: AngleAlignMethod = "die_render",
    die_render_roi_ratio: float = DEFAULT_DIE_RENDER_ROI_RATIO,
    die_render_max_dim: int = DEFAULT_DIE_RENDER_MAX_DIM,
    die_render_fft_max_dim: int = DEFAULT_DIE_RENDER_FFT_MAX_DIM,
    die_render_search_deg: float = DEFAULT_DIE_RENDER_SEARCH_DEG,
    die_render_coarse_step: float = DEFAULT_DIE_RENDER_COARSE_STEP,
    die_render_fine_step: float = DEFAULT_DIE_RENDER_FINE_STEP,
    die_render_agree_tol_deg: float = DEFAULT_DIE_RENDER_AGREE_TOL_DEG,
    die_render_full_scan_deg: float = DEFAULT_DIE_RENDER_FULL_SCAN_DEG,
    die_render_max_iter: int = DEFAULT_DIE_RENDER_MAX_ITER,
    die_render_min_angle_deg: float = DEFAULT_DIE_RENDER_MIN_ANGLE_DEG,
    die_render_fallback_to_yolo: bool = True,
    **kwargs: Any,
) -> _base.WaferDieMap:
    """Build a YOLO die map with optional V5 full-wafer die-render angle.

    ``angle_align_method="die_render"`` replaces only the final map/aligned
    image angle. Centre selection, pitch, wafer boundary, clipping, indexing,
    and all other behavior remain the current :mod:`wafer_via` implementation.
    Use ``"yolo"`` to retain the centre-clip angle unchanged.
    """

    method = str(angle_align_method).strip().lower().replace("-", "_")
    if method in ("render", "die", "grid_render"):
        method = "die_render"
    if method in ("current", "clip", "robust"):
        method = "yolo"
    if method not in ("die_render", "yolo"):
        raise ValueError("angle_align_method must be 'die_render' or 'yolo'.")

    base_kwargs = dict(kwargs)
    return_aligned_image = bool(base_kwargs.pop("return_aligned_image", True))
    interpolation = int(base_kwargs.pop("alignment_interpolation", cv2.INTER_CUBIC))
    border_value = tuple(base_kwargs.pop("alignment_border_value", (0, 0, 0)))
    base_dm = _base.build_die_map_from_yolo(
        wafer_image,
        clip_image,
        detections,
        return_aligned_image=False,
        **base_kwargs,
    )
    wafer = _base._load_bgr(wafer_image)
    yolo_angle = float(base_dm.grid_angle_deg)
    info: Dict[str, Any] = {
        "angle": yolo_angle,
        "total_angle": yolo_angle,
        "confidence": float(base_dm.angle_confidence),
        "agree": True,
        "source": "yolo",
    }

    final_angle = yolo_angle
    if method == "die_render":
        measure_kwargs = {
            "roi_ratio": float(die_render_roi_ratio),
            "max_dim": int(die_render_max_dim),
            "fft_max_dim": int(die_render_fft_max_dim),
            "search_deg": float(die_render_search_deg),
            "coarse_step": float(die_render_coarse_step),
            "fine_step": float(die_render_fine_step),
            "agree_tol_deg": float(die_render_agree_tol_deg),
            "full_scan_deg": float(die_render_full_scan_deg),
        }
        measured_angle, info = _measure_iterative(
            wafer,
            (float(base_dm.wafer_cx), float(base_dm.wafer_cy)),
            float(base_dm.wafer_r),
            max_iter=die_render_max_iter,
            min_angle_deg=float(die_render_min_angle_deg),
            measure_kwargs=measure_kwargs,
        )
        if float(info.get("confidence") or 0.0) > 0.0:
            final_angle = float(measured_angle)
            info["source"] = "die_render"
        elif die_render_fallback_to_yolo:
            final_angle = yolo_angle
            info["source"] = "yolo_fallback"
        else:
            final_angle = float(measured_angle)
            info["source"] = "die_render_no_signal"

    final_source = str(info.get("source") or method)
    uses_pixel_angle = final_source.startswith("die_render")

    grid_estimate = base_dm.grid_estimate
    if grid_estimate is not None:
        replacement = {
            "angle_deg": float(final_angle),
            "angle_confidence": float(info.get("confidence") or 0.0),
            "angle_mode": final_source,
        }
        if uses_pixel_angle:
            # die_render consumes full-wafer pixels, not discrete YOLO pairs.
            # Keep pair fields empty so the regular clip overlay cannot imply
            # that the centre-clip vectors produced the final angle.
            replacement.update({
                "angle_pairs_clip": (),
                "angle_pairs_raw_clip": (),
                "angle_pair_axes": (),
                "angle_pair_angles_deg": (),
                "angle_pair_residuals_deg": (),
                "angle_candidate_count": 0,
            })
        grid_estimate = replace(grid_estimate, **replacement)
    result = _base.generate_die_map(
        base_dm.wafer_boundary,
        base_dm.image_shape,
        (base_dm.x0, base_dm.y0),
        base_dm.pitch_x,
        base_dm.pitch_y,
        final_angle,
        pixel_per_unit=base_dm.pixel_per_unit,
        include_edge=bool(base_kwargs.get("include_edge", True)),
        edge_margin=float(base_kwargs.get("edge_margin", 1.0)),
        edge_mode=str(base_kwargs.get("edge_mode", "circle")),
        angle_confidence=float(info.get("confidence") or 0.0),
        grid_estimate=grid_estimate,
    )
    _copy_geometry_diagnostics(base_dm, result)
    if uses_pixel_angle:
        result.angle_pairs_full = ()
        result.angle_pairs_raw_full = ()
    else:
        result.angle_pairs_full = base_dm.angle_pairs_full
        result.angle_pairs_raw_full = base_dm.angle_pairs_raw_full

    matrix, inverse = _base._alignment_matrices(
        (result.wafer_cx, result.wafer_cy), result.grid_angle_deg
    )
    result.original_to_aligned_matrix = matrix
    result.aligned_to_original_matrix = inverse
    if return_aligned_image:
        result.aligned_image, _, _ = _base.align_wafer_image(
            wafer,
            (result.wafer_cx, result.wafer_cy),
            result.grid_angle_deg,
            interpolation=interpolation,
            border_value=border_value,
        )

    result.angle_align_method = final_source
    result.yolo_angle_deg = yolo_angle
    result.yolo_angle_pairs_full = base_dm.angle_pairs_full
    result.yolo_angle_pairs_raw_full = base_dm.angle_pairs_raw_full
    result.die_render_info = dict(info)
    result.angle_agree = bool(info.get("agree", False))
    return result


build_die_map = build_die_map_from_yolo
