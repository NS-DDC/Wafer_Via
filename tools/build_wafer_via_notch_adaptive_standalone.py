"""Build a full copy-paste pipeline whose only change is notch angle detection."""

from pathlib import Path

from build_wafer_via_notch_standalone import _append_io_footer, _strip_comments


ROOT = Path(__file__).resolve().parents[1]
BASE_STANDALONE = ROOT / "codex" / "wafer_via_notch_standalone.py"
ADAPTIVE_SOURCE = ROOT / "codex" / "wafer_notch_v5_adaptive_background.py"
OUTPUT_PATH = ROOT / "codex" / "wafer_via_notch_adaptive_standalone.py"


ADAPTER = r'''

# [SECTOR: 86_ADAPTIVE_BACKGROUND_ANGLE_OVERRIDE] ---------------------------
# Everything above is the existing full copy-paste notch/DM pipeline. Only the
# global detect_wafer_notch function is replaced below. Python resolves that
# global at call time, so build_die_map_from_yolo, locate_die, pitch, overlays,
# coordinate transforms, and return fields remain unchanged.

draw_aligned_wafer_notch_guide_adaptive = (
    _adaptive_draw_aligned_wafer_notch_guide
)
__all__.extend([
    "AdaptiveBackgroundNotchGuideResult",
    "draw_aligned_wafer_notch_guide_adaptive",
])


_geometry_detect_wafer_notch = detect_wafer_notch


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
    background_palette_size: int = 3,
    background_distance_threshold_lab: Optional[float] = None,
    background_noise_margin_lab: float = 6.0,
    min_background_distance_lab: float = 8.0,
    border_band_px: int = 16,
) -> NotchAngleResult:
    """Drop-in notch detector using border-adaptive V5 geometry only.

    The signature keeps all established arguments. Compatibility-only
    ``baseline_window_deg``, ``radial_inner_ratio``, and
    ``min_wide_notch_deg`` are accepted but do not alter the V5 radial method.
    """

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    _ = baseline_window_deg, radial_inner_ratio, min_wide_notch_deg

    # A manual ROI is an explicit request for the local semicircle detector
    # embedded in the base standalone. With no ROI this derived file keeps its
    # historical adaptive-background V5 angle override.
    if notch_roi_center_px is not None:
        return _geometry_detect_wafer_notch(
            image,
            reference_angle_deg=reference_angle_deg,
            max_dimension=max_dimension,
            angle_samples=angle_samples,
            baseline_window_deg=baseline_window_deg,
            radial_inner_ratio=radial_inner_ratio,
            min_notch_depth_px=min_notch_depth_px,
            min_notch_depth_ratio=min_notch_depth_ratio,
            min_wide_notch_deg=min_wide_notch_deg,
            search_center_angle_deg=search_center_angle_deg,
            search_half_width_deg=search_half_width_deg,
            wafer_center_hint_px=wafer_center_hint_px,
            wafer_radius_hint_px=wafer_radius_hint_px,
            notch_roi_center_px=notch_roi_center_px,
            notch_roi_half_size_px=notch_roi_half_size_px,
            notch_semicircle_radius_range_px=notch_semicircle_radius_range_px,
            notch_semicircle_min_score=notch_semicircle_min_score,
            notch_use_roi_background=notch_use_roi_background,
            notch_background_palette_size=notch_background_palette_size,
            notch_background_outer_band_fraction=notch_background_outer_band_fraction,
            notch_background_distance_threshold_lab=notch_background_distance_threshold_lab,
            notch_background_noise_margin_lab=notch_background_noise_margin_lab,
            notch_background_morph_px=notch_background_morph_px,
            failure_mode=mode,
        )

    guide = _adaptive_draw_aligned_wafer_notch_guide(
        image,
        reference_angle_deg=reference_angle_deg,
        search_center_angle_deg=search_center_angle_deg,
        search_half_width_deg=search_half_width_deg,
        max_analysis_dimension=max_dimension,
        border_band_px=border_band_px,
        background_palette_size=background_palette_size,
        background_distance_threshold_lab=background_distance_threshold_lab,
        background_noise_margin_lab=background_noise_margin_lab,
        min_background_distance_lab=min_background_distance_lab,
        angle_samples=angle_samples,
        min_notch_depth_px=min_notch_depth_px,
        min_notch_depth_ratio=min_notch_depth_ratio,
        wafer_center_hint_px=wafer_center_hint_px,
        wafer_radius_hint_px=wafer_radius_hint_px,
        failure_mode=mode,
        draw_text=False,
    )

    center = guide.wafer_center_px
    radius = float(guide.wafer_radius_px)
    if guide.found:
        notch_angle = float(guide.notch_angle_deg)
        notch_point = guide.notch_point_px
        deepest_point = (
            guide.notch_deepest_point_px
            or guide.notch_center_px
            or notch_point
        )
        correction = _normalise_angle(
            notch_angle - float(reference_angle_deg)
        )
    else:
        notch_angle = float(reference_angle_deg) % 360.0
        notch_angle_rad = math.radians(notch_angle)
        notch_point = (
            float(center[0] + radius * math.cos(notch_angle_rad)),
            float(center[1] + radius * math.sin(notch_angle_rad)),
        )
        deepest_point = notch_point
        correction = 0.0

    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(guide.notch_width_deg) / 2.0)
    )
    threshold = max(0.1, float(guide.effective_depth_threshold_px))
    depth_ratio = float(guide.notch_depth_px) / threshold
    confidence = (
        float(np.clip((depth_ratio - 0.70) / 1.30, 0.0, 1.0))
        if guide.found else 0.0
    )
    contour_points = guide.wafer_contour_px.reshape(-1, 2).astype(np.float64)
    contour_radii = np.hypot(
        contour_points[:, 0] - float(center[0]),
        contour_points[:, 1] - float(center[1]),
    )
    circle_residual = float(np.median(np.abs(contour_radii - radius)))
    auto_min_depth = (
        float(min_notch_depth_px)
        if min_notch_depth_px is not None
        else max(1.25 / max(guide.analysis_scale, 1e-9), radius * float(min_notch_depth_ratio))
    )
    radial_noise = max(
        0.0, float(guide.effective_depth_threshold_px) - auto_min_depth
    )

    return NotchAngleResult(
        found=bool(guide.found),
        wafer_center_px=center,
        wafer_radius_px=radius,
        notch_point_px=notch_point,
        notch_deepest_point_px=deepest_point,
        notch_angle_deg=notch_angle,
        reference_angle_deg=float(reference_angle_deg),
        correction_angle_deg=float(correction),
        notch_depth_px=float(guide.notch_depth_px),
        notch_width_deg=float(guide.notch_width_deg),
        notch_width_px=notch_width_px,
        confidence=confidence,
        radial_noise_px=radial_noise,
        candidate_arc_px=guide.candidate_arc_px,
        wafer_contour_px=guide.wafer_contour_px,
        segmentation_threshold=float(guide.segmentation_threshold_lab),
        scale=float(guide.analysis_scale),
        failure_mode=mode,
        detection_method="v5_border_adaptive_angle_only",
        search_center_angle_deg=float(search_center_angle_deg) % 360.0,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=1.0 if guide.found else 0.0,
        circle_fit_residual_px=circle_residual,
    )
'''


def main() -> None:
    base = BASE_STANDALONE.read_text(encoding="utf-8").rstrip()
    adaptive = ADAPTIVE_SOURCE.read_text(encoding="utf-8")
    start = adaptive.index(
        "@dataclass(frozen=True)\nclass AdaptiveBackgroundNotchGuideResult"
    )
    end = adaptive.index("\ndef _main()")
    adaptive_tail = adaptive[start:end].rstrip().replace(
        "def draw_aligned_wafer_notch_guide(\n",
        "def _adaptive_draw_aligned_wafer_notch_guide(\n",
        1,
    )
    header = (
        '"""Full copy-paste Wafer_Via pipeline with adaptive-background '
        'V5 notch angle.\n\n'
        'The die-map, YOLO, pitch, locate_die, coordinate, image-alignment, and\n'
        'return APIs are inherited unchanged from wafer_via_notch_standalone.\n'
        'Only detect_wafer_notch is overridden at the end of this file.\n'
        '"""\n\n'
    )
    # Keep the base future import legal by replacing only its opening docstring.
    first_end = base.index('"""', 3) + 3
    base_without_docstring = base[first_end:].lstrip("\r\n")
    output = (
        header
        + base_without_docstring
        + "\n\n\n"
        + adaptive_tail
        + ADAPTER
        + "\n"
    )
    output = _strip_comments(output)
    output = _append_io_footer(output)
    OUTPUT_PATH.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
