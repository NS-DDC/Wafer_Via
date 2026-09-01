"""Build the copy-paste single-file notch die-map distribution."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "codex" / "wafer_via.py"
NOTCH_PATH = ROOT / "codex" / "wafer_notch_angle.py"
PIPELINE_PATH = ROOT / "codex" / "wafer_via_notch.py"
OUTPUT_PATH = ROOT / "codex" / "wafer_via_notch_standalone.py"


def _remove_between(source: str, start: str, end: str) -> str:
    """Remove one generated-only legacy region while keeping ``end``."""

    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[:start_index] + source[end_index:]


def _make_notch_only_base(source: str) -> str:
    """Keep shared geometry/YOLO-pitch helpers, drop every YOLO angle path."""

    source = source.replace('AngleMode = Literal["robust", "local"]\n', "", 1)
    for export in (
        '    "estimate_grid_from_yolo",\n',
        '    "build_die_map_from_yolo",\n',
        '    "build_die_map",\n',
    ):
        source = source.replace(export, "", 1)

    # These fields only existed to explain the removed robust/local estimator.
    source = _remove_between(
        source,
        '    angle_mode: str = "local"\n',
        "\n    @property\n    def pitch_x_points_clip",
    )
    source = _remove_between(
        source,
        '            "angle_mode": self.angle_mode,\n',
        "        }\n\n\n@dataclass(frozen=True)\nclass WaferBoundary",
    )
    source = source.replace(
        "    angle_pairs_full: Tuple[PointPair, ...] = ()\n"
        "    angle_pairs_raw_full: Tuple[PointPair, ...] = ()\n",
        "",
        1,
    )

    # Keep _fold_grid_angle and _select_axis_neighbour because notch-guided
    # pitch selection uses them. Remove only robust/local angle estimation.
    source = _remove_between(
        source,
        "def _weighted_median(",
        "def _select_axis_neighbour(",
    )
    source = _remove_between(
        source,
        "def estimate_grid_from_yolo(",
        "# [SECTOR: 40_WAFER_BOUNDARY]",
    )

    # Clip overlay remains useful for pitch debugging, but no longer draws or
    # labels the deleted YOLO angle-pair estimator.
    source = _remove_between(
        source,
        "    if estimate.angle_pairs_clip:\n",
        "    if estimate.raw_points_clip:\n",
    )
    old_label = (
        '        f"Px={estimate.pitch_x:.2f} Py={estimate.pitch_y:.2f} "\n'
        '        f"A={estimate.angle_deg:.3f}deg({estimate.angle_mode}) "\n'
        '        f"N={len(estimate.angle_pairs_clip)}/{estimate.angle_candidate_count} "\n'
        '        f"R={estimate.refinement_mode}"\n'
    )
    new_label = (
        '        f"Px={estimate.pitch_x:.2f} Py={estimate.pitch_y:.2f} "\n'
        '        f"A={estimate.angle_deg:.3f}deg(notch) "\n'
        '        f"R={estimate.refinement_mode}"\n'
    )
    if old_label not in source:
        raise RuntimeError("The clip-overlay legacy angle label changed upstream.")
    source = source.replace(old_label, new_label, 1)

    # The old end-to-end builder and its usage examples estimate angle from
    # YOLO. The notch-only builder is appended from wafer_via_notch.py below.
    source = source[:source.index("# [SECTOR: 80_PIPELINE]")].rstrip() + "\n"
    return source


def main() -> None:
    base = BASE_PATH.read_text(encoding="utf-8")
    notch = NOTCH_PATH.read_text(encoding="utf-8")
    pipeline = PIPELINE_PATH.read_text(encoding="utf-8")

    base = base.replace(
        '"""YOLO cross-points to a wafer die map.',
        '"""Standalone notch-aligned YOLO wafer die-map pipeline.',
        1,
    )
    base = _make_notch_only_base(base)

    notch_tail = notch[notch.index("@dataclass(frozen=True)\nclass NotchAngleResult"):]
    pipeline_tail = pipeline[
        pipeline.index("def _affine_point("):
    ].replace("_base.", "").replace(
        '        angle_mode="notch",\n',
        "",
        1,
    )
    exports = """

# [SECTOR: 85_NOTCH_ANGLE] ---------------------------------------------------
# Angle is detected only from the notch. YOLO points are retained solely for
# centre-corner and X/Y pitch selection.
__all__.extend([
    "AlignedNotchGuideResult",
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "draw_aligned_wafer_notch_guide",
    "make_notch_overlay",
    "make_notch_zoom",
    "estimate_grid_from_yolo_notch",
    "build_die_map_from_yolo",
    "build_die_map",
])

"""
    output = base.rstrip() + exports + notch_tail.rstrip() + "\n\n\n" + pipeline_tail
    OUTPUT_PATH.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
