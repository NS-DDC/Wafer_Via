"""Build the copy-paste single-file notch die-map distribution."""

import io
from pathlib import Path
import tokenize


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "codex" / "wafer_via.py"
NOTCH_PATH = ROOT / "codex" / "wafer_notch_angle.py"
PIPELINE_PATH = ROOT / "codex" / "wafer_via_notch.py"
OUTPUT_PATH = ROOT / "codex" / "wafer_via_notch_standalone.py"

IO_FOOTER = """
# INPUT
# wafer_image: 전체 wafer 경로 또는 uint8 BGR ndarray (H, W, 3)
# clip_image: YOLO를 실행한 중심 clip 경로 또는 BGR ndarray
# detections: YOLO 중심점/box의 numpy 배열 또는 list
# clip_origin: 중심 clip이 아닐 때 전체 이미지 기준 좌상단 (x, y)
# notch_roi_center_px: 원본 좌표 (x, y); notch_fallback_mode="rim_intrusion"(기본, 실패시만)/"none"
# OUTPUT
# 반환값: WaferDieMap
# dm.aligned_image: notch angle이 보정된 전체 이미지 (기본 결과 이미지)
# dm.grid_angle_deg: 보정 좌표계이므로 0.0
# dm.notch_result: notch 수치 결과; fallback_used / fallback_reason = 보완 사용 여부와 사유
# dm.dies / locate_die(): 보정 이미지 좌표계의 die-map 결과
# dm.notch_overlay_image / dm.notch_zoom_image: return_notch_visuals=True일 때만 생성
"""


def _strip_comments(source: str) -> str:
    """Remove Python comments while preserving strings and executable code."""

    lines = source.splitlines()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line_index, column = token.start[0] - 1, token.start[1]
        lines[line_index] = lines[line_index][:column].rstrip()

    compact = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            compact.append(line.rstrip())
        else:
            blank_count += 1
            if blank_count <= 2:
                compact.append("")
    return "\n".join(compact).rstrip() + "\n"


def _append_io_footer(source: str) -> str:
    return source.rstrip() + "\n\n" + IO_FOOTER.strip() + "\n"


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
    output = _strip_comments(output)
    output = _append_io_footer(output)
    OUTPUT_PATH.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
