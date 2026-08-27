"""Build the copy-paste single-file notch die-map distribution."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "codex" / "wafer_via.py"
NOTCH_PATH = ROOT / "codex" / "wafer_notch_angle.py"
PIPELINE_PATH = ROOT / "codex" / "wafer_via_notch.py"
OUTPUT_PATH = ROOT / "codex" / "wafer_via_notch_standalone.py"


def main() -> None:
    base = BASE_PATH.read_text(encoding="utf-8")
    notch = NOTCH_PATH.read_text(encoding="utf-8")
    pipeline = PIPELINE_PATH.read_text(encoding="utf-8")

    base = base.replace(
        '"""YOLO cross-points to a wafer die map.',
        '"""Standalone notch-aligned YOLO wafer die-map pipeline.',
        1,
    )
    base = base.replace(
        "def build_die_map_from_yolo(\n",
        "def _legacy_build_die_map_from_yolo(\n",
        1,
    )
    base = base.replace(
        "build_die_map = build_die_map_from_yolo\n",
        "build_die_map = _legacy_build_die_map_from_yolo\n",
        1,
    )

    notch_tail = notch[notch.index("@dataclass(frozen=True)\nclass NotchAngleResult"):]
    pipeline_tail = pipeline[
        pipeline.index("def _affine_point("):
    ].replace("_base.", "")
    exports = """

# [SECTOR: 85_NOTCH_ANGLE] ---------------------------------------------------
# The notch detector and notch-only DM builder are embedded below. The legacy
# YOLO angle helpers above are never called by the exported builder.
__all__.extend([
    "AlignedNotchGuideResult",
    "NotchAngleResult",
    "detect_wafer_notch",
    "align_wafer_by_notch",
    "draw_aligned_wafer_notch_guide",
    "make_notch_overlay",
    "make_notch_zoom",
    "estimate_grid_from_yolo_notch",
])

"""
    output = base.rstrip() + exports + notch_tail.rstrip() + "\n\n\n" + pipeline_tail
    OUTPUT_PATH.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
