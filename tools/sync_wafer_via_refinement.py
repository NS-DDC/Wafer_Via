"""Copy maintained refinement into the otherwise independent die-render module.

Run this after changing codex/wafer_via.py, then run the two existing notch
standalone builders. The marked refinement section and mode alias are copied;
the die-render pipeline and its public arguments remain its own source.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "codex" / "wafer_via.py"
TARGET = ROOT / "codex" / "wafer_via_die_render.py"
START = "# [SECTOR: 20_COLOR_INVARIANT_REFINEMENT]"
END = "# [SECTOR: 30_GRID_ESTIMATION]"


def _bounds(source: str) -> tuple[int, int]:
    if source.count(START) != 1 or source.count(END) != 1:
        raise RuntimeError("Expected one refinement and one grid-estimation section.")
    start = source.index(START)
    end = source.index(END, start)
    return start, end


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    target = TARGET.read_text(encoding="utf-8")
    source_start, source_end = _bounds(source)
    target_start, target_end = _bounds(target)
    replacement = target[:target_start] + source[source_start:source_end] + target[target_end:]
    source_alias = [line for line in source.splitlines() if line.startswith("RefinementMode = ")]
    target_alias = [line for line in target.splitlines() if line.startswith("RefinementMode = ")]
    if len(source_alias) != 1 or len(target_alias) != 1:
        raise RuntimeError("Expected one single-line RefinementMode alias in each module.")
    replacement = replacement.replace(target_alias[0], source_alias[0], 1)
    if replacement != target:
        TARGET.write_text(replacement, encoding="utf-8")
    print(f"Synchronized refinement: {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
