"""Create reproducible synthetic evidence; no user/camera images are modified."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_wafer_notch_rim_geometry import irregular_fixture, irregular_options
from codex.wafer_notch_angle import detect_wafer_notch, make_notch_overlay


def main():
    rows = []
    for shape, rotation in (("jagged", 0.0), ("step", 13.0), ("double", 0.0)):
        image, roi = irregular_fixture(shape, rotation)
        before = detect_wafer_notch(image, **irregular_options(roi, notch_fallback_mode="none"))
        after = detect_wafer_notch(image, **irregular_options(roi))
        annotated = make_notch_overlay(image, after, thickness=1)
        cx, cy = (int(round(value)) for value in roi)
        crop_box = (slice(max(0, cy - 100), min(900, cy + 55)), slice(cx - 140, cx + 140))
        panels = []
        labels = (
            (image, "SYNTHETIC " + shape + " | primary found=" + str(before.found)),
            (annotated, "found=" + str(after.found) + " fallback=" + str(after.fallback_used)),
        )
        for source, label in labels:
            crop = cv2.resize(source[crop_box], (672, 372), interpolation=cv2.INTER_NEAREST)
            panel = np.full((442, 672, 3), 30, np.uint8)
            panel[70:] = crop
            cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 1, cv2.LINE_AA)
            text = (after.fallback_reason or "primary_success_unchanged") if source is annotated else "rotation=" + str(rotation) + " deg"
            cv2.putText(panel, text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .56, (190, 215, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.hstack(panels))
        print(shape, dict(primary_found=before.found, found=after.found,
                          fallback_used=after.fallback_used, reason=after.fallback_reason,
                          notch_angle_deg=after.notch_angle_deg))
    destination = ROOT / "codex" / "sample_img" / "notch_fallback_preview.png"
    if not cv2.imwrite(str(destination), np.vstack(rows)):
        raise RuntimeError("Cannot write fallback preview")
    print(destination)


if __name__ == "__main__":
    main()
