# -*- coding: utf-8 -*-
"""search_deg(기본 +-6.0) 경계 밖에서 무슨 일이 일어나는지 측정한다."""
import os
import glob
import json

import cv2
import numpy as np

import wafer_via_die_render as M

SRC = r"E:\app_dir\V10_Wafer\sample_img"
OUT = r"E:\app_dir\V10_Wafer\sample_img\result"
os.makedirs(OUT, exist_ok=True)

THETAS = [-12.0, -10.0, -8.0, -7.0, -6.0, -5.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


emit("search_deg = %.1f (기본값). 이 범위 밖의 기울기를 주면 어떻게 되는가." %
     M.DEFAULT_DIE_RENDER_SEARCH_DEG)
emit("기대: 측정각 = -theta.  경계 밖이면 경계값으로 붙는다(clamp).")
emit()

report = {}
for path in sorted(glob.glob(os.path.join(SRC, "*.png"))):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    wb = M.detect_wafer_boundary(img)
    cx, cy = (int(round(v)) for v in wb.center_px)
    r = int(round(float(wb.radius_px)))
    base = M.measure_die_render_angle(img, cx, cy, r) or 0.0

    emit("=" * 68)
    emit(name)
    emit("  %-9s %-12s %-12s %s" % ("넣은각", "기대", "측정각", "판정"))
    rows = []
    for th in THETAS:
        Mrot = cv2.getRotationMatrix2D((img.shape[1] / 2.0, img.shape[0] / 2.0), th, 1.0)
        rimg = cv2.warpAffine(img, Mrot, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        got = M.measure_die_render_angle(rimg, cx, cy, r)
        exp = base - th
        if got is None:
            emit("  %+8.2f  %+11.4f  %-12s %s" % (th, exp, "None", "신호없음"))
            rows.append({"theta": th, "expected": round(exp, 4), "measured": None})
            continue
        err = got - exp
        if abs(err) < 0.10:
            verdict = "OK"
        elif abs(got) > M.DEFAULT_DIE_RENDER_SEARCH_DEG - 0.35:
            verdict = "CLAMP (경계에 붙음, 오차 %+.2f)" % err
        else:
            verdict = "WRONG (오차 %+.2f)" % err
        emit("  %+8.2f  %+11.4f  %+11.4f  %s" % (th, exp, got, verdict))
        rows.append({"theta": th, "expected": round(exp, 4),
                     "measured": round(got, 4), "err": round(err, 4),
                     "verdict": verdict.split()[0]})
    report[name] = rows
    emit()

with open(os.path.join(OUT, "eval_limit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUT, "eval_limit.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("saved")
