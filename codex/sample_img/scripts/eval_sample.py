# -*- coding: utf-8 -*-
"""sample_img 3장으로 measure_die_render_angle(V5 bare) 을 평가한다.

정답 각도가 없으므로 왕복(round-trip) 평가를 한다:
  알려진 각도 theta 로 이미지를 회전 -> 측정값이 theta 를 되찾는가.
오차 = measured - theta.
"""
import os
import glob
import json

import cv2
import numpy as np

import wafer_via_die_render as M

SRC = r"E:\app_dir\V10_Wafer\sample_img"
OUT = r"E:\app_dir\V10_Wafer\sample_img\result"
os.makedirs(OUT, exist_ok=True)

THETAS = [-4.0, -3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]


def rotate(img, deg):
    h, w = img.shape[:2]
    Mrot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, Mrot, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def pitch_of(gray, cx, cy, r):
    """중앙 ROI 에서 열 투영의 자기상관 주기 = pitch(px). 참고용."""
    half = int(r * 0.5)
    roi = gray[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
    prof = roi.astype(np.float64).mean(axis=0)
    prof -= prof.mean()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    lo, hi = 8, min(120, len(ac) - 1)
    if hi <= lo:
        return float("nan")
    return float(lo + int(np.argmax(ac[lo:hi])))


report = {}
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


for path in sorted(glob.glob(os.path.join(SRC, "*.png"))):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wb = M.detect_wafer_boundary(img)
    cx, cy = (int(round(v)) for v in wb.center_px)
    r = int(round(float(wb.radius_px)))

    p = pitch_of(gray, cx, cy, r)

    emit("=" * 74)
    emit("%s   %dx%d" % (name, img.shape[1], img.shape[0]))
    emit("  wafer  center=(%d,%d)  r=%d   (method=%s)" % (cx, cy, r, wb.method))
    emit("  pitch(auto-corr) = %.0f px    ROI(roi_ratio .55) = %d px  -> %.1f 주기"
         % (p, int(r * 0.55 * 2), (r * 0.55 * 2) / p if p == p else float("nan")))

    base = M.measure_die_render_angle(img, cx, cy, r)
    emit("  원본 측정각 = %s" % ("None" if base is None else "%+.4f deg" % base))
    emit("")
    emit("  [왕복] theta 로 회전 -> 되찾는가")
    emit("    %-9s %-12s %-10s" % ("넣은각", "측정각", "오차"))

    rows = []
    errs = []
    for th in THETAS:
        rimg = rotate(img, th)
        # 회전해도 웨이퍼 중심/반경은 그대로(중심 기준 회전).
        got = M.measure_die_render_angle(rimg, cx, cy, r)
        if got is None:
            emit("    %+8.2f  %-12s %-10s" % (th, "None", "-"))
            rows.append({"theta": th, "measured": None, "err": None})
            continue
        # 원본이 이미 base 만큼 기울어 있으므로 기대값은 base - th (회전 부호 반대).
        exp = (base if base is not None else 0.0) - th
        err = got - exp
        errs.append(abs(err))
        flag = "" if abs(err) < 0.10 else ("  <-- 벗어남" if abs(err) < 0.5 else "  <== 실패")
        emit("    %+8.2f  %+11.4f  %+9.4f%s" % (th, got, err, flag))
        rows.append({"theta": th, "measured": round(got, 4),
                     "expected": round(exp, 4), "err": round(err, 4)})

    if errs:
        emit("")
        emit("  오차 |max| = %.4f deg,  중앙값 = %.4f deg,  RMS = %.4f deg"
             % (max(errs), float(np.median(errs)),
                float(np.sqrt(np.mean(np.square(errs))))))

    # 정렬 결과 이미지 저장
    if base is not None:
        aligned = rotate(img, base)
        cv2.imwrite(os.path.join(OUT, "%s_aligned.png" % name), aligned)

    report[name] = {
        "size": [img.shape[1], img.shape[0]],
        "wafer": {"cx": cx, "cy": cy, "r": r, "method": wb.method},
        "pitch_px": p,
        "base_angle_deg": None if base is None else round(base, 4),
        "roundtrip": rows,
        "abs_err_max": None if not errs else round(max(errs), 4),
        "abs_err_rms": None if not errs else round(float(np.sqrt(np.mean(np.square(errs)))), 4),
    }
    emit("")

with open(os.path.join(OUT, "eval_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT, "eval_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("saved ->", OUT)
