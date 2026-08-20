# -*- coding: utf-8 -*-
"""
test_grid_claude.py
====================================================================
센터 코너 / pitch_x / pitch_y / 회전각이 **정답과 얼마나 맞는지** 잰다.

측정 항목
--------------------------------------------------------------------
  pitch_x, pitch_y : 정답 pitch 대비 오차 (px)
  angle            : 정답 회전각 대비 오차 (deg)
  center           : "센터 코너"가 실제 격자점 위에 있나 (px)
  index            : 좌표 -> 격자 인덱스 왕복이 맞나

왜 각도 오차가 중요한가
--------------------------------------------------------------------
이 각도로 10000x10000 웨이퍼에 격자를 외삽한다. 중심에서 5000 px 떨어진
끝에서 밀리는 양은 5000 * tan(오차) 다.
    0.1 deg -> 8.7 px      0.01 deg -> 0.87 px      0.001 deg -> 0.09 px
die 하나가 100~200 px 이므로, 0.1 deg 면 끝에서 한 칸 가까이 밀린다.
그래서 "각도가 맞다"는 0.01 deg 수준을 봐야 한다.

또 하나: 서브픽셀 보정을 **안 하면** 얼마나 나쁜지도 같이 잰다.
보정이 실제로 각도에 기여하는지 확인하려면 그 비교가 있어야 한다.

실행
--------------------------------------------------------------------
    python test_grid_claude.py
    python test_grid_claude.py --n 60
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from synth_clip_claude import PALETTE_PRESETS, make_clip, simulate_yolo
from via_grid_claude import analyze_clip, fit_grid


def _wrap90(a: float) -> float:
    """각도 차를 -45~45 로 접는다 (정사각 격자는 90도 대칭)."""
    return ((a + 45.0) % 90.0) - 45.0


def run(n: int, verbose: bool) -> dict:
    e_px, e_py, e_ang, e_ctr = [], [], [], []
    e_ang_raw = []                      # 보정 없이 raw YOLO 로만 했을 때
    e_ang_fb = []                       # 가장자리 점까지 되살려 쓴 경우만 따로
    n_ok = n_fail = 0
    idx_bad = 0
    idx_frac: list = []
    fails = []
    fbs = []

    for i in range(n):
        pal = i % len(PALETTE_PRESETS) if i < len(PALETTE_PRESETS) else None
        img, truth = make_clip(seed=i, palette=pal)
        yolo = simulate_yolo(truth, seed=i)

        res, g = analyze_clip(img, yolo)
        # analyze_clip 이 가장자리 점을 되살렸으면 reason 에 적혀 있다.
        # 그 경우는 정확도가 다른 등급이라 통계를 섞으면 안 된다.
        fb = g.ok and bool(g.reason)

        if not g.ok:
            n_fail += 1
            fails.append((i, truth.palette_name, g.reason))
            if verbose:
                print("  #%02d %-24s FAIL: %s" % (i, truth.palette_name[:24], g.reason))
            continue
        n_ok += 1

        dpx = abs(g.pitch_x - truth.pitch_x)
        dpy = abs(g.pitch_y - truth.pitch_y)
        dan = abs(_wrap90(g.angle_deg - truth.angle_deg))

        # 센터 코너가 진짜 격자점 위인가
        gt = np.asarray(truth.points, dtype=np.float64)
        dct = float(np.sqrt(((gt - g.center) ** 2).sum(axis=1)).min())

        # 좌표 -> 인덱스가 맞나.
        # xy 복원 오차(px)로 재면 안 된다 - pitch 가 194 px 이면 3 px 틀려도
        # 인덱스는 멀쩡하다. 인덱스가 뒤집히는 건 소수부가 0.5 를 넘을 때다.
        # 그래서 "정수에서 얼마나 벗어났나(칸)"를 재고, 0.5 까지의 여유를 본다.
        for q in gt:
            ijf = g.index_of(float(q[0]), float(q[1]), snap=False)
            frac = float(np.abs(ijf - np.round(ijf)).max())
            idx_frac.append(frac)
            if frac > 0.5:
                idx_bad += 1

        # 보정 없이 raw 좌표로만 격자를 세워 각도 비교
        g_raw = fit_grid(yolo, img_shape=img.shape)
        if g_raw.ok:
            e_ang_raw.append(abs(_wrap90(g_raw.angle_deg - truth.angle_deg)))

        if fb:
            e_ang_fb.append(dan)
            fbs.append((i, truth.palette_name, truth.pitch_x, dan, g.reason))
        else:
            e_px.append(dpx); e_py.append(dpy); e_ang.append(dan); e_ctr.append(dct)
        if verbose:
            print("  #%02d %-22s n=%2d  pitch %.2f/%.2f (오차 %.3f/%.3f)  "
                  "각 %+.3f (오차 %.4f)  센터 %.3f  잔차 %.3f"
                  % (i, truth.palette_name[:22], g.n_used,
                     g.pitch_x, g.pitch_y, dpx, dpy,
                     g.angle_deg, dan, dct, g.residual))

    f = lambda a: (float(np.median(a)), float(np.percentile(a, 90)), float(np.max(a)))
    return {
        "n": n, "ok": n_ok, "fail": n_fail, "fails": fails, "idx_bad": idx_bad,
        "idx_n": len(idx_frac),
        "idx_worst": float(np.max(idx_frac)) if idx_frac else float("nan"),
        "px": f(e_px), "py": f(e_py), "ang": f(e_ang), "ctr": f(e_ctr),
        "ang_raw": f(e_ang_raw) if e_ang_raw else (float("nan"),) * 3,
        "n_fb": len(fbs), "fbs": fbs,
        "ang_fb": f(e_ang_fb) if e_ang_fb else (float("nan"),) * 3,
    }


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("격자 추정 정확도 (합성 %d장)" % args.n)
    print("=" * 78)
    r = run(args.n, verbose=not args.quiet)

    print("-" * 78)
    print("성공 %d / %d   (실패 %d, 그중 가장자리 되살림 %d)"
          % (r["ok"], r["n"], r["fail"], r["n_fb"]))
    for i, nm, why in r["fails"]:
        print("    #%02d %-24s %s" % (i, nm[:24], why))
    print("아래 표는 **엄격 경로 %d장만** (되살림 건은 등급이 달라 따로 뺐다)"
          % (r["ok"] - r["n_fb"]))
    print("                     중앙값      p90        최악")
    print("  pitch_x 오차 :  %8.4f  %8.4f  %8.4f  px" % r["px"])
    print("  pitch_y 오차 :  %8.4f  %8.4f  %8.4f  px" % r["py"])
    print("  각도    오차 :  %8.4f  %8.4f  %8.4f  deg" % r["ang"])
    print("  센터코너 이탈:  %8.4f  %8.4f  %8.4f  px" % r["ctr"])
    print("-" * 78)
    if r["n_fb"]:
        print("가장자리 점까지 되살려 쓴 %d장 (pitch 가 커서 클립에 십자가 몇 개 안 들어온 경우)"
              % r["n_fb"])
        print("  각도 오차: 중앙값 %.4f  p90 %.4f  최악 %.4f deg" % r["ang_fb"])
        for i, nm, pxx, dan, why in r["fbs"]:
            print("    #%02d %-16s pitch %.0f  각오차 %.4f deg  (끝에서 %.0f px)"
                  % (i, nm[:16], pxx, dan, 5000.0 * math.tan(math.radians(dan))))
        print("-" * 78)
    print("  보정 없이(raw YOLO) 각도 오차: 중앙값 %.4f  p90 %.4f  최악 %.4f deg"
          % r["ang_raw"])
    gain = r["ang_raw"][0] / max(r["ang"][0], 1e-12)
    print("  -> 서브픽셀 보정이 각도를 %.1f배 좋게 만든다" % gain)
    print("-" * 78)
    print("  좌표->인덱스 오배정: %d / %d 개" % (r["idx_bad"], r["idx_n"]))
    print("    정수에서 벗어난 최악 %.4f 칸 (0.5 를 넘어야 옆칸으로 잘못 간다)"
          % r["idx_worst"])
    end = 5000.0 * math.tan(math.radians(r["ang"][2]))
    print("  최악 각도 오차로 10000px 웨이퍼 끝에서 밀리는 양: %.2f px" % end)
    print("=" * 78)


if __name__ == "__main__":
    _main()
