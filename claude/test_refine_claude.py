# -*- coding: utf-8 -*-
"""
test_refine_claude.py
====================================================================
서브픽셀 보정이 실제로 좋아지는지 **숫자로** 잰다.

측정 대상
--------------------------------------------------------------------
  raw     : YOLO 원본 좌표 -> 정답까지의 거리
  refined : 보정 좌표      -> 정답까지의 거리

보정이 의미 있으려면 refined 의 중앙값/p90 이 raw 보다 확실히 작아야 한다.
"작아졌다"가 아니라 "얼마나"를 봐야 해서 분포로 찍는다.

오검출(FP) 구분
--------------------------------------------------------------------
simulate_yolo 는 die 한복판에도 점을 뿌린다. 그 점들은 보정해도 십자가
아니므로 오차 통계에 넣으면 안 된다. 정답까지 거리가 FP_CUT 보다 먼 점을
FP 로 보고 따로 뺀 뒤, confidence 가 진짜 점과 갈리는지 확인한다.
(갈리지 않으면 호출부가 FP 를 거를 방법이 없다는 뜻이라 그것도 실패다)

실행
--------------------------------------------------------------------
    python test_refine_claude.py               # 프리셋 8 + 랜덤 22
    python test_refine_claude.py --n 60
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from synth_clip_claude import PALETTE_PRESETS, make_clip, simulate_yolo
from via_refine_claude import refine_points, rough_pitch_from_points

FP_CUT = 8.0        # 정답까지 이보다 멀면 오검출로 간주 (jitter sigma 2.5 의 3.2배)


def _nn_dist(pts: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """각 점에서 가장 가까운 정답까지의 거리와 그 정답 인덱스."""
    if len(pts) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    d2 = np.sum((pts[:, None, :] - truth[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    return np.sqrt(d2[np.arange(len(pts)), idx]), idx


def run(n: int, presets_first: bool, verbose: bool) -> dict:
    raw_all: List[float] = []
    ref_all: List[float] = []
    conf_true: List[float] = []
    conf_fp: List[float] = []
    rows = []

    for i in range(n):
        pal = (i % len(PALETTE_PRESETS)) if (presets_first and i < len(PALETTE_PRESETS)) else None
        img, truth = make_clip(seed=i, palette=pal)
        yolo = simulate_yolo(truth, seed=i)

        pts = np.asarray(yolo, dtype=np.float32)
        gt = np.asarray(truth.points, dtype=np.float32)

        d_raw, _ = _nn_dist(pts, gt)
        is_true = d_raw <= FP_CUT

        p_hint = rough_pitch_from_points(yolo)
        res = refine_points(img, yolo, pitch_hint=p_hint)
        d_ref, _ = _nn_dist(res.points, gt)

        # border 점은 refine 이 "못 믿겠다"고 표시한 것들이라 통계에서 뺀다.
        # (윈도우가 이미지 밖으로 나가 대칭이 깨진 점들. conf 도 0 으로 눌려 있다)
        use = is_true & ~res.border
        raw_all.extend(d_raw[use].tolist())
        ref_all.extend(d_ref[use].tolist())
        conf_true.extend(res.confidence[use].tolist())
        conf_fp.extend(res.confidence[~is_true & ~res.border].tolist())

        rw = float(np.median(d_raw[use])) if use.any() else float("nan")
        rf = float(np.median(d_ref[use])) if use.any() else float("nan")
        rows.append((i, truth.palette_name, len(gt), int(use.sum()),
                     int((~is_true).sum()), rw, rf, res.win))
        if verbose:
            print("  #%02d %-24s gt=%2d use=%2d bd=%d fp=%d  raw=%.2f -> ref=%.3f  win=%d"
                  % (i, truth.palette_name[:24], len(gt), int(use.sum()),
                     int(res.border.sum()), int((~is_true).sum()), rw, rf, res.win))

    raw = np.asarray(raw_all)
    ref = np.asarray(ref_all)
    ct = np.asarray(conf_true)
    cf = np.asarray(conf_fp)

    return {
        "n_img": n, "n_pts": len(raw),
        "raw_med": float(np.median(raw)), "raw_p90": float(np.percentile(raw, 90)),
        "ref_med": float(np.median(ref)), "ref_p90": float(np.percentile(ref, 90)),
        "ref_max": float(ref.max()) if len(ref) else float("nan"),
        "conf_true_med": float(np.median(ct)) if len(ct) else float("nan"),
        "conf_true_p10": float(np.percentile(ct, 10)) if len(ct) else float("nan"),
        "conf_fp_med": float(np.median(cf)) if len(cf) else float("nan"),
        "conf_fp_p90": float(np.percentile(cf, 90)) if len(cf) else float("nan"),
        "n_fp": len(cf),
        "rows": rows,
    }


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("서브픽셀 보정 정확도 (합성 %d장)" % args.n)
    print("=" * 78)
    r = run(args.n, presets_first=True, verbose=not args.quiet)

    print("-" * 78)
    print("정답점 %d개 (이미지 %d장)" % (r["n_pts"], r["n_img"]))
    print("  raw     중앙값 %.3f px   p90 %.3f px" % (r["raw_med"], r["raw_p90"]))
    print("  refined 중앙값 %.3f px   p90 %.3f px   최악 %.3f px"
          % (r["ref_med"], r["ref_p90"], r["ref_max"]))
    gain = r["raw_med"] / max(r["ref_med"], 1e-9)
    print("  개선 배수 %.2fx" % gain)
    print("-" * 78)
    print("confidence 분리도 (오검출 %d개)" % r["n_fp"])
    print("  진짜 점  중앙값 %.3f   하위10%% %.3f" % (r["conf_true_med"], r["conf_true_p10"]))
    print("  오검출   중앙값 %.3f   상위10%% %.3f" % (r["conf_fp_med"], r["conf_fp_p90"]))
    sep = "OK" if r["conf_true_p10"] > r["conf_fp_p90"] else "겹침 - 임계값으로 못 가른다"
    print("  판정: %s" % sep)
    print("=" * 78)


if __name__ == "__main__":
    _main()
