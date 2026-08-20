# -*- coding: utf-8 -*-
"""
test_diemap_claude.py - 클립 -> 격자 -> 웨이퍼 die map 전체 사슬 검증

무엇을 재는가
--------------------------------------------------------------------
지금까지 테스트한 건 "512 클립 안에서 pitch/angle 이 맞나"까지였다.
여기서는 그 다음을 잰다.

    1) 외곽선   : detect_wafer_adaptive 가 찾은 (cx,cy,r) vs 정답
    2) 좌표이동 : 클립 격자 원점 + clip_origin 이 웨이퍼 좌표에서 맞나
                  (build_die_map_via 의 핵심 가정. 지금까지 미검증)
    3) die 개수 : 원 넓이 / die 넓이 와 맞는가
    4) 인덱스   : 임의 좌표 -> die index 가 정답 격자와 같은가

인덱스 비교 방법 (주의)
--------------------------------------------------------------------
analyze_clip 은 **클립 중앙의 십자점**을 자기 die(0,0) 으로 삼는다.
정답 격자의 원점은 다른 칸이므로, 두 인덱스는 항상 **상수만큼** 어긋난다.
그래서 절대값을 비교하면 안 되고, 기준점 하나로 오프셋을 구한 뒤
나머지 전부가 같은 오프셋인지를 본다. 한 점이라도 다르면 실패다.

실행
--------------------------------------------------------------------
    python test_diemap_claude.py --n 8 --size 2000
    python test_diemap_claude.py --n 1 --size 10000 --save out
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Tuple

import numpy as np

from synth_wafer_claude import make_wafer, WaferTruth
from via_grid_claude import analyze_clip
from via_diemap_claude import build_die_map_via, locate_die_via, save_debug_overlay


def clip_yolo_points(t: WaferTruth, rng: np.random.Generator,
                     *, jitter: float = 3.0, drop: float = 0.15,
                     n_fp: int = 2) -> List[Tuple[float, float]]:
    """정답 격자에서 클립 안 십자점을 뽑아 YOLO 출력을 흉내낸다.

    실제 YOLO 는 몇 px 흔들리고, 몇 개를 놓치고, 가끔 엉뚱한 곳을 찍는다.
    셋 다 넣어야 의미 있는 테스트가 된다.
    """
    cs = t.clip_size
    n = int(math.ceil(cs * 1.5 / min(t.pitch_x, t.pitch_y))) + 2
    ii, jj = np.mgrid[-n:n + 1, -n:n + 1]
    ii, jj = ii.ravel(), jj.ravel()
    pts = (np.array(t.origin_px)[None, :]
           + ii[:, None] * t.vx[None, :] + jj[:, None] * t.vy[None, :])
    # 웨이퍼 이미지 좌표 -> 클립 좌표
    pts = pts - np.array([t.clip_x0, t.clip_y0], float)[None, :]
    keep = ((pts[:, 0] >= 8) & (pts[:, 0] <= cs - 9) &
            (pts[:, 1] >= 8) & (pts[:, 1] <= cs - 9))
    pts = pts[keep]

    out: List[Tuple[float, float]] = []
    for p in pts:
        if rng.random() < drop:
            continue
        out.append((float(p[0] + rng.normal(0, jitter)),
                    float(p[1] + rng.normal(0, jitter))))
    for _ in range(n_fp):
        out.append((float(rng.uniform(0, cs - 1)), float(rng.uniform(0, cs - 1))))
    return out


def run_one(seed: int, size: int, save_dir: str = "") -> dict:
    rng = np.random.default_rng(10000 + seed)
    img, t = make_wafer(seed=seed, size=size)
    clip = img[t.clip_y0:t.clip_y0 + t.clip_size,
               t.clip_x0:t.clip_x0 + t.clip_size].copy()

    yolo = clip_yolo_points(t, rng)
    res, g = analyze_clip(clip, yolo)
    if not g.ok:
        return {"ok": False, "why": "격자 실패: %s" % g.reason, "seed": seed}

    dm = build_die_map_via(img, g, clip_origin=(t.clip_x0, t.clip_y0))

    # --- 1) 외곽선 ---------------------------------------------------
    d_ctr = math.hypot(dm.wafer_cx - t.wafer_cx, dm.wafer_cy - t.wafer_cy)
    d_r = abs(dm.wafer_r - t.wafer_r)

    # --- 2) 좌표이동 : 격자 원점이 정답 격자 위에 있나 ------------------
    # 원점 자체는 다른 칸일 수 있으므로, 정답 격자로 환산했을 때
    # **정수에서 얼마나 벗어났는지**를 본다. 0 이면 완벽히 같은 격자다.
    V = np.stack([t.vx, t.vy], axis=1)
    ij = np.linalg.solve(V, dm.origin - np.array(t.origin_px))
    origin_frac = float(np.abs(ij - np.round(ij)).max())

    # --- 3) die 개수 -------------------------------------------------
    expect = math.pi * t.wafer_r ** 2 / (t.pitch_x * t.pitch_y)
    n_ratio = len(dm.dies) / expect if expect > 0 else 0.0

    # --- 4) 좌표 -> 인덱스 --------------------------------------------
    # 웨이퍼 안 임의의 점 500개. 정답 인덱스와 **상수 오프셋**만 나야 한다.
    ang = rng.uniform(0, 2 * math.pi, 500)
    rad = t.wafer_r * 0.95 * np.sqrt(rng.random(500))
    qx = t.wafer_cx + rad * np.cos(ang)
    qy = t.wafer_cy + rad * np.sin(ang)

    # 여기서 "정수에서 벗어난 정도"를 재면 안 된다. 질의점이 균등 난수라
    # 소수부도 균등이고 최악은 구조적으로 0.5 가 나온다 (오차가 0 이어도).
    # 재야 할 것은 **두 격자가 몇 칸 어긋났나** 다.
    Vt = np.stack([t.vx, t.vy], axis=1)
    Vm = np.stack([dm.vx, dm.vy], axis=1)
    Q = np.stack([qx, qy], axis=1)
    ij_t = np.linalg.solve(Vt, (Q - np.array(t.origin_px)).T).T
    ij_m = np.linalg.solve(Vm, (Q - dm.origin).T).T
    d_ij = ij_m - ij_t
    off = np.round(d_ij[0])                     # 상수 오프셋 (원점이 다른 칸)
    cell_err = float(np.abs(d_ij - off).max())  # 칸 단위 불일치 = 진짜 오차

    # 인덱스가 실제로 갈리는 건 질의점이 경계에서 cell_err 안쪽일 때뿐이다.
    bad = 0
    for x, y in zip(qx, qy):
        gi, gj = t.index_at(float(x), float(y))
        mi, mj = dm.index_of(float(x), float(y))
        if (mi - gi, mj - gj) != (int(off[0]), int(off[1])):
            bad += 1

    # --- 5) locate_die_via 가 die map 과 일치하나 ----------------------
    mis = 0
    for x, y in zip(qx[:50], qy[:50]):
        r = locate_die_via(dm, point=(float(x), float(y)))
        if r["die_index"] != dm.index_of(float(x), float(y)):
            mis += 1
        e = dm.get_die(*r["die_index"])
        if e is not None and e["center_px"] != r["die_center_px"]:
            mis += 1

    if save_dir:
        save_debug_overlay(img, dm, "%s/wafer_%02d.png" % (save_dir, seed))

    return {
        "ok": True, "seed": seed, "palette": t.palette_name,
        "d_ctr": d_ctr, "d_r": d_r,
        "d_px": abs(dm.pitch_x - t.pitch_x), "d_py": abs(dm.pitch_y - t.pitch_y),
        "d_ang": abs(dm.angle_deg - t.angle_deg),
        "origin_frac": origin_frac,
        "n_dies": len(dm.dies), "n_ratio": n_ratio,
        "idx_bad": bad, "cell_err": cell_err, "locate_mis": mis,
        "fallback": bool(g.reason),
        "elapsed": dm.diagnostics.elapsed_sec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    t0 = time.time()
    rows = []
    for s in range(a.n):
        try:
            rows.append(run_one(s, a.size, a.save))
        except Exception as e:            # noqa: BLE001
            rows.append({"ok": False, "seed": s, "why": "%s: %s"
                         % (type(e).__name__, e)})

    good = [r for r in rows if r.get("ok")]
    print("=" * 72)
    print("die map 전체 사슬 테스트  (size=%d, %d장, %.1fs)"
          % (a.size, a.n, time.time() - t0))
    print("=" * 72)
    print("%-5s %-10s %7s %6s %7s %7s %8s %7s %6s %5s %9s"
          % ("seed", "palette", "d_ctr", "d_r", "d_px", "d_py", "d_ang",
             "org_fr", "dies", "idx", "cell_err"))
    for r in rows:
        if not r.get("ok"):
            print("%-5d %s" % (r["seed"], r.get("why", "?")))
            continue
        print("%-5d %-10s %7.2f %6.1f %7.3f %7.3f %8.4f %7.4f %6d %5d %9.2e%s"
              % (r["seed"], r["palette"][:10], r["d_ctr"], r["d_r"],
                 r["d_px"], r["d_py"], r["d_ang"], r["origin_frac"],
                 r["n_dies"], r["idx_bad"], r["cell_err"],
                 "  FB" if r["fallback"] else ""))

    print("-" * 72)
    if not good:
        print("성공 0 장 - 전부 실패했다")
        return
    A = lambda k: np.array([r[k] for r in good], float)   # noqa: E731
    print("성공        : %d / %d" % (len(good), a.n))
    print("중심 오차   : 중앙값 %.2f  최악 %.2f px" % (np.median(A("d_ctr")), A("d_ctr").max()))
    print("반지름 오차 : 중앙값 %.2f  최악 %.2f px" % (np.median(A("d_r")), A("d_r").max()))
    print("pitch 오차  : x 최악 %.3f  y 최악 %.3f px" % (A("d_px").max(), A("d_py").max()))
    print("각도 오차   : 중앙값 %.4f  최악 %.4f deg" % (np.median(A("d_ang")), A("d_ang").max()))
    print("원점 정수이탈: 최악 %.4f 칸  (0 이면 정답 격자와 동일)" % A("origin_frac").max())
    print("die 수/이론 : 중앙값 %.3f  (원 넓이/die 넓이 대비)" % np.median(A("n_ratio")))
    ce = A("cell_err").max()
    print("격자 불일치 : 최악 %.2e 칸  (= die pitch 의 %.2e 배)" % (ce, ce))
    nq = 500 * len(good)
    print("인덱스 오답 : %d / %d 점  (기대치 ~%.1f)"
          % (int(A("idx_bad").sum()), nq, 2 * ce * 2 * nq))
    print("  ! 오답은 질의점이 die 경계에서 %.2e 칸 안쪽에 떨어졌을 때만 난다." % ce)
    print("    경계 위의 점은 원래 어느 칸인지 애매하므로 오차가 아니다.")
    print("locate 불일치: %d 건" % int(A("locate_mis").sum()))
    print("die map 생성: 중앙값 %.2f s" % np.median(A("elapsed")))


if __name__ == "__main__":
    main()
