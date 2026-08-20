# -*- coding: utf-8 -*-
"""
test_yolo_api_claude.py
====================================================================
`build_die_map_from_yolo` 를 **사용자가 쓸 그 모양 그대로** 부른다.

    dm = build_die_map_from_yolo(
        wafer_image=wafer_bgr,
        clip_image=center_clip_bgr,
        detections=results[0].boxes.xywh.cpu().numpy(),
        detection_format="xywh",
        refine=True, refine_mode="auto", refine_radius=24,
        refine_noise_kernel=5, refine_min_confidence=0.15,
    )

여기서는 합성 웨이퍼를 만들고, 정중앙 512 클립을 잘라, YOLO 가 냈을 법한
(N,4) xywh 배열을 흉내낸다. 정답을 알고 있으므로 오차가 숫자로 나온다.

측정
--------------------------------------------------------------------
  x0, y0        : center corner 가 진짜 격자점 위인가 (칸 단위)
  pitch, angle  : 정답 대비 오차
  angle_conf    : 신뢰도가 실제 오차와 맞아떨어지나
  dies_by_index : dies 와 같은 걸 가리키나
  index         : 좌표 -> die index 가 정답 격자와 몇 칸 어긋나나
  aligned_image : 실제로 축정렬이 되나 + 비용이 얼마나 드나

실행
--------------------------------------------------------------------
    python test_yolo_api_claude.py --n 6 --size 2000
    python test_yolo_api_claude.py --n 1 --size 10000 --aligned
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import time

import numpy as np

from synth_wafer_claude import make_wafer

# 통짜 파일(wafer_via_claude.py)도 **같은 테스트**로 재도록 모듈을 갈아끼운다.
# 합친 파일이 원본 세 모듈과 갈라졌는지 확인하는 게 목적이다.
#     python test_yolo_api_claude.py                      # 세 모듈
#     VIA_MODULE=wafer_via_claude python test_yolo_api_claude.py   # 통짜 파일
_M = importlib.import_module(os.environ.get("VIA_MODULE", "via_diemap_claude"))
build_die_map_from_yolo = _M.build_die_map_from_yolo
detections_to_points = _M.detections_to_points
locate_die_via = _M.locate_die_via


def _wrap90(a: float) -> float:
    return ((a + 45.0) % 90.0) - 45.0


def truth_ij_frac(truth, x: float, y: float) -> np.ndarray:
    """정답 격자에서의 **실수** 인덱스 (index_at 은 정수만 준다)."""
    V = np.stack([truth.vx, truth.vy], axis=1)
    return np.linalg.solve(V, np.asarray([x, y], float)
                           - np.asarray(truth.origin_px, float))


def fake_yolo_xywh(truth, clip_origin, clip_size, seed=0, box=18.0):
    """
    십자점을 감싸는 bbox 를 (N,4) xywh 로 만든다 - ultralytics 출력 흉내.

    YOLO 는 정확히 중심을 안 맞추므로 중심에 ±1.5 px 흔들림을 준다
    (실측 raw YOLO 오차 중앙값이 3.0 px 이라 오히려 후한 편이다).
    폭/높이도 흔들어서 "w,h 는 안 쓴다"는 걸 확인한다.
    """
    rng = np.random.default_rng(seed)
    ox, oy = clip_origin
    org = np.asarray(truth.origin_px, float)
    # 클립 네 귀퉁이를 격자 인덱스로 바꿔 훑을 범위를 정한다
    corners = [truth_ij_frac(truth, ox + dx * clip_size, oy + dy * clip_size)
               for dx in (0, 1) for dy in (0, 1)]
    C = np.array(corners)
    i0, j0 = np.floor(C.min(axis=0)).astype(int) - 1
    i1, j1 = np.ceil(C.max(axis=0)).astype(int) + 1

    out = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            g = org + i * truth.vx + j * truth.vy      # 웨이퍼 좌표의 십자점
            x, y = g[0] - ox, g[1] - oy
            if not (0 <= x < clip_size and 0 <= y < clip_size):
                continue
            out.append((x + rng.normal(0.0, 1.5), y + rng.normal(0.0, 1.5),
                        box * rng.uniform(0.7, 1.4), box * rng.uniform(0.7, 1.4)))
    return np.asarray(out, dtype=np.float32)      # (N,4) float32, torch 와 동일


def run_one(seed: int, size: int, want_aligned: bool, verbose: bool) -> dict:
    img, truth = make_wafer(seed=seed, size=size)
    H, W = img.shape[:2]
    cs = 512
    ox, oy = (W - cs) // 2, (H - cs) // 2
    clip = img[oy:oy + cs, ox:ox + cs].copy()

    det = fake_yolo_xywh(truth, (ox, oy), cs, seed=seed)
    if len(det) < 4:
        return {"seed": seed, "skip": "클립 안 십자가 %d개뿐" % len(det)}

    t0 = time.time()
    dm = build_die_map_from_yolo(
        wafer_image=img,
        clip_image=clip,
        detections=det,
        detection_format="xywh",
        refine=True,
        refine_mode="auto",
        refine_radius=24,
        refine_noise_kernel=5,
        refine_min_confidence=0.15,
    )
    t_build = time.time() - t0

    # --- 요청된 반환값이 전부 있고 말이 되나 -----------------------------
    assert isinstance(dm.x0, float) and isinstance(dm.y0, float)
    assert dm.num_dies == len(dm.dies) == len(dm.dies_by_index)
    assert dm.grid_angle_deg == dm.angle_deg
    wb = dm.wafer_boundary
    bcx, bcy, br = wb                              # 언패킹도 되어야 한다
    assert (bcx, bcy, br) == (dm.wafer_cx, dm.wafer_cy, dm.wafer_r)

    # dies_by_index 가 dies 와 **같은 객체**를 가리키나 (사본이면 수정이 갈린다)
    same_obj = all(dm.dies_by_index[tuple(d["index"])] is d for d in dm.dies)

    # --- 정확도 ----------------------------------------------------------
    d_ctr = math.hypot(dm.wafer_cx - truth.wafer_cx, dm.wafer_cy - truth.wafer_cy)
    d_r = abs(dm.wafer_r - truth.wafer_r)
    d_px = abs(dm.pitch_x - truth.pitch_x)
    d_py = abs(dm.pitch_y - truth.pitch_y)
    d_ang = abs(_wrap90(dm.grid_angle_deg - truth.angle_deg))

    # center corner 가 진짜 격자점 위인가 - 정답 격자로 재면 정수여야 한다
    f = truth_ij_frac(truth, dm.x0, dm.y0)
    org_fr = float(np.abs(f - np.round(f)).max())

    # 좌표 -> 인덱스: 두 격자가 몇 칸 어긋났나 (정수 이탈이 아니라 **칸 차이**)
    rng = np.random.default_rng(seed + 7)
    q = []
    while len(q) < 400:
        x = rng.uniform(0, W); y = rng.uniform(0, H)
        if math.hypot(x - truth.wafer_cx, y - truth.wafer_cy) < truth.wafer_r * 0.9:
            q.append((x, y))
    d_ij = np.array([np.asarray(dm.index_of(x, y, snap=False))
                     - truth_ij_frac(truth, x, y)
                     for (x, y) in q])
    off = np.round(d_ij[0])
    cell_err = float(np.abs(d_ij - off).max())

    # locate_die_via 가 index_of 와 같은 답을 주나
    bad_loc = 0
    for (x, y) in q[:100]:
        info = locate_die_via(dm, point=(x, y))
        if tuple(info["die_index"]) != dm.index_of(x, y):
            bad_loc += 1

    # --- aligned_image ---------------------------------------------------
    al = {"used": False}
    if want_aligned:
        t1 = time.time()
        A = dm.aligned_image
        al["sec"] = time.time() - t1
        al["used"] = True
        al["shape"] = None if A is None else A.shape
        al["mb"] = None if A is None else A.nbytes / 1024.0 / 1024.0
        # 정말 축정렬이 됐나: vx 를 같은 행렬로 돌리면 y 성분이 0 이어야 한다
        M = dm.aligned_transform()
        v = M[:, :2] @ dm.vx
        al["vx_y"] = float(abs(v[1]))              # 0 에 가까워야 한다
        al["cached"] = dm.aligned_image is A       # 두 번째 접근은 캐시

    r = {"seed": seed, "n_det": len(det), "n_used": dm.diagnostics.n_dies,
         "d_ctr": d_ctr, "d_r": d_r, "d_px": d_px, "d_py": d_py,
         "d_ang": d_ang, "org_fr": org_fr, "cell_err": cell_err,
         "bad_loc": bad_loc, "num_dies": dm.num_dies,
         "conf": dm.angle_confidence, "sigma": dm.angle_sigma_deg,
         "same_obj": same_obj, "t_build": t_build,
         "contour": None if wb.contour is None else len(wb.contour),
         "aligned": al}
    if verbose:
        print("  seed %2d  det %3d  dies %5d  pitch오차 %.3f/%.3f  "
              "각오차 %.4f (conf %.3f)  원점 %.4f칸  격자 %.2e칸  "
              "locate불일치 %d  %.2fs"
              % (seed, len(det), dm.num_dies, d_px, d_py, d_ang,
                 dm.angle_confidence, org_fr, cell_err, bad_loc, t_build))
    return r


def _check_formats() -> None:
    """detection_format 변환이 맞나 + 정규화 좌표를 막나."""
    xywh = np.array([[100.0, 200.0, 20.0, 30.0],
                     [300.0, 400.0, 10.0, 10.0]], np.float32)
    p = detections_to_points(xywh, "xywh")
    assert np.allclose(p, [[100, 200], [300, 400]]), p

    xyxy = np.array([[90.0, 185.0, 110.0, 215.0],
                     [295.0, 395.0, 305.0, 405.0]], np.float32)
    assert np.allclose(detections_to_points(xyxy, "xyxy"),
                       [[100, 200], [300, 400]])

    # conf/cls 가 더 붙어도 앞 4열만
    six = np.hstack([xywh, np.ones((2, 2), np.float32)])
    assert np.allclose(detections_to_points(six, "xywh"), [[100, 200], [300, 400]])

    # 정규화 좌표(xywhn)는 거부해야 한다
    try:
        detections_to_points(xywh / 512.0, "xywh")
    except ValueError as e:
        assert "정규화" in str(e)
    else:
        raise AssertionError("정규화 좌표를 그냥 통과시켰다")
    print("  detection_format 변환/검증 OK (xywh, xyxy, +conf/cls, xywhn 거부)")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--aligned", action="store_true",
                    help="aligned_image 까지 만들어 비용을 잰다 (메모리 주의)")
    args = ap.parse_args()

    print("=" * 78)
    print("build_die_map_from_yolo 전체 사슬 (합성 웨이퍼 %d장, %dx%d)"
          % (args.n, args.size, args.size))
    print("=" * 78)
    _check_formats()
    print("-" * 78)

    rows = []
    for s in range(args.n):
        r = run_one(s, args.size, args.aligned, verbose=True)
        if "skip" in r:
            print("  seed %2d  건너뜀: %s" % (s, r["skip"]))
            continue
        rows.append(r)

    if not rows:
        print("측정할 게 없다")
        return
    g = lambda k: np.array([r[k] for r in rows], float)
    print("-" * 78)
    print("성공 %d / %d" % (len(rows), args.n))
    print("  웨이퍼 중심 오차 : 최악 %.2f px" % g("d_ctr").max())
    print("  웨이퍼 반지름    : 최악 %.2f px" % g("d_r").max())
    print("  pitch 오차       : x 최악 %.4f, y 최악 %.4f px"
          % (g("d_px").max(), g("d_py").max()))
    print("  회전각 오차      : 중앙 %.4f, 최악 %.4f deg"
          % (np.median(g("d_ang")), g("d_ang").max()))
    print("  angle_confidence : 최저 %.3f (sigma 최악 %.4f deg)"
          % (g("conf").min(), g("sigma").max()))
    ok_conf = int(((g("conf") >= 0.5) == (g("d_ang") < 0.05)).sum())
    print("    신뢰도>=0.5 와 실제오차<0.05deg 가 일치: %d / %d"
          % (ok_conf, len(rows)))
    print("  x0,y0 격자 이탈  : 최악 %.4f 칸" % g("org_fr").max())
    print("  격자 불일치      : 최악 %.2e 칸" % g("cell_err").max())
    print("  locate 불일치    : %d 건" % int(g("bad_loc").sum()))
    print("  dies_by_index 가 dies 와 같은 객체: %s"
          % ("전부 그렇다" if all(r["same_obj"] for r in rows) else "**아니다**"))
    print("  wafer_boundary.contour 점 개수: %s"
          % [r["contour"] for r in rows])
    print("  die map 생성     : 최악 %.2f s (die %d개까지)"
          % (g("t_build").max(), int(g("num_dies").max())))
    if args.aligned:
        a = [r["aligned"] for r in rows if r["aligned"]["used"]]
        print("-" * 78)
        print("  aligned_image    : %s, %.0f MB, %.2f s"
              % (a[-1]["shape"], a[-1]["mb"], max(x["sec"] for x in a)))
        print("    회전 후 vx 의 y성분: 최악 %.2e px (0 이면 축정렬)"
              % max(x["vx_y"] for x in a))
        print("    두 번째 접근이 캐시인가: %s"
              % ("그렇다" if all(x["cached"] for x in a) else "**아니다**"))
    print("=" * 78)


if __name__ == "__main__":
    _main()
