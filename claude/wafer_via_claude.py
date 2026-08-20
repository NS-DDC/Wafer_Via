# -*- coding: utf-8 -*-
"""
wafer_via_claude.py
====================================================================
YOLO 십자점 -> 격자 -> 웨이퍼 die map.  **이 파일 하나면 된다.**

    pip install numpy opencv-python        # 이게 전부

쓰는 법
--------------------------------------------------------------------
    import cv2
    from ultralytics import YOLO
    from wafer_via_claude import build_die_map_from_yolo

    wafer_bgr      = cv2.imread("wafer.png")            # 10000x10000
    center_clip_bgr = wafer_bgr[h//2-256:h//2+256,      # 정중앙 512x512
                                w//2-256:w//2+256]
    results = YOLO("best.pt")(center_clip_bgr)

    dm = build_die_map_from_yolo(
        wafer_image=wafer_bgr,
        clip_image=center_clip_bgr,
        detections=results[0].boxes.xywh.cpu().numpy(),
        detection_format="xywh",
        refine=True,
        refine_mode="auto",
        refine_radius=24,
        refine_noise_kernel=5,
        refine_min_confidence=0.15,
    )

    print(dm.x0, dm.y0)          # 전체 wafer 좌표의 center corner
    print(dm.pitch_x, dm.pitch_y)
    print(dm.grid_angle_deg)
    print(dm.angle_confidence)   # P(|각도오차| < 0.05deg), 0..1
    print(dm.num_dies)
    print(dm.dies)               # [{index, center_px, quad_px, ...}, ...]
    print(dm.dies_by_index)      # {(i,j): 위와 **같은 객체**}
    print(dm.wafer_boundary)     # WaferBoundary(cx, cy, r, contour)
    print(dm.aligned_image)      # 회전 보정본 (처음 볼 때 만든다, 286 MB)

    # 좌표 -> die
    info = locate_die_via(dm, point=(7321.0, 4180.5))
    print(info["die_index"], info["die_center_px"], info["is_edge"])

파이프라인
--------------------------------------------------------------------
    512x512 클립 + YOLO 검출 (N,4)
        |  detections_to_points     bbox 중심만 뽑는다
        |  refine_points            서브픽셀 보정 (색-무관)
        |  fit_grid                 센터 코너 / pitch_x,y / 회전각 / 신뢰도
        |  detect_wafer_adaptive    웨이퍼 외곽선 (색-무관)
        v  build_die_map_via        격자를 웨이퍼 전체로 외삽
    ViaDieMap

색에 안 흔들리는 이유
--------------------------------------------------------------------
회색조로 안 바꾼다. 회색조는 밝기가 같고 색만 다른 die/street 쌍을
통째로 지워버린다. 대신 **배경색으로부터의 Lab 거리**를 본다.
"검정보다 밝은가"가 아니라 "배경과 다른가"라서 극성 가정이 없다.

이 파일은 build_single_claude.py 가 만든 것이다 (직접 고치지 말고
via_refine/via_grid/via_diemap 을 고친 뒤 다시 만들 것).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    # --- 진입점 (보통 이것만 쓴다) ---------------------------------
    "build_die_map_from_yolo", "locate_die_via",
    # --- 결과 형식 -------------------------------------------------
    "ViaDieMap", "WaferBoundary", "GridFit", "RefineResult",
    "WaferProfile", "DieMapDiag",
    # --- 단계별로 직접 부르고 싶을 때 ------------------------------
    "detections_to_points", "refine_points", "streetness_map",
    "rough_pitch_from_points", "fit_grid", "analyze_clip",
    "detect_wafer_adaptive", "clean_wafer_outside", "build_die_map_via",
    # --- 잘라내기 / 확인용 -----------------------------------------
    "clip_die", "crop_die", "save_debug_overlay",
]


# ####################################################################
# 1) 서브픽셀 보정 - YOLO 좌표를 십자 중심으로 당긴다  (원본: wafer_via_claude.py)
# ####################################################################

CENTER_TOL = 2.0    # 봉우리가 이만큼(px) 벗어나면 confidence 가 절반이 된다


# ====================================================================
# 1) streetness 맵
# ====================================================================

def streetness_map(img: np.ndarray,
                   bg_ksize: int = 81,
                   dot_ksize: int = 5,
                   normalize: bool = True) -> np.ndarray:
    """
    색/극성에 무관하게 street 를 밝게 만드는 스칼라 맵. float32, 대략 [0,1].

    bg_ksize : die 색 지도를 만드는 median 크기 (홀수).

               **pitch 에 맞먹는 크기로 크게 잡아야 한다.** (기본 81)
               직관과 반대라 이유를 적어 둔다:
                 - 창이 street 폭의 2~3배 정도로 어중간하면, 십자 중심에서는
                   창 면적의 80% 넘게가 street 라서 median 자체가 street 색이
                   된다 -> 정작 십자 위에서 streetness 가 0 으로 꺼진다.
                 - 창이 pitch 만큼 크면 한 주기를 통째로 보는데, die 면적이
                   보통 78% 이상이라 median 이 안정적으로 die 색이 된다.
               실측(합성 5장, street 폭 5~16px):
                   k=pitch/6  -> 십자 위 streetness 0.15~0.24 (die 0.04~0.20)
                                 심하면 die 가 십자보다 밝게 뒤집힘
                   k=pitch*0.8-> 십자 위 streetness 0.90~0.99 (die 0.03~0.14)
               속도는 같다(히스토그램 기반이라 k 에 거의 무관, 512x512 에 ~35ms).
    dot_ksize: die 위 점 노이즈를 죽이는 median 크기 (3 또는 5).
               OpenCV 제약상 float32 에는 3/5 만 된다.

    normalize=True 면 99 퍼센타일로 나눠 [0,1] 로 만든다.
    (절대값이 아니라 상대값이라 이미지마다 대비가 달라도 같은 스케일)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("BGR 3채널 이미지가 필요하다. shape=%r" % (img.shape,))

    k = int(bg_ksize) | 1                      # 홀수 강제
    k = max(3, k)
    src = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)

    bg = cv2.medianBlur(src, k)

    lab_i = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.sqrt(np.sum((lab_i - lab_b) ** 2, axis=2, dtype=np.float32))

    dk = int(dot_ksize) | 1
    if dk in (3, 5):
        d = cv2.medianBlur(d, dk)              # 점 노이즈 제거

    if normalize:
        hi = float(np.percentile(d, 99.0))
        d = np.clip(d / max(hi, 1e-6), 0.0, 1.0).astype(np.float32)
    return d


# ====================================================================
# 2) 대략적 pitch (윈도우 크기 정하는 용도)
# ====================================================================

def rough_pitch_from_points(points: Sequence[Tuple[float, float]]) -> float:
    """
    점들의 최근접 이웃 거리 중앙값. 정확할 필요 없다 —
    보정 윈도우 크기를 정하는 데만 쓴다.

    점이 2개 미만이면 0.0 을 돌려주고, 호출부가 기본값을 쓰게 한다.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return 0.0
    # N 이 작아서(수십 개) 전수 거리 계산이 제일 단순하고 빠르다.
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return float(np.median(nn))


# ====================================================================
# 3) 보정
# ====================================================================

@dataclass
class RefineResult:
    points: np.ndarray        # (N,2) float32 보정 좌표
    confidence: np.ndarray    # (N,)  float32 0~1. border 인 점은 0 으로 눌러 둔다
    conf_raw: np.ndarray      # (N,)  float32 border 를 누르기 **전** 값.
                              #       점이 너무 적어 border 점까지 써야 할 때만 쓴다
    moved: np.ndarray         # (N,)  float32 원래 좌표에서 움직인 거리(px)
    border: np.ndarray        # (N,)  bool   윈도우가 이미지 밖으로 나간 점
    win: int                  # 실제로 쓴 윈도우 반경
    streetness: np.ndarray    # (H,W) float32 (디버그/오버레이용)


def _peak_centroid(prof: np.ndarray, rel: float = 0.25
                   ) -> Optional[Tuple[float, float]]:
    """
    1D 프로파일에서 **가장 높은 봉우리 하나**의 무게중심과 그 봉우리 높이.

    바닥(median)을 빼고, 최대점에서 좌우로 rel*최대값 위에 머무는 동안만
    확장한 뒤 그 구간에서만 무게중심을 잡는다.
    구간을 제한하는 이유: 윈도우가 넓으면 옆 street 의 봉우리가 같이 들어오는데,
    그걸 함께 평균 내면 중심이 두 봉우리 사이로 끌려간다.

    반환 (무게중심 인덱스, 봉우리 높이). 봉우리가 없으면 None.
    봉우리 높이는 "이 축에 street 가 정말 있나"의 근거로 쓴다.
    """
    q = prof - float(np.median(prof))
    np.clip(q, 0.0, None, out=q)
    if not np.isfinite(q).all():
        return None
    i = int(np.argmax(q))
    top = float(q[i])
    if top <= 1e-9:
        return None

    thr = rel * top
    lo = i
    while lo - 1 >= 0 and q[lo - 1] > thr:
        lo -= 1
    hi = i
    while hi + 1 < len(q) and q[hi + 1] > thr:
        hi += 1

    seg = q[lo:hi + 1]
    w = float(seg.sum())
    if w <= 1e-9:
        return None
    idx = np.arange(lo, hi + 1, dtype=np.float64)
    return float((idx * seg).sum() / w), top


def _keep_mask(win: int, excl: int) -> Tuple[int, np.ndarray]:
    """중심 띠를 뺀 대칭 마스크. (excl 이 너무 크면 줄여서라도 행을 남긴다)"""
    w = int(win)
    e = int(max(1, excl))
    if e >= w - 1:
        e = max(1, w // 3)
    keep = np.ones(2 * w + 1, dtype=bool)
    keep[w - e: w + e + 1] = False
    return w, keep


def _profiles(s: np.ndarray, cx: float, cy: float, w: int, keep: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """(cx,cy) 중심 패치에서 중심 띠를 뺀 x/y 프로파일. getRectSubPix 가 서브픽셀 보간."""
    side = 2 * w + 1
    patch = cv2.getRectSubPix(s, (side, side), (cx, cy))
    return patch[keep, :].mean(axis=0), patch[:, keep].mean(axis=1)


def prominence_at(s: np.ndarray, pts: np.ndarray, win: int, excl: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 점에서 x/y 프로파일의 (봉우리 높이, 중심에서 벗어난 거리)를 잰다.
    반환 (prom (N,2), off (N,2)).

    **보정이 끝난 최종 위치에서** 재야 한다. 보정 도중 값을 그대로 쓰면,
    옆 십자로 튀었다가 되돌려진 오검출이 그 십자의 높은 점수를 들고 온다.
    (실측: die 한복판 오검출이 conf 0.81~0.88 로 진짜와 안 갈렸다)

    off 가 필요한 이유: 봉우리 높이만 보면 "윈도우 **안에** 십자가 있나"를
    재는 셈이라, 십자에서 30 px 떨어진 점도 높게 나온다(win 이 그만큼 넓다).
    "십자 **위에** 있나"를 재려면 봉우리가 윈도우 중앙에 와야 한다.
    """
    w, keep = _keep_mask(win, excl)
    prom = np.zeros((len(pts), 2), dtype=np.float32)
    off = np.full((len(pts), 2), float(w), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        px, py = _profiles(s, float(x), float(y), w, keep)
        rx = _peak_centroid(px)
        ry = _peak_centroid(py)
        if rx is not None:
            prom[i, 0], off[i, 0] = rx[1], abs(rx[0] - w)
        if ry is not None:
            prom[i, 1], off[i, 1] = ry[1], abs(ry[0] - w)
    return prom, off


def _refine_one(s: np.ndarray, x: float, y: float, win: int, excl: int,
                n_iter: int = 3) -> Tuple[float, float]:
    """
    점 하나를 띠 프로파일 무게중심으로 반복 보정.

    excl 이 핵심이다. streetness 는 street 위에서 포화(=1)라서,
    십자 중심을 **지나는** 띠는 모든 열이 밝다 -> 봉우리가 없다.
    그래서 중심 근처 |dy| < excl 을 **빼고**, 위/아래 두 조각만 쓴다.
    그 영역에는 세로 street 말고 밝은 게 없으므로 봉우리가 하나만 남는다.

    위/아래를 **대칭**으로 쓰는 게 중요하다. 격자가 기울면 위 조각의
    street 는 왼쪽, 아래 조각은 오른쪽으로 밀리는데, 대칭이면 그 둘이
    상쇄돼 무게중심이 제자리를 지킨다. (그래서 각도를 몰라도 된다)
    """
    w, keep = _keep_mask(win, excl)
    cx, cy = float(x), float(y)

    for _ in range(n_iter):
        px_prof, py_prof = _profiles(s, cx, cy, w, keep)
        rx = _peak_centroid(px_prof)      # 열(=x) 프로파일
        ry = _peak_centroid(py_prof)      # 행(=y) 프로파일
        if rx is None and ry is None:
            break
        dx = (rx[0] - w) if rx is not None else 0.0
        dy = (ry[0] - w) if ry is not None else 0.0
        cx += dx
        cy += dy
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            break
    return cx, cy


def refine_points(img: np.ndarray,
                  points: Sequence[Tuple[float, float]],
                  pitch_hint: Optional[float] = None,
                  win: Optional[int] = None,
                  excl: Optional[int] = None,
                  bg_ksize: Optional[int] = None,
                  max_move: Optional[float] = None,
                  streetness: Optional[np.ndarray] = None,
                  ) -> RefineResult:
    """
    YOLO 점을 서브픽셀 보정한다.

    pitch_hint : 대략적 pitch(px). None 이면 points 로 추정.
    win        : 윈도우 반경. None 이면 pitch*0.3 (6~70 으로 클립).
                 옆 십자를 물면 안 되므로 pitch/2 보다 작아야 한다.
    excl       : 중심에서 제외할 띠의 반높이. None 이면 pitch*0.10 (4~24).
                 street 반폭(최대 pitch*0.055)보다 확실히 커야
                 가로 street 가 프로파일에 안 섞인다.
    bg_ksize   : streetness 의 median 크기. None 이면 pitch*0.8 을 홀수로
                 (최소 9, 최대 201). 왜 크게 잡는지는 streetness_map 참고
                 (작게 잡으면 십자 중심에서 streetness 가 꺼진다).
    max_move   : 이 거리보다 많이 움직인 보정은 **되돌린다**.
                 보정이 실패하면 엉뚱한 곳으로 튀는데, 원래 YOLO 좌표보다
                 나쁜 답을 주느니 원본을 쓰는 게 낫다.
                 None 이면 pitch*0.25 (윈도우 안에서만 움직이게).

    반환 RefineResult. 오검출은 **버리지 않고** confidence 로 표시만 한다.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return RefineResult(pts.copy(), z, z.copy(), z.copy(),
                            np.zeros((0,), dtype=bool), 0,
                            np.zeros(img.shape[:2], dtype=np.float32))

    p_hint = float(pitch_hint) if pitch_hint else rough_pitch_from_points(pts)
    if p_hint <= 0:
        p_hint = 120.0                            # 512 클립에 4~5개 들어가는 크기

    if bg_ksize is None:
        bg_ksize = int(np.clip(round(p_hint * 0.8) | 1, 9, 201))
    if win is None:
        win = int(np.clip(round(p_hint * 0.30), 6, 70))
    if excl is None:
        excl = int(np.clip(round(p_hint * 0.10), 4, 24))
    if max_move is None:
        max_move = float(p_hint * 0.25)

    s = streetness_map(img, bg_ksize=bg_ksize) if streetness is None else streetness

    ref = np.empty_like(pts)
    for i, (x, y) in enumerate(pts):
        ref[i] = _refine_one(s, float(x), float(y), int(win), int(excl))

    moved = np.sqrt(np.sum((ref - pts) ** 2, axis=1)).astype(np.float32)

    # 발산 방지: 너무 많이 움직였거나 NaN 이면 원본으로 되돌린다.
    bad = ~np.isfinite(ref).all(axis=1) | (moved > max_move)
    if np.any(bad):
        ref[bad] = pts[bad]
        moved[bad] = 0.0

    # 이미지 밖으로 나간 것도 되돌린다.
    h, w = s.shape[:2]
    oob = ((ref[:, 0] < 0) | (ref[:, 0] > w - 1) |
           (ref[:, 1] < 0) | (ref[:, 1] > h - 1))
    if np.any(oob):
        ref[oob] = pts[oob]
        moved[oob] = 0.0

    # 신뢰도 = 두 축 봉우리 높이 중 **약한 쪽**. (되돌리기까지 끝난 최종 위치에서)
    #
    # 보정 위치의 streetness 를 그대로 쓰면 안 된다(처음엔 그렇게 했다).
    # street 한복판을 잘못 찍은 오검출은 streetness 가 1.0 이라 진짜 십자와
    # 구분이 안 된다(실측: 진짜 하위10% 0.904 vs 오검출 상위10% 0.936, 겹침).
    # 십자는 **가로/세로 street 가 둘 다** 있어야 하므로, 한 축이라도
    # 봉우리가 약하면 십자가 아니다. min 이 그걸 정확히 잡아낸다.
    #
    # 배치 안에서 다시 정규화하지 않는다. streetness 가 이미 이미지의 99
    # 퍼센타일로 [0,1] 이라, 진짜 십자의 봉우리 높이는 자연스럽게 0.9 근처,
    # die 한복판은 0 근처로 나온다. 배치 정규화를 하면 그 배치에 오검출이
    # 몇 개 섞였느냐에 따라 같은 점의 confidence 가 달라져서 못 쓴다.
    # 윈도우가 이미지 밖으로 나가는 점은 신뢰할 수 없다.
    # getRectSubPix 는 밖을 **가장자리 픽셀 복사**로 채우는데, 그러면 위/아래
    # 조각의 대칭이 깨진다. 복사된 쪽은 가장자리 행의 street 위치만 반복해서
    # 강조하므로, 격자가 기울어 있으면 봉우리가 그쪽으로 끌려간다.
    #   실측(합성 30장): 윈도우가 완전히 안에 있는 289점 -> 오차 최대 0.249 px
    #                    가장자리에 걸친 81점        -> p90 1.109, 최대 2.241 px
    #                    (십자에서 9 px 벗어난 채 conf 0.67 로 나온 사례도 있었다)
    # 그래서 좌표는 그대로 두되(그래도 raw 보단 낫다) confidence 를 0 으로
    # 눌러서 호출부가 거르게 한다. border 플래그로 이유도 같이 알려 준다.
    border = ((ref[:, 0] < win) | (ref[:, 0] >= w - win) |
              (ref[:, 1] < win) | (ref[:, 1] >= h - win))

    prom, off = prominence_at(s, ref, int(win), int(excl))
    conf = np.clip(np.minimum(prom[:, 0], prom[:, 1]), 0.0, 1.0)

    # 봉우리가 윈도우 중앙에 안 오면 그만큼 깎는다.
    # 보정이 성공한 점은 0.01 px 이내로 수렴하므로 깎이지 않는다. 반대로
    # max_move 에 걸려 되돌려진 오검출은 봉우리가 수십 px 밖에 있어서 죽는다.
    # CENTER_TOL 은 "수렴했다고 볼 여유" 라서 값이 예민하지 않다(1~3 px 동일).
    conf = (conf / (1.0 + (off.max(axis=1) / CENTER_TOL) ** 2)).astype(np.float32)
    conf_raw = conf.copy()
    conf = conf.copy()
    conf[border] = 0.0

    return RefineResult(points=ref, confidence=conf, conf_raw=conf_raw,
                        moved=moved, border=border, win=int(win), streetness=s)


# ####################################################################
# 2) 격자 - 센터 코너 / pitch_x,y / 회전각 / 신뢰도  (원본: wafer_via_claude.py)
# ####################################################################

# ====================================================================
# 결과 형식
# ====================================================================

@dataclass
class GridFit:
    ok: bool                  # 격자를 세웠나
    reason: str               # ok=False 면 실패 사유.
                              # ok=True 인데도 비어있지 않으면 "세우긴 했는데
                              # 이런 사정이 있었다"는 경고다 (analyze_clip 참고).

    center: np.ndarray        # (2,) 센터 코너 (클립 중앙에 가장 가까운 십자)
    origin: np.ndarray        # (2,) 격자 원점 = center 와 동일 (i=j=0 기준점)
    vx: np.ndarray            # (2,) 한 칸 오른쪽 벡터
    vy: np.ndarray            # (2,) 한 칸 아래쪽 벡터

    pitch_x: float            # |vx|
    pitch_y: float            # |vy|
    angle_deg: float          # 격자 회전각 (-45 ~ 45)

    n_used: int               # 격자 맞춤에 쓴 점 개수
    residual: float           # 맞춤 잔차 RMS (px). 크면 뭔가 틀린 것
    ij: np.ndarray            # (n_used, 2) int, 쓴 점들의 격자 인덱스
    used: np.ndarray          # (N,) bool, 입력 점 중 어떤 걸 썼나

    angle_sigma_deg: float = float("nan")   # 회전각 추정치의 표준편차 (deg)
    angle_confidence: float = 0.0           # P(|각도오차| < 0.05deg), 0..1

    def xy_of(self, i: float, j: float) -> np.ndarray:
        """격자 인덱스 -> 픽셀 좌표."""
        return self.origin + i * self.vx + j * self.vy

    def index_of(self, x: float, y: float, snap: bool = True):
        """
        픽셀 좌표 -> 격자 인덱스.
        snap=True 면 정수로 반올림, False 면 실수 그대로.
        """
        V = np.stack([self.vx, self.vy], axis=1)          # 2x2, 열이 기저
        ij = np.linalg.solve(V, np.asarray([x, y], np.float64) - self.origin)
        return (int(round(ij[0])), int(round(ij[1]))) if snap else ij


def _fail(reason: str, n: int) -> GridFit:
    z2 = np.zeros(2, np.float64)
    return GridFit(ok=False, reason=reason, center=z2, origin=z2,
                   vx=z2.copy(), vy=z2.copy(), pitch_x=0.0, pitch_y=0.0,
                   angle_deg=0.0, n_used=0, residual=float("nan"),
                   ij=np.zeros((0, 2), int), used=np.zeros(n, bool))


# ====================================================================
# 1) 이웃 차 벡터 -> 기저 (vx, vy)
# ====================================================================

def _basis_from_pairs(p: np.ndarray,
                      band: Tuple[float, float] = (0.45, 3.0),
                      min_sin: float = 0.3,
                      off_tol: float = 0.2,
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    점들의 "한 칸짜리" 차 벡터를 모아 기저 두 개를 만든다.

    두 단계로 한다.

    1) 가장 짧은 벡터 v1, 그리고 v1 과 평행하지 않은 것 중 가장 짧은 v2.
       (격자 기저 축소 - 한 칸 벡터는 정의상 가장 짧은 두 독립 벡터다)
    2) 모든 차 벡터를 (v1,v2) 로 분해해 인덱스가 정확히 (±1,0)/(0,±1) 인
       것만 모아 평균낸다. 대각선/두 칸 벡터는 여기서 자동으로 빠진다.

    왜 이렇게 바꿨나 (길이 문턱 하나로 자르면 안 되는 이유)
    ------------------------------------------------------------------
    예전에는 "최근접 거리 중앙값 * 1.4 이하"를 한 칸으로 봤다.
    1.4 는 **정사각** 격자의 대각선 배율 sqrt(2)=1.414 에서 온 값인데,
    pitch_x != pitch_y 면 그 가정이 깨진다.

        실측 실패 예 (합성 seed 18): px=121.94, py=136.92
        최근접 거리 중앙값 step = 136.86 -> 문턱 191.60
        대각선 = sqrt(px^2+py^2) = 183.35  < 191.60   ** 통과해 버린다 **

    대각선 5개가 섞이자 |dx|<|dy| 라는 이유로 전부 "세로 방향"으로 분류돼
    vy 평균이 망가졌고, pitch_y 가 136.9 대신 61.1 로, 회전각이 -3.51 대신
    -7.36 도로 나왔다. die map 전체가 60 칸 어긋났다.
    길이만 보는 한 어떤 상수를 넣어도 pitch 비율에 따라 언젠가 깨진다.

    band    : v1/v2 후보 길이 범위 = (하한, 상한) * 최근접거리 중앙값.
              하한은 중복 검출(거의 같은 자리 두 점)을 막고,
              상한은 pitch 비가 커도 반대 방향 한 칸이 들어오게 넉넉히 둔다.
              어차피 "가장 짧은 것"을 고르므로 상한이 넉넉해도 안전하다.
    min_sin : v1 과 독립으로 볼 최소 |sin|. 0.3 = 약 17도.
    off_tol : 2단계에서 정수 인덱스로 인정할 오차 (칸 단위).
    """
    n = len(p)
    if n < 3:
        return None

    d = p[:, None, :] - p[None, :, :]                  # (n,n,2) 양방향 다 있음
    L = np.sqrt((d ** 2).sum(axis=2))
    np.fill_diagonal(L, np.inf)
    step = float(np.median(L.min(axis=1)))             # 최근접 거리 중앙값
    if not np.isfinite(step) or step <= 1e-6:
        return None

    iu = np.triu_indices(n, 1)
    v_all = d[iu]                                      # (k,2) 한쪽 방향만
    l_all = L[iu]
    sel = (l_all >= band[0] * step) & (l_all <= band[1] * step)
    if sel.sum() < 2:
        return None
    vb, lb = v_all[sel], l_all[sel]

    # --- 1) 가장 짧은 두 독립 벡터 -----------------------------------
    order = np.argsort(lb)
    v1 = vb[order[0]]
    n1 = float(np.hypot(v1[0], v1[1]))
    v2 = None
    for k in order[1:]:
        vk = vb[k]
        nk = float(np.hypot(vk[0], vk[1]))
        if nk <= 1e-6:
            continue
        sin = abs(float(v1[0] * vk[1] - v1[1] * vk[0])) / (n1 * nk)
        if sin >= min_sin:
            v2 = vk
            break
    if v2 is None:
        return None

    V = np.stack([v1, v2], axis=1).astype(np.float64)
    if abs(float(np.linalg.det(V))) < 1e-6:
        return None

    # --- 2) 정확히 한 칸인 벡터만 모아 평균 ----------------------------
    a = d.reshape(-1, 2).astype(np.float64)
    ijf = np.linalg.solve(V, a.T).T
    ij = np.round(ijf)
    fit = np.abs(ijf - ij).max(axis=1) < off_tol

    def _mean_step(di: int, dj: int) -> Optional[np.ndarray]:
        pos = fit & (ij[:, 0] == di) & (ij[:, 1] == dj)
        neg = fit & (ij[:, 0] == -di) & (ij[:, 1] == -dj)
        if not pos.any() and not neg.any():
            return None
        return np.concatenate([a[pos], -a[neg]], axis=0).mean(axis=0)

    e1 = _mean_step(1, 0)
    e2 = _mean_step(0, 1)
    if e1 is None or e2 is None:
        return None

    # |회전각| < 45도 가정 -> 두 기저 중 |dx|>|dy| 인 쪽이 가로다
    if abs(e1[0]) >= abs(e1[1]):
        vx, vy = e1, e2
    else:
        vx, vy = e2, e1
    if vx[0] < 0:                                      # vx 는 오른쪽(+x)
        vx = -vx
    if vy[1] < 0:                                      # vy 는 아래(+y)
        vy = -vy
    return vx.astype(np.float64), vy.astype(np.float64)


# ====================================================================
# 2) 최소제곱 격자 맞춤
# ====================================================================

def _lstsq_grid(p: np.ndarray, ij: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    p ~= o + i*vx + j*vy 를 최소제곱으로 푼다.

    미지수 6개(o, vx, vy) 를 x/y 각각 3개씩 나눠 푼다.
    설계행렬 A = [1, i, j] 는 x, y 가 공유하므로 한 번만 만든다.

    돌려주는 sii, sjj 는 (A^T A)^-1 의 대각 성분이다. 잔차와 곱하면
    vx, vy 각 성분의 분산이 되고, 그게 회전각 불확실도의 재료다.
    (자세한 건 _angle_sigma 참고)
    """
    A = np.column_stack([np.ones(len(ij)), ij[:, 0], ij[:, 1]]).astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, p.astype(np.float64), rcond=None)   # (3,2)
    o, vx, vy = sol[0], sol[1], sol[2]
    resid = p - A @ sol
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    try:
        cov = np.linalg.inv(A.T @ A)
        sii, sjj = float(cov[1, 1]), float(cov[2, 2])
    except np.linalg.LinAlgError:                 # 점이 한 줄로 늘어선 경우
        sii = sjj = float("nan")
    return o, vx, vy, rms, sii, sjj


# 회전각 신뢰도를 만들 때 "이 정도면 맞다"고 볼 각도 오차 (deg).
# 반지름 5000 px 웨이퍼 끝에서 5000*tan(0.05deg) = 4.36 px 밀린다.
# die pitch 가 100~200 px 이니 한 칸의 2~4% 다.
ANGLE_TOL_DEG = 0.05

# 자유도가 0 이면 잔차가 0 으로 나온다. 그때 "오차 0" 이라고 우기면 안 되므로
# 서브픽셀 보정의 실측 오차 분포(중앙값 0.070, p90 0.155 px)에서 가져온
# 바닥값을 대신 쓴다. test_refine_claude.py 참고.
REFINE_NOISE_FLOOR_PX = 0.155


def _angle_sigma(rms: float, n: int, sii: float, sjj: float,
                 pitch_x: float, pitch_y: float) -> float:
    """
    회전각 추정치의 표준편차 (deg). 못 내면 nan.

    _angle_from_basis 는 단위벡터 ux 와 rot90(vy) 를 **더해서** 각을 낸다.
    단위벡터 둘의 합이 가리키는 방향은 정확히 두 각의 이등분선이므로
    추정량은 그냥 평균이다:  t_hat = (a_x + a_y) / 2.

    a_x 의 흔들림은 vx 의 **수직** 성분 오차를 |vx| 로 나눈 것이고,
    a_y 도 같은 식이다. 두 항의 공분산은

        cov(perp_x, perp_y) = (1/(px*py)) * sigma^2 * S_ij * (-sin t cos t + sin t cos t)
                            = 0

    으로 1차항에서 상쇄된다 (x/y 좌표가 같은 설계행렬을 공유하기 때문).
    따라서

        sigma_t = 0.5 * sqrt( (sigma_c*sqrt(Sii)/px)^2 + (sigma_c*sqrt(Sjj)/py)^2 )

    sigma_c 는 좌표 한 성분의 잔차 표준편차다. rms 는 x,y 를 합쳐 잰 값이라
    관측 2n 개 - 미지수 6 개 = 2n-6 자유도로 나눈다.
    """
    if not np.isfinite(sii) or not np.isfinite(sjj) or pitch_x <= 0 or pitch_y <= 0:
        return float("nan")
    dof = 2 * n - 6
    if dof <= 0:
        sig_c = REFINE_NOISE_FLOOR_PX            # 잔차가 구조적으로 0 인 구간
    else:
        sig_c = math.sqrt(max(n * rms * rms, 0.0) / dof)
        sig_c = max(sig_c, REFINE_NOISE_FLOOR_PX * 0.5)
    sx = sig_c * math.sqrt(max(sii, 0.0)) / pitch_x
    sy = sig_c * math.sqrt(max(sjj, 0.0)) / pitch_y
    return math.degrees(0.5 * math.hypot(sx, sy))


def _angle_confidence(sigma_deg: float, tol_deg: float = ANGLE_TOL_DEG) -> float:
    """
    각도 표준편차 -> 0..1 신뢰도.

    "참값이 +-tol 안에 있을 확률" 을 정규분포로 읽은 것이다:
        P(|err| < tol) = erf( tol / (sigma * sqrt(2)) )
    임의로 만든 점수가 아니라 확률이라 해석이 된다.
    sigma 를 못 내면 0.0 (모른다 = 못 믿는다).
    """
    if not np.isfinite(sigma_deg):
        return 0.0
    if sigma_deg <= 0:
        return 1.0
    return float(math.erf(tol_deg / (sigma_deg * math.sqrt(2.0))))


# ====================================================================
# 3) 메인
# ====================================================================

def fit_grid(points: Sequence[Tuple[float, float]],
             conf: Optional[Sequence[float]] = None,
             conf_min: float = 0.3,
             img_shape: Optional[Tuple[int, ...]] = None,
             center_xy: Optional[Tuple[float, float]] = None,
             max_resid: float = 3.0,
             ) -> GridFit:
    """
    보정된 십자 좌표에서 격자를 세운다.

    points   : (N,2) 보정 좌표 (refine_points 결과를 넣으면 된다)
    conf     : (N,) 신뢰도. 주면 conf_min 미만은 버린다.
               refine_points 는 오검출과 가장자리 점을 여기서 0 으로 눌러 준다.
    conf_min : 신뢰도 문턱. 실측상 진짜 점은 0.47 이상, 오검출은 0.01 이하라
               0.3 이면 넉넉히 갈린다.
    img_shape: 클립 크기. 주면 그 중심에서 가장 가까운 점을 센터 코너로 잡는다.
    center_xy: 센터 코너 기준점을 직접 주고 싶을 때. (img_shape 보다 우선)
    max_resid: 맞춤 잔차가 이보다 크면 ok=False. 격자가 아닌 걸 격자라고
               우기지 않게 하는 안전장치.

    실패해도 예외를 던지지 않고 ok=False + reason 으로 돌려준다.
    (호출부가 클립 한 장 실패로 전체가 죽는 걸 원하지 않을 것이므로)
    """
    p_all = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n_in = len(p_all)

    keep = np.ones(n_in, dtype=bool)
    if conf is not None:
        c = np.asarray(conf, dtype=np.float64).ravel()
        if len(c) != n_in:
            return _fail("conf 길이가 points 와 다르다", n_in)
        keep &= c >= conf_min

    p = p_all[keep]
    if len(p) < 3:
        return _fail("신뢰할 점이 %d개뿐이다 (최소 3개)" % len(p), n_in)

    base = _basis_from_pairs(p)
    if base is None:
        return _fail("이웃 벡터에서 기저를 못 만들었다", n_in)
    vx, vy = base

    # 기저로 정수 인덱스를 매긴다. 기준점은 아무거나(첫 점) — 뒤에서 옮긴다.
    V = np.stack([vx, vy], axis=1)
    if abs(float(np.linalg.det(V))) < 1e-6:
        return _fail("기저 두 벡터가 평행하다", n_in)
    ij_f = np.linalg.solve(V, (p - p[0]).T).T
    ij = np.round(ij_f).astype(int)

    # 정수에서 너무 벗어난 점은 격자 밖(오검출)이므로 뺀다.
    off = np.abs(ij_f - ij).max(axis=1)
    good = off < 0.25
    if good.sum() < 3:
        return _fail("격자에 붙는 점이 %d개뿐이다" % int(good.sum()), n_in)
    p, ij = p[good], ij[good]

    # 같은 격자칸에 두 점이 겹치면(중복 검출) 하나만 남긴다.
    _, uniq = np.unique(ij, axis=0, return_index=True)
    p, ij = p[np.sort(uniq)], ij[np.sort(uniq)]
    if len(p) < 3:
        return _fail("중복 제거 후 점이 %d개뿐이다" % len(p), n_in)

    o, vx, vy, rms, sii, sjj = _lstsq_grid(p, ij)

    # 센터 코너 = 클립 중앙(또는 지정 좌표)에 가장 가까운 실제 십자
    if center_xy is not None:
        cxy = np.asarray(center_xy, dtype=np.float64)
    elif img_shape is not None:
        cxy = np.asarray([img_shape[1] * 0.5, img_shape[0] * 0.5], dtype=np.float64)
    else:
        cxy = p.mean(axis=0)
    k = int(np.argmin(((p - cxy) ** 2).sum(axis=1)))

    # 인덱스 원점을 센터 코너로 옮긴다 -> 센터가 (0,0), 옆이 (1,0), 밑이 (0,1)
    # origin 은 p[k] 원본이 아니라 **최소제곱이 예측한** 센터 위치를 쓴다.
    # 그래야 그 점 하나의 보정 오차가 격자 전체에 실리지 않는다.
    ij_k = ij[k].copy()
    ij = ij - ij_k
    o = o + ij_k[0] * vx + ij_k[1] * vy

    ang = _angle_from_basis(vx, vy)

    used = np.zeros(n_in, dtype=bool)
    used[np.nonzero(keep)[0][good][np.sort(uniq)]] = True

    px_, py_ = float(np.hypot(*vx)), float(np.hypot(*vy))
    a_sig = _angle_sigma(rms, len(p), sii, sjj, px_, py_)

    return GridFit(ok=(rms <= max_resid), reason=("" if rms <= max_resid else
                   "잔차 %.2f px 가 한계 %.2f px 를 넘었다" % (rms, max_resid)),
                   center=p[k].copy(), origin=o,
                   vx=vx, vy=vy,
                   pitch_x=px_, pitch_y=py_,
                   angle_deg=ang, n_used=len(p), residual=rms,
                   ij=ij, used=used,
                   angle_sigma_deg=a_sig,
                   angle_confidence=_angle_confidence(a_sig))


def analyze_clip(clip: np.ndarray,
                 yolo_points: Sequence[Tuple[float, float]],
                 conf_min: float = 0.3,
                 *,
                 refine: bool = True,
                 win: Optional[int] = None,
                 noise_kernel: int = 0,
                 ):
    """
    **이게 사용자가 부르는 함수다.**
    512x512 센터 클립 + YOLO 점 리스트 -> 보정 결과 + 격자.

    refine       : False 면 서브픽셀 보정을 건너뛰고 raw 좌표로 격자를 세운다.
                   실측상 각도 오차가 56배 나빠지므로 비교용이 아니면 켜 둔다.
    win          : 보정 윈도우 **반경**. None 이면 pitch 에서 자동으로 뽑는다
                   (실측상 pitch*0.30 이 제일 좋았다).
    noise_kernel : 보정 전에 클립에 걸 medianBlur 커널 (홀수, 0 이면 안 건다).
                   "흰색+갈색 노이즈" 같은 점잡음용이다. streetness 의
                   bg_ksize(≈pitch) 와는 **다른 물건**이니 헷갈리면 안 된다.

        res, g = analyze_clip(clip, [(x, y), ...])
        g.pitch_x, g.pitch_y, g.angle_deg, g.center

    반환 (RefineResult, GridFit).

    가장자리 점 되살리기
    ----------------------------------------------------------------
    refine_points 는 윈도우가 이미지 밖으로 나간 점의 confidence 를 0 으로
    누른다(대칭이 깨져 못 믿으므로). 그런데 pitch 가 아주 크면(190 px 이상)
    512 클립 안에 십자가 2~3개밖에 안 들어와서, 그걸 다 빼면 격자를 못 세운다.
    그럴 때만 conf_raw(=누르기 전 값)로 한 번 더 시도한다.
    성공하면 GridFit.reason 에 그 사실을 적어 둔다 — 조용히 넘어가지 않는다.
    """

    src = clip
    k = int(noise_kernel)
    if k >= 3:
        src = cv2.medianBlur(clip, k | 1)

    res = refine_points(src, yolo_points, win=win,
                        pitch_hint=rough_pitch_from_points(yolo_points))
    if not refine:
        # 보정을 끄더라도 신뢰도/가장자리 판정은 그대로 쓴다. 좌표만 raw 로 되돌린다.
        res.points = np.asarray(yolo_points, np.float32).reshape(-1, 2).copy()

    g = fit_grid(res.points, conf=res.confidence,
                 conf_min=conf_min, img_shape=clip.shape)
    if g.ok:
        return res, g

    g2 = fit_grid(res.points, conf=res.conf_raw,
                  conf_min=conf_min, img_shape=clip.shape)
    if g2.ok:
        g2.reason = ("점이 부족해 가장자리 점까지 썼다 "
                     "(%d개 중 %d개가 가장자리) - 정확도가 낮을 수 있다"
                     % (int(g2.used.sum()), int((g2.used & res.border).sum())))
        return res, g2
    return res, g


def _angle_from_basis(vx: np.ndarray, vy: np.ndarray) -> float:
    """
    vx, vy 를 **둘 다** 써서 회전각을 낸다.

    vy 를 -90도 돌리면 vx 와 같은 방향이 된다(직교 기저니까).
    그 둘을 단위벡터로 만들어 더하면, 한쪽 방향 점이 적어도 각도가 안 흔들린다.
    결과는 -45~45 로 접는다 (90도 대칭이라 그 이상은 의미가 없다).
    """
    ux = vx / max(np.hypot(*vx), 1e-9)
    # vy 를 시계 반대로 90도 회전: (x,y) -> (y,-x)
    uy = np.asarray([vy[1], -vy[0]], dtype=np.float64)
    uy /= max(np.hypot(*uy), 1e-9)
    if float(ux @ uy) < 0:            # 반대로 뒤집혀 있으면 맞춰 준다
        uy = -uy
    u = ux + uy
    a = math.degrees(math.atan2(float(u[1]), float(u[0])))
    return ((a + 45.0) % 90.0) - 45.0


# ####################################################################
# 3) die map - 웨이퍼 외곽선, 격자 외삽, 좌표->die  (원본: wafer_via_claude.py)
# ####################################################################

_EPS = 1e-9

DEFAULT_PIXEL_PER_UNIT = 1
DEFAULT_EDGE_MARGIN = 1.0
DEFAULT_EDGE_MODE = "circle"          # "circle" | "ring" | "both"
DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_MARGIN_X = 0
DEFAULT_MARGIN_Y = 0


# ====================================================================
# 설정 / 진단
# ====================================================================

@dataclass
class WaferProfile:
    """외곽선 검출 파라미터. **전부 None/기본값이면 자동으로 돈다.**

    아는 값만 채워도 된다. (v6 ColorProfile 에서 외곽선에 쓰이는 것만 남김)
    """
    background_bgr: Optional[Tuple[int, int, int]] = None   # 배경색을 알면 지정
    wafer_fill_bgr: Optional[Tuple[int, int, int]] = None   # clean 시 채울 색
    wafer_otsu_max_dim: int = 1024      # Otsu 임계를 계산할 다운스케일 크기
    wafer_open_ksize: int = 5           # 3000px 기준 open 커널 (해상도에 비례 조정)
    wafer_close_ksize: int = 9          # 3000px 기준 close 커널 (시작값, 점진 확대)


@dataclass
class DieMapDiag:
    """자가진단. 조용히 틀리는 대신 여기에 근거를 남긴다."""
    background_bgr: Optional[Tuple[int, int, int]] = None
    background_source: str = ""
    wafer_circle: Optional[Tuple[int, int, int]] = None
    wafer_coverage: Optional[float] = None
    wafer_fallback: str = ""
    wafer_contour: Optional[np.ndarray] = None   # 실제 외곽선 (원이 아닌 notch 포함)
    n_dies: int = 0
    elapsed_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def report(self) -> str:
        L = ["[die map 자가진단]"]
        L.append("  배경색      : %s (%s)" % (self.background_bgr,
                                              self.background_source or "-"))
        L.append("  웨이퍼 원   : %s" % (self.wafer_circle,))
        if self.wafer_coverage is not None:
            L.append("  화면 점유율 : %.3f" % self.wafer_coverage)
        if self.wafer_fallback:
            L.append("  외곽선 대체 : %s" % self.wafer_fallback)
        L.append("  die 개수    : %d" % self.n_dies)
        L.append("  소요        : %.3f s" % self.elapsed_sec)
        if self.warnings:
            L.append("  경고 %d 건:" % len(self.warnings))
            L.extend("    - " + w for w in self.warnings)
        else:
            L.append("  경고        : 없음")
        return "\n".join(L)


@dataclass
class WaferBoundary:
    """웨이퍼 외곽 원. (cx, cy, r) 로 그냥 언패킹해도 된다."""
    cx: int
    cy: int
    r: int
    contour: Optional[np.ndarray] = None    # (M,1,2) int32, 있으면 실제 외곽선

    def __iter__(self):
        return iter((self.cx, self.cy, self.r))

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.r - margin

    def __repr__(self) -> str:
        return ("WaferBoundary(cx=%d, cy=%d, r=%d, contour=%s)"
                % (self.cx, self.cy, self.r,
                   "None" if self.contour is None else "%d pts" % len(self.contour)))


@dataclass
class ViaDieMap:
    """die map 결과. v5 WaferDieMap 의 필드 이름을 최대한 유지했다."""
    wafer_cx: int
    wafer_cy: int
    wafer_r: int

    origin: np.ndarray          # (2,) 격자 원점 = die(0,0) 의 좌상 꼭지점 (십자점)
    vx: np.ndarray              # (2,) i 방향 한 칸 벡터
    vy: np.ndarray              # (2,) j 방향 한 칸 벡터 (아래쪽)
    pitch_x: float
    pitch_y: float
    angle_deg: float

    die_w: int
    die_h: int
    pixel_per_unit: int

    dies: List[Dict[str, Any]]
    dies_by_index: Dict[Tuple[int, int], Dict[str, Any]]
    image_shape: Tuple[int, int]
    wafer_mask: np.ndarray
    edge_mode: str
    quadrant_report: Dict[str, Any]
    diagnostics: DieMapDiag

    # -- 아래는 기본값이 있는 필드 (기존 호출부를 안 깨려고 뒤에 붙였다) ----
    angle_sigma_deg: float = float("nan")   # 회전각 표준편차 (GridFit 에서)
    angle_confidence: float = 0.0           # P(|각도오차| < 0.05deg), 0..1
    wafer_contour: Optional[np.ndarray] = None      # 실제 외곽선 (있으면)
    source_image: Optional[np.ndarray] = None       # aligned_image 재료 (참조만)
    _aligned: Optional[np.ndarray] = field(default=None, repr=False)

    # -- 요청된 별칭 ---------------------------------------------------
    # 격자 원점 origin 이 곧 "전체 wafer 좌표의 center corner" 다.
    # 클립 안에서 찾은 센터 코너를 웨이퍼 좌표로 옮긴 값이라 그대로 쓰면 된다.
    @property
    def x0(self) -> float:
        return float(self.origin[0])

    @property
    def y0(self) -> float:
        return float(self.origin[1])

    @property
    def grid_angle_deg(self) -> float:
        return self.angle_deg

    @property
    def num_dies(self) -> int:
        return len(self.dies)

    @property
    def wafer_boundary(self) -> WaferBoundary:
        return WaferBoundary(self.wafer_cx, self.wafer_cy, self.wafer_r,
                             self.wafer_contour)

    @property
    def aligned_image(self) -> Optional[np.ndarray]:
        """
        격자가 축정렬이 되도록 -angle_deg 만큼 돌린 이미지. 처음 쓸 때 만든다.

        **주의: 이건 보기용이지 계산용이 아니다.**
        die map 계산은 이미지를 절대 돌리지 않는다. 10000x10000 을 warp 하면
        1억 픽셀을 손실 있게 재샘플링하게 되고, 32bit 파이썬에서는 그 자체로
        위험하다. vx/vy 를 알고 있으니 기울어진 기저에 그대로 die 를 놓고
        좌표->인덱스는 2x2 역행렬로 푸는 게 더 정확하고 더 싸다.

        그래서 여기서는 **참조만** 들고 있다가 실제로 접근할 때 한 번 만든다.
        (source_image 는 사본이 아니라 호출자가 준 배열 그대로다.)

        실측 (32bit 파이썬, OpenCV 12스레드, 10000x10000x3):
            warpAffine 0.15 s,  결과 배열 286 MB
        die map 생성 자체가 3.9 s 이니 시간은 문제가 아니고, **메모리가**
        문제다. 안 건드리면 0 이고, 한 번 건드리면 286 MB 가 눌러앉는다.
        32bit 는 주소공간이 ~1.5 GiB 뿐이라 원본(286) + 정렬본(286) 이면
        절반이 넘는다. 필요 없으면 접근하지 말고, 다 썼으면 dm._aligned = None.

        원본을 안 줬으면 None 이다 (build_die_map_from_yolo 는 자동으로 준다).
        회전 뒤 좌표는 원본과 다르므로, 이 이미지 위에서 인덱스를 구하려면
        aligned_transform() 으로 좌표를 같이 옮겨야 한다.
        """
        if self._aligned is None and self.source_image is not None:
            M = self.aligned_transform()
            h, w = self.source_image.shape[:2]
            self._aligned = cv2.warpAffine(
                self.source_image, M, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self._aligned

    def aligned_transform(self) -> np.ndarray:
        """aligned_image 로 가는 2x3 아핀 행렬. 좌표도 같이 옮길 때 쓴다."""
        return cv2.getRotationMatrix2D(
            (float(self.wafer_cx), float(self.wafer_cy)), self.angle_deg, 1.0)

    # -- 격자 <-> 좌표 -------------------------------------------------
    def corner_xy(self, i: int, j: int) -> np.ndarray:
        """die(i,j) 의 좌상 꼭지점 = 십자점 좌표."""
        return self.origin + i * self.vx + j * self.vy

    def center_xy(self, i: int, j: int) -> np.ndarray:
        """die(i,j) 의 중심. 십자점은 모서리라 반 칸을 더한다."""
        return self.origin + (i + 0.5) * self.vx + (j + 0.5) * self.vy

    def index_of(self, x: float, y: float, snap: bool = True):
        """좌표 -> die 인덱스. V^-1 (q - origin) 를 내림한다.

        die(i,j) 는 [i, i+1) x [j, j+1) 칸이므로 **floor** 다.
        (center_xy 처럼 반 칸 더하는 게 아니다 - 그건 중심을 구할 때)
        """
        V = np.stack([self.vx, self.vy], axis=1).astype(np.float64)   # 열이 기저
        ij = np.linalg.solve(V, np.asarray([x, y], np.float64) - self.origin)
        if not snap:
            return ij
        return int(math.floor(ij[0])), int(math.floor(ij[1]))

    def get_die(self, i: int, j: int) -> Optional[Dict[str, Any]]:
        return self.dies_by_index.get((int(i), int(j)))

    def __repr__(self) -> str:
        return ("ViaDieMap(wafer=(%d,%d,r=%d) pitch=%.2f/%.2f angle=%+.4f "
                "dies=%d)" % (self.wafer_cx, self.wafer_cy, self.wafer_r,
                              self.pitch_x, self.pitch_y, self.angle_deg,
                              len(self.dies)))


# ====================================================================
# 잡동사니
# ====================================================================

def _load_bgr(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """경로/배열 -> BGR uint8 3채널. BGRA/회색조/float 을 조용히 안 깨뜨린다."""
    if isinstance(image, (str, Path)):
        p = str(image)
        buf = np.fromfile(p, dtype=np.uint8)        # 한글 경로 대응
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("이미지를 읽을 수 없다: %s" % p)
    else:
        img = np.asarray(image)

    if img.dtype != np.uint8:
        img = _to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return np.ascontiguousarray(img)
    raise ValueError("지원하지 않는 채널 수: %s" % (img.shape,))


def _to_uint8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < _EPS:
        return np.zeros(a.shape, np.uint8)
    if lo >= 0.0 and hi <= 1.0:
        a = a * 255.0
    elif lo < 0.0 or hi > 255.0:
        a = (a - lo) * (255.0 / (hi - lo))
    return np.clip(a, 0, 255).astype(np.uint8)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _downscale(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    s = float(max_dim) / float(m)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA), s


def _imwrite_unicode(path: str, img: np.ndarray) -> str:
    """한글 경로에도 저장되게. cv2.imwrite 는 비ASCII 경로에서 조용히 실패한다."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("imencode 실패: %s" % path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(path)
    return path


def _put_text(canvas: np.ndarray, text: str, org: Tuple[int, int],
              color: Tuple[int, int, int], scale: float = 0.5) -> None:
    """검은 외곽선 + 색 글씨. 배경색이 뭐든 읽힌다."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


# ====================================================================
# 1) 웨이퍼 외곽선 검출  (v6 에서 그대로 가져옴)
# ====================================================================

def _estimate_background(img_bgr: np.ndarray) -> Tuple[Tuple[int, int, int], str]:
    """이미지 4코너 블록에서 배경 BGR 을 robust 하게 추정.

    웨이퍼는 보통 화면 중앙의 원이므로 코너는 배경이다. 4코너 중 서로 가장
    비슷한 3개를 채택(한 코너에 라벨/노이즈가 있어도 견딤)해 median 을 쓴다.
    """
    h, w = img_bgr.shape[:2]
    bs = max(8, min(h, w) // 16)
    blocks = [
        img_bgr[0:bs, 0:bs],
        img_bgr[0:bs, w - bs:w],
        img_bgr[h - bs:h, 0:bs],
        img_bgr[h - bs:h, w - bs:w],
    ]
    meds = np.array([np.median(b.reshape(-1, 3), axis=0) for b in blocks],
                    dtype=np.float64)                      # (4,3)
    # 4개 중 나머지 3개와의 거리합이 가장 큰 1개를 이상치로 제거
    d = np.abs(meds[:, None, :] - meds[None, :, :]).sum(axis=2).sum(axis=1)
    keep = np.argsort(d)[:3]
    bg = np.median(meds[keep], axis=0)
    return (int(round(bg[0])), int(round(bg[1])), int(round(bg[2]))), "corners"


def _lab_dist(img_bgr: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
    """한 덩어리(전체 또는 가로 밴드)의 Lab 거리 float32 맵."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab -= bg_lab.reshape(1, 1, 3)
    lab *= lab
    return np.sqrt(lab.sum(axis=2))


def _bg_distance_map(img_bgr: np.ndarray,
                     bg_bgr: Tuple[int, int, int],
                     band_rows: int = 512) -> np.ndarray:
    """배경색으로부터의 Lab 거리 맵 (uint8 0..255 정규화).

    색-무관의 핵심: "검정보다 밝은가"가 아니라 "배경색과 다른가"를 본다.
    그래서 흰 배경/검은 배경/색 배경이 전부 같은 코드로 처리된다.

    왜 밴드로 나눠 도나 (실측 실패)
    ------------------------------------------------------------------
    예전에는 전체 이미지를 한 번에 float32 로 올렸다. 10000x10000 에서는

        MemoryError: Unable to allocate 1.12 GiB for an array
        with shape (10000, 10000, 3) and data type float32

    32bit 파이썬 주소공간(~1.5GB)에서는 이 한 줄로 끝난다.
    가로 전체 x band_rows 줄씩 끊어 돌면 중간 float 버퍼가 밴드 크기로
    줄어든다 (512줄이면 ~60MB). 결과는 완전히 동일하다.

    정규화 배율(99.5 퍼센타일)만 예외로 **다운스케일본**에서 구한다.
    전체 거리맵을 float 로 들고 있어야 퍼센타일을 정확히 낼 수 있는데,
    그게 바로 피하려던 할당이기 때문이다. 이 값은 단순 스케일 상수이고
    뒤따르는 Otsu 임계도 같은 스케일 위에서 계산되므로 결과가 안 바뀐다.
    """
    swatch = np.zeros((1, 1, 3), np.uint8)
    swatch[0, 0] = np.array(bg_bgr, dtype=np.uint8)
    bg_lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    H, W = img_bgr.shape[:2]
    if H * W <= band_rows * 4096:                 # 작은 이미지는 그냥 한 번에
        dist = _lab_dist(img_bgr, bg_lab)
        hi = float(np.percentile(dist, 99.5))
        if hi < _EPS:
            return np.zeros((H, W), np.uint8)
        return np.clip(dist * (255.0 / hi), 0, 255).astype(np.uint8)

    small, _ = _downscale(img_bgr, 2048)
    hi = float(np.percentile(_lab_dist(small, bg_lab), 99.5))
    if hi < _EPS:
        return np.zeros((H, W), np.uint8)
    g = 255.0 / hi
    out = np.empty((H, W), np.uint8)
    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        b = _lab_dist(img_bgr[y0:y1], bg_lab)
        b *= g
        np.clip(b, 0, 255, out=b)
        out[y0:y1] = b.astype(np.uint8)
    return out


def detect_wafer_adaptive(img_bgr: np.ndarray,
                          profile: Optional[WaferProfile] = None,
                          diag: Optional[DieMapDiag] = None
                          ) -> Tuple[int, int, int, np.ndarray]:
    """색-무관 wafer 외곽선 검출 -> (cx, cy, r, silhouette uint8 0/1).

    절차
    ----
    1) 배경색 자동 추정 (또는 profile.background_bgr).
    2) 배경색과의 Lab 거리 맵 -> Otsu (임계는 다운스케일본에서 계산).
    3) morphology open -> 점진적 close -> 가장 큰 연결성분.
    4) 외곽 컨투어를 채워 실루엣 확정.
       RETR_EXTERNAL + drawContours(-1) 이므로 내부 구멍(어두운 die)은 메워지고
       notch 같은 '경계 오목부'는 그대로 보존된다.
    5) minEnclosingCircle. 면적환산 반지름과 크게 어긋나면(림 돌출 노이즈)
       면적환산 반지름 + 무게중심으로 대체.
    6) coverage 가 비정상(~1 또는 ~0)이면 내접원 fallback.
    """
    prof = profile or WaferProfile()
    img_bgr = _load_bgr(img_bgr)
    H, W = img_bgr.shape[:2]

    if prof.background_bgr is not None:
        bg = (int(prof.background_bgr[0]), int(prof.background_bgr[1]),
              int(prof.background_bgr[2]))
        src = "user"
    else:
        bg, src = _estimate_background(img_bgr)
    if diag is not None:
        diag.background_bgr = bg
        diag.background_source = src

    dist = _bg_distance_map(img_bgr, bg)

    # Otsu 임계는 다운스케일본에서(속도), 적용은 원본 해상도에서(정밀도)
    small, _ = _downscale(dist, prof.wafer_otsu_max_dim)
    thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(dist, float(thr), 255, cv2.THRESH_BINARY)

    ko = _odd(max(3, int(round(prof.wafer_open_ksize * min(H, W) / 3000.0)) or 3))
    kc = _odd(max(5, int(round(prof.wafer_close_ksize * min(H, W) / 3000.0)) or 5))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))

    # --- 점진적 CLOSE ---------------------------------------------------
    # 고정 커널은 scribe lane 이 넓으면(예: pitch 150 / lane 13px) lane 을
    # 못 메워서 마스크가 die 단위로 산산조각 난다. 그러면 RETR_EXTERNAL 의
    # '가장 큰 컨투어'가 die 하나가 되어 coverage~0 오판이 난다.
    # -> 최대 컨투어가 전경 면적의 대부분을 차지할 때까지 커널을 키운다.
    fg = float(np.count_nonzero(mask))
    k_max = _odd(max(kc, int(round(0.04 * min(H, W)))))
    k = _odd(kc)
    closed = mask
    while True:
        closed = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs or fg < _EPS:
            break
        if cv2.contourArea(max(cs, key=cv2.contourArea)) >= 0.85 * fg:
            break
        if k >= k_max:
            break
        k = _odd(min(k_max, int(round(k * 1.8)) + 1))
    mask = closed

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback = ""
    if not cnts:
        fallback = "no-contour"
    else:
        big = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(big))
        cov = area / float(H * W)
        sil = np.zeros((H, W), np.uint8)
        cv2.drawContours(sil, [big], -1, 1, thickness=-1)
        (ecx, ecy), er = cv2.minEnclosingCircle(big)
        r_area = math.sqrt(max(area, 1.0) / math.pi)
        if er > 1.10 * r_area:
            # 림 밖으로 삐져나온 노이즈가 원을 부풀림 -> 면적환산 + 무게중심
            m = cv2.moments(big)
            if abs(m["m00"]) > _EPS:
                ecx = m["m10"] / m["m00"]
                ecy = m["m01"] / m["m00"]
            er = r_area
            if diag is not None:
                diag.warn("minEnclosingCircle 이 림 노이즈로 부풀어 "
                          "면적환산 반지름으로 대체했다")
        if cov > 0.985:
            fallback = "coverage~1 (배경색 추정 실패?)"
        elif cov < 0.02:
            fallback = "coverage~0 (웨이퍼가 배경과 안 갈린다)"
        else:
            if diag is not None:
                diag.wafer_coverage = cov
                diag.wafer_circle = (int(round(ecx)), int(round(ecy)),
                                     int(round(er)))
                diag.wafer_contour = big
            return (int(round(ecx)), int(round(ecy)), int(round(er)), sil)

    # ---- fallback : 이미지 내접원 ---------------------------------------
    cx, cy = W // 2, H // 2
    r = int(min(H, W) // 2) - 1
    sil = np.zeros((H, W), np.uint8)
    cv2.circle(sil, (cx, cy), r, 1, thickness=-1)
    if diag is not None:
        diag.wafer_fallback = fallback
        diag.wafer_coverage = math.pi * r * r / float(H * W)
        diag.wafer_circle = (cx, cy, r)
        diag.warn("웨이퍼 검출 fallback: %s" % fallback)
    return cx, cy, r, sil


def clean_wafer_outside(img_bgr: np.ndarray,
                        wafer_cx: int, wafer_cy: int, wafer_r: int,
                        sil: np.ndarray,
                        fill_bgr: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """wafer 원판 밖(실루엣 밖 OR 원 밖)을 fill_bgr 로 채워 반환.

    v5 는 항상 검정으로 채웠지만, 배경이 흰색인 이미지에서 검정으로 채우면
    없던 강한 경계가 생겨 이후 처리를 방해한다. 여기서는 **추정된 배경색**으로
    채워 색상 팔레트에 중립적으로 동작한다.
    """
    H, W = img_bgr.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    disc = (xx - wafer_cx) ** 2 + (yy - wafer_cy) ** 2 <= wafer_r * wafer_r
    keep = (sil > 0) & disc
    out = img_bgr.copy()
    out[~keep] = np.array(fill_bgr, dtype=img_bgr.dtype)
    return out


# ====================================================================
# 2) die 순회 보조
# ====================================================================

def _crop_rect(cx: float, cy: float, die_w: int, die_h: int,
               offset_x: int, offset_y: int,
               margin_x: int, margin_y: int) -> Tuple[int, int, int, int]:
    """die 중심 -> crop 사각형. offset 은 위치 보정, margin 은 영역 확장."""
    x_a = int(round(cx)) - die_w // 2 + int(offset_x) - int(margin_x)
    y_a = int(round(cy)) - die_h // 2 + int(offset_y) - int(margin_y)
    x_b = x_a + die_w + 2 * int(margin_x)
    y_b = y_a + die_h + 2 * int(margin_y)
    return x_a, y_a, x_b, y_b


def _rect_crosses_circle(x1: int, y1: int, x2: int, y2: int,
                         cx: int, cy: int, r: int) -> bool:
    """사각형이 원 경계를 물고 있나 = 네 꼭지점 중 일부만 원 안인가."""
    ins = [(x - cx) ** 2 + (y - cy) ** 2 <= r * r
           for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    return any(ins) and not all(ins)


def _normalize_edge_mode(edge_mode: str) -> str:
    m = str(edge_mode).strip().lower()
    if m not in ("circle", "ring", "both"):
        raise ValueError("edge_mode 는 'circle'|'ring'|'both' 중 하나 (받은 값: %r)"
                         % (edge_mode,))
    return m


def _resolve_edge_flag(is_partial: bool, is_ring: bool, edge_mode: str) -> bool:
    """is_edge 의 의미를 고른다.

    circle : 원 경계를 물었나 (물리적으로 잘린 die)
    ring   : 격자에서 이웃이 빈 자리가 있나 (최외곽 링)
    both   : 둘 중 하나라도
    """
    if edge_mode == "circle":
        return bool(is_partial)
    if edge_mode == "ring":
        return bool(is_ring)
    return bool(is_partial or is_ring)


def validate_quadrant_edges(dies: List[Dict[str, Any]],
                            cx: int, cy: int, r: int) -> Dict[str, Any]:
    """4분면 균형 검사.

    격자 원점이나 pitch 가 틀리면 die 가 한쪽으로 쏠린다. 4분면별 die 수를
    이론 면적비와 비교해 쏠림을 잡아낸다. 정상이면 네 값이 비슷하다.
    """
    q = [0, 0, 0, 0]
    for d in dies:
        x, y = d["center_px"]
        q[(0 if x >= cx else 1) + (0 if y < cy else 2)] += 1
    tot = sum(q)
    if tot == 0:
        return {"counts": q, "coverage": [0.0] * 4, "balanced": False,
                "coverage_spread": 0.0, "min_coverage": 0.0}
    cov = [4.0 * v / tot for v in q]          # 균형이면 각 1.0
    spread = max(cov) - min(cov)
    return {"counts": q, "coverage": [round(c, 4) for c in cov],
            "balanced": bool(spread <= 0.25 and min(cov) >= 0.6),
            "coverage_spread": round(spread, 4),
            "min_coverage": round(min(cov), 4)}


def clip_die(image: np.ndarray, center_x: int, center_y: int,
             die_w: int, die_h: int) -> Optional[np.ndarray]:
    """die 영역을 잘라낸다. 이미지 밖으로 나가면 None (패딩 안 함)."""
    H, W = image.shape[:2]
    x_a = int(center_x) - int(die_w) // 2
    y_a = int(center_y) - int(die_h) // 2
    x_b, y_b = x_a + int(die_w), y_a + int(die_h)
    if x_a < 0 or y_a < 0 or x_b > W or y_b > H:
        return None
    return image[y_a:y_b, x_a:x_b].copy()


def crop_die(image: np.ndarray, center_x: float, center_y: float,
             die_w: int, die_h: int, *,
             offset_x: int = DEFAULT_OFFSET_X,
             offset_y: int = DEFAULT_OFFSET_Y,
             margin_x: int = DEFAULT_MARGIN_X,
             margin_y: int = DEFAULT_MARGIN_Y,
             border_mode: str = "pad",
             fill_bgr: Tuple[int, int, int] = (0, 0, 0)) -> Optional[np.ndarray]:
    """offset/margin 을 적용해 die 를 잘라낸다.

    border_mode : "pad"  -> 밖으로 나간 부분을 fill_bgr 로 채워 항상 같은 크기
                  "crop" -> 밖으로 나가면 None
    """
    H, W = image.shape[:2]
    x_a, y_a, x_b, y_b = _crop_rect(center_x, center_y, die_w, die_h,
                                    offset_x, offset_y, margin_x, margin_y)
    if x_a >= 0 and y_a >= 0 and x_b <= W and y_b <= H:
        return image[y_a:y_b, x_a:x_b].copy()
    if border_mode == "crop":
        return None
    out = np.empty((y_b - y_a, x_b - x_a) + image.shape[2:], dtype=image.dtype)
    out[...] = np.array(fill_bgr, dtype=image.dtype) if image.ndim == 3 else 0
    sx_a, sy_a = max(0, x_a), max(0, y_a)
    sx_b, sy_b = min(W, x_b), min(H, y_b)
    if sx_b > sx_a and sy_b > sy_a:
        out[sy_a - y_a: sy_b - y_a, sx_a - x_a: sx_b - x_a] = \
            image[sy_a:sy_b, sx_a:sx_b]
    return out


def crop_die_rotated(image: np.ndarray, dm: "ViaDieMap", i: int, j: int,
                     *, margin: int = 0,
                     fill_bgr: Tuple[int, int, int] = (0, 0, 0)
                     ) -> np.ndarray:
    """die(i,j) 를 **격자 기울기를 되돌려서** 축정렬로 잘라낸다.

    왜 필요한가: 우리는 이미지를 회전시키지 않는다. 그래서 격자가 기울어 있으면
    축정렬 사각형(rect_px)은 die 를 정확히 담지 못한다. 8도 기울면 die 를 다
    담으려는 축정렬 상자가 한 변에서 cos8+sin8 = 1.13 배, 즉 13% 커진다.
    이 함수는 die 하나만 warpAffine 하므로 전체 이미지를 워핑하는 것보다
    (10000x10000 = 1억 픽셀) 훨씬 싸고, 나머지 픽셀은 원본으로 남는다.
    """
    cx, cy = dm.center_xy(int(i), int(j))
    w = dm.die_w + 2 * int(margin)
    h = dm.die_h + 2 * int(margin)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), dm.angle_deg, 1.0)
    # 회전 후 die 중심이 출력의 중앙에 오도록 평행이동을 더한다
    M[0, 2] += w / 2.0 - float(cx)
    M[1, 2] += h / 2.0 - float(cy)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill_bgr)


# ====================================================================
# 3) die map 생성  --  웨이퍼 외곽선 + 클립 격자
# ====================================================================

def build_die_map_via(image: Union[str, Path, np.ndarray],
                      grid,
                      *,
                      clip_origin: Optional[Tuple[float, float]] = None,
                      clip_size: int = 512,
                      profile: Optional[WaferProfile] = None,
                      pixel_per_unit: int = DEFAULT_PIXEL_PER_UNIT,
                      include_edge: bool = True,
                      edge_margin: float = DEFAULT_EDGE_MARGIN,
                      edge_mode: str = DEFAULT_EDGE_MODE,
                      with_crops: bool = False,
                      border_mode: str = "pad",
                      offset_x: int = DEFAULT_OFFSET_X,
                      offset_y: int = DEFAULT_OFFSET_Y,
                      margin_x: int = DEFAULT_MARGIN_X,
                      margin_y: int = DEFAULT_MARGIN_Y,
                      keep_source_image: bool = False,
                      ) -> ViaDieMap:
    """웨이퍼 이미지 + 클립에서 뽑은 격자 -> die map.

    Parameters
    ----------
    image       : 웨이퍼 이미지 (경로 또는 BGR ndarray). 10000x10000 을 가정하지만
                  크기는 아무래도 된다 - 외곽선을 직접 검출한다.
    grid        : `analyze_clip` 이 준 GridFit. **좌표가 클립 기준**이므로
                  아래 clip_origin 으로 웨이퍼 기준으로 옮긴다.
    clip_origin : 클립의 (0,0) 픽셀이 웨이퍼 이미지의 어디인가.
                  None 이면 "이미지 정중앙 clip_size 정사각형"으로 계산한다
                  (사용자 요구: "이미지 센터 Clip(512x512)").
    clip_size   : clip_origin 이 None 일 때 쓰는 클립 한 변 길이.
    include_edge: True 면 원 안 die 전부 포함(잘린 die 포함)
    edge_margin : die 포함 기준 = (중심거리 <= r * edge_margin)
    edge_mode   : is_edge 의 정의 "circle"(기본) | "ring" | "both"
    with_crops  : True 면 각 die 에 "image" (축정렬 crop) 을 넣는다.
                  격자가 많이 기울었으면 crop_die_rotated 를 쓰는 게 맞다.
    keep_source_image : True 면 dm.aligned_image 를 쓸 수 있게 원본 **참조**를
                  들고 있는다. 복사가 아니라 참조라 추가 메모리는 없지만,
                  aligned_image 에 실제로 접근하면 그때 회전본이 만들어진다
                  (10000x10000 이면 300 MB / 수 초. ViaDieMap.aligned_image 참고).

    Returns
    -------
    ViaDieMap
    """
    t0 = time.time()
    if not getattr(grid, "ok", False):
        raise ValueError("격자가 세워지지 않았다: %s"
                         % getattr(grid, "reason", "(사유 없음)"))

    diag = DieMapDiag()
    prof = profile or WaferProfile()
    img = _load_bgr(image)
    H, W = img.shape[:2]

    # --- 1) 웨이퍼 외곽선 -------------------------------------------------
    wcx, wcy, wr, sil = detect_wafer_adaptive(img, prof, diag)

    # --- 2) 클립 좌표 -> 웨이퍼 좌표 --------------------------------------
    # 클립은 웨이퍼 이미지에서 잘라낸 조각이므로, 격자 원점에 그 오프셋만
    # 더하면 된다. 회전/스케일은 없다(단순 crop 이므로).
    if clip_origin is None:
        cs = int(clip_size)
        if cs > min(H, W):
            raise ValueError("clip_size %d 가 이미지 %dx%d 보다 크다" % (cs, W, H))
        clip_origin = ((W - cs) // 2, (H - cs) // 2)
    off = np.asarray(clip_origin, dtype=np.float64)

    origin = np.asarray(grid.origin, dtype=np.float64) + off
    vx = np.asarray(grid.vx, dtype=np.float64)
    vy = np.asarray(grid.vy, dtype=np.float64)
    px, py = float(grid.pitch_x), float(grid.pitch_y)
    die_w, die_h = int(round(px)), int(round(py))
    if die_w < 2 or die_h < 2:
        raise RuntimeError("die 크기가 말이 안 된다: %dx%d" % (die_w, die_h))

    # --- 3) 웨이퍼를 덮는 인덱스 범위 -------------------------------------
    # 웨이퍼 중심을 격자 좌표로 옮기면 몇 칸 떨어져 있는지 바로 나온다.
    # vx ⊥ vy 이므로 i 방향으로 r/pitch_x 칸, j 방향으로 r/pitch_y 칸이면 충분.
    # (원점이 웨이퍼 중심에서 멀어도 이 식은 그대로 맞는다)
    V = np.stack([vx, vy], axis=1)
    ij_c = np.linalg.solve(V, np.asarray([wcx, wcy], np.float64) - origin)
    ri, rj = wr / px, wr / py
    i0, i1 = int(math.floor(ij_c[0] - ri)) - 2, int(math.ceil(ij_c[0] + ri)) + 2
    j0, j1 = int(math.floor(ij_c[1] - rj)) - 2, int(math.ceil(ij_c[1] + rj)) + 2

    # --- 4) die 순회 -----------------------------------------------------
    margin = edge_margin if include_edge else 0.98
    r_lim_sq = (wr * margin) ** 2

    ii, jj = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.float64),
                         np.arange(j0, j1 + 1, dtype=np.float64),
                         indexing="ij")
    # die 중심 = origin + (i+0.5)vx + (j+0.5)vy   (십자점은 모서리라 반 칸)
    cxs = origin[0] + (ii + 0.5) * vx[0] + (jj + 0.5) * vy[0]
    cys = origin[1] + (ii + 0.5) * vx[1] + (jj + 0.5) * vy[1]
    inside = ((cxs - wcx) ** 2 + (cys - wcy) ** 2) <= r_lim_sq

    dies: List[Dict[str, Any]] = []
    dies_by_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    ci = np.cos(math.radians(grid.angle_deg))
    si = np.sin(math.radians(grid.angle_deg))

    for a, b in zip(*np.nonzero(inside)):
        i = int(ii[a, b])
        j = int(jj[a, b])
        cx_d = int(round(float(cxs[a, b])))
        cy_d = int(round(float(cys[a, b])))

        x_a = cx_d - die_w // 2
        y_a = cy_d - die_h // 2
        x_b = x_a + die_w
        y_b = y_a + die_h

        entry: Dict[str, Any] = {
            "index": (i, j),
            "center_px": (cx_d, cy_d),
            # rect_px 는 **축정렬 근사**다. 격자가 기울면 실제 die 와 어긋난다.
            # 정확한 네 꼭지점은 quad_px, 정확한 crop 은 crop_die_rotated.
            "rect_px": (x_a, y_a, x_b, y_b),
            "quad_px": (
                tuple(np.round(origin + i * vx + j * vy).astype(int)),
                tuple(np.round(origin + (i + 1) * vx + j * vy).astype(int)),
                tuple(np.round(origin + (i + 1) * vx + (j + 1) * vy).astype(int)),
                tuple(np.round(origin + i * vx + (j + 1) * vy).astype(int)),
            ),
            "crop_rect_px": _crop_rect(cx_d, cy_d, die_w, die_h,
                                       offset_x, offset_y, margin_x, margin_y),
            # 물리 좌표는 v5 처럼 y 가 **위로** 양수다 (격자 j 와 방향이 반대)
            "real_coord": ((cx_d - wcx) / pixel_per_unit,
                           (wcy - cy_d) / pixel_per_unit),
            "is_edge_partial": _rect_crosses_circle(x_a, y_a, x_b, y_b,
                                                    wcx, wcy, wr),
            "is_edge_ring": False,
            "is_edge": False,
        }
        if with_crops:
            crop = crop_die(img, cx_d, cy_d, die_w, die_h,
                            offset_x=offset_x, offset_y=offset_y,
                            margin_x=margin_x, margin_y=margin_y,
                            border_mode=border_mode)
            if crop is None:
                continue
            entry["image"] = crop
        dies.append(entry)
        dies_by_index[(i, j)] = entry

    # --- 5) edge 판정 (링은 이웃 존재 여부라 순회가 끝나야 알 수 있다) ------
    emode = _normalize_edge_mode(edge_mode)
    present = set(dies_by_index.keys())
    for d in dies:
        i, j = d["index"]
        ring = any((i + di, j + dj) not in present
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if not (di == 0 and dj == 0))
        d["is_edge_ring"] = bool(ring)
        d["is_edge"] = _resolve_edge_flag(d["is_edge_partial"], ring, emode)

    qrep = validate_quadrant_edges(dies, wcx, wcy, wr)
    if not qrep["balanced"]:
        diag.warn("4분면 분포가 치우쳤다 (spread=%s, min=%s) "
                  "- 격자 원점이나 pitch 를 의심해야 한다"
                  % (qrep["coverage_spread"], qrep["min_coverage"]))

    diag.n_dies = len(dies)
    diag.elapsed_sec = time.time() - t0
    if not dies:
        diag.warn("die 가 하나도 안 나왔다 - 격자가 거의 확실히 틀렸다")

    return ViaDieMap(
        wafer_cx=wcx, wafer_cy=wcy, wafer_r=wr,
        origin=origin, vx=vx, vy=vy,
        pitch_x=px, pitch_y=py, angle_deg=float(grid.angle_deg),
        die_w=die_w, die_h=die_h, pixel_per_unit=pixel_per_unit,
        dies=dies, dies_by_index=dies_by_index, image_shape=(H, W),
        wafer_mask=sil, edge_mode=emode, quadrant_report=qrep,
        diagnostics=diag,
        angle_sigma_deg=float(getattr(grid, "angle_sigma_deg", float("nan"))),
        angle_confidence=float(getattr(grid, "angle_confidence", 0.0)),
        wafer_contour=diag.wafer_contour,
        source_image=(img if keep_source_image else None),
    )


# ====================================================================
# 5-b) YOLO 결과에서 바로 die map  (통짜 진입점)
# ====================================================================

def detections_to_points(detections: Any,
                         detection_format: str = "xywh") -> np.ndarray:
    """
    YOLO 검출 배열 -> (N,2) 중심점.

        detections = results[0].boxes.xywh.cpu().numpy()   # (N,4) float

    detection_format
        "xywh"  : [center_x, center_y, w, h]   <- ultralytics boxes.xywh
        "xyxy"  : [x1, y1, x2, y2]             <- ultralytics boxes.xyxy
        "xy"    : [x, y]                       <- 이미 중심점만 있는 경우

    conf/cls 가 뒤에 더 붙어 있어도(예: (N,6)) 앞의 4열만 쓴다.
    torch.Tensor 를 그냥 넘겨도 __array__ 로 받아진다.

    **정규화 좌표는 거부한다.** boxes.xywhn 을 실수로 넣으면 값이 전부 0~1 이라
    격자가 소수점 크기로 잡히고, 조용히 완전히 틀린 die map 이 나온다.
    조용히 틀리느니 여기서 멈추는 게 낫다.
    """
    fmt = str(detection_format).lower().replace("_", "")
    d = np.asarray(detections, dtype=np.float64)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.ndim != 2 or d.shape[0] == 0:
        raise ValueError("detections 모양이 이상하다: %s" % (d.shape,))

    if fmt in ("xywh", "cxcywh"):
        if d.shape[1] < 4:
            raise ValueError("xywh 인데 열이 %d개뿐이다" % d.shape[1])
        pts = d[:, :2].copy()
    elif fmt == "xyxy":
        if d.shape[1] < 4:
            raise ValueError("xyxy 인데 열이 %d개뿐이다" % d.shape[1])
        pts = np.column_stack([(d[:, 0] + d[:, 2]) * 0.5,
                               (d[:, 1] + d[:, 3]) * 0.5])
    elif fmt in ("xy", "points", "point"):
        if d.shape[1] < 2:
            raise ValueError("xy 인데 열이 %d개뿐이다" % d.shape[1])
        pts = d[:, :2].copy()
    else:
        raise ValueError("모르는 detection_format: %r "
                         "(xywh / xyxy / xy 중 하나)" % detection_format)

    if len(pts) >= 2 and float(np.nanmax(np.abs(pts))) <= 1.5:
        raise ValueError(
            "좌표가 전부 0~1 이다 - 정규화된 값(boxes.xywhn)을 넣은 것 같다. "
            "512x512 클립 기준의 픽셀 좌표(boxes.xywh)를 넣어야 한다")
    if not np.isfinite(pts).all():
        raise ValueError("detections 에 nan/inf 가 있다")
    return pts


def build_die_map_from_yolo(wafer_image: Union[str, Path, np.ndarray],
                            clip_image: Union[str, Path, np.ndarray],
                            detections: Any,
                            *,
                            detection_format: str = "xywh",
                            refine: bool = True,
                            refine_mode: str = "auto",
                            refine_radius: int = 24,
                            refine_noise_kernel: int = 5,
                            refine_min_confidence: float = 0.15,
                            clip_origin: Optional[Tuple[float, float]] = None,
                            keep_source_image: bool = True,
                            **kwargs: Any) -> ViaDieMap:
    """
    YOLO 결과 -> 웨이퍼 전체 die map. **이거 하나만 부르면 된다.**

        dm = build_die_map_from_yolo(
            wafer_image=wafer_bgr,
            clip_image=center_clip_bgr,
            detections=results[0].boxes.xywh.cpu().numpy(),
            detection_format="xywh",
            refine=True,
            refine_mode="auto",
            refine_radius=24,
            refine_noise_kernel=5,
            refine_min_confidence=0.15,
        )

        print(dm.x0, dm.y0)          # 전체 wafer 좌표의 center corner
        print(dm.pitch_x, dm.pitch_y)
        print(dm.grid_angle_deg)
        print(dm.angle_confidence)
        print(dm.num_dies)
        print(dm.dies)               # [{index, center_px, rect_px, quad_px,
                                     #   crop_rect_px, real_coord, is_edge...}, ...]
        print(dm.dies_by_index)      # {(i,j): 위 dict 와 **같은 객체**}
        print(dm.wafer_boundary)     # WaferBoundary(cx, cy, r, contour)
        print(dm.aligned_image)      # 회전 보정본 (처음 볼 때 만든다)

    Parameters
    ----------
    wafer_image : 웨이퍼 전체 이미지 (10000x10000 가정, 크기는 자유).
                  외곽선을 여기서 직접 검출한다.
    clip_image  : YOLO 에 넣은 그 클립 (512x512). **웨이퍼 이미지의 정중앙에서
                  잘라낸 것**이라고 가정한다. 아니면 clip_origin 을 준다.
    detections  : (N,4) xywh 배열. detections_to_points 참고.

    refine      : 서브픽셀 보정 on/off. 끄면 실측상 각도 오차가 56배 나빠진다.
    refine_mode : "auto"  보정 윈도우를 pitch 에서 자동으로 정한다 (권장).
                          이때 refine_radius 는 **쓰이지 않는다** - pitch*0.30
                          이 실측상 더 좋았기 때문이다.
                  "fixed" refine_radius 를 그대로 윈도우 반경으로 쓴다.
                  "off"   refine=False 와 같다.
    refine_radius : "fixed" 일 때의 윈도우 반경 (px).
    refine_noise_kernel : 보정 전에 클립에 걸 medianBlur 커널 (홀수, 0=안 걸음).
                  점잡음("흰색+갈색 노이즈")용이다. streetness 의 bg_ksize
                  (≈pitch) 와 혼동하면 안 된다 - 그건 별도로 자동 결정된다.
    refine_min_confidence : 이 신뢰도 미만인 점은 격자 맞춤에서 뺀다.
                  실측 분포는 진짜 점 p10 = 0.469, 오검출 p90 = 0.003 이라
                  0.15 는 둘 사이 빈 구간에 있다.
    clip_origin : 클립의 (0,0) 이 웨이퍼 이미지의 어디인가. None 이면 정중앙.
    keep_source_image : dm.aligned_image 를 쓰려면 True 여야 한다 (기본 True).
                  원본 **참조**만 들고 있으므로 추가 메모리는 없다.
    **kwargs    : build_die_map_via 로 그대로 넘어간다
                  (include_edge, edge_margin, edge_mode, profile, with_crops ...)

    Raises
    ------
    ValueError : 격자를 못 세웠을 때. 조용히 틀린 die map 을 주지 않는다.
    """

    wafer = _load_bgr(wafer_image)
    clip = _load_bgr(clip_image)
    H, W = wafer.shape[:2]
    ch, cw = clip.shape[:2]
    if ch > H or cw > W:
        raise ValueError("클립 %dx%d 이 웨이퍼 %dx%d 보다 크다" % (cw, ch, W, H))

    pts = detections_to_points(detections, detection_format)

    mode = str(refine_mode).lower()
    if mode not in ("auto", "fixed", "off"):
        raise ValueError("모르는 refine_mode: %r (auto/fixed/off)" % refine_mode)
    do_refine = bool(refine) and mode != "off"
    win = int(refine_radius) if mode == "fixed" else None

    res, g = analyze_clip(clip, pts,
                          conf_min=float(refine_min_confidence),
                          refine=do_refine, win=win,
                          noise_kernel=int(refine_noise_kernel))
    if not g.ok:
        raise ValueError("클립에서 격자를 못 세웠다: %s "
                         "(검출 %d개 중 신뢰 %d개)"
                         % (g.reason, len(pts),
                            int((res.confidence >= refine_min_confidence).sum())))

    # 클립은 웨이퍼의 정중앙 조각이라고 가정한다 (사용자 요구: "이미지 센터 Clip").
    if clip_origin is None:
        clip_origin = ((W - cw) // 2, (H - ch) // 2)

    dm = build_die_map_via(wafer, g, clip_origin=clip_origin,
                           keep_source_image=keep_source_image, **kwargs)

    dm.diagnostics.warnings.extend(
        w for w in (
            ("YOLO 검출 %d개 중 %d개만 격자에 썼다" % (len(pts), g.n_used))
            if g.n_used < len(pts) else "",
            g.reason,                      # ok=True 인데 사유가 있으면 경고다
            ("회전각 신뢰도가 낮다 (%.3f, sigma %.4f deg) "
             "- 웨이퍼 끝에서 %.0f px 밀릴 수 있다"
             % (dm.angle_confidence, dm.angle_sigma_deg,
                dm.wafer_r * math.tan(math.radians(dm.angle_sigma_deg * 3.0))))
            if dm.angle_confidence < 0.5 and np.isfinite(dm.angle_sigma_deg) else "",
        ) if w)
    return dm


# ====================================================================
# 6) 좌표 -> die 조회  (사용자 요구 "좌표 넣으면 die index")
# ====================================================================

def locate_die_via(dm: ViaDieMap,
                   point: Optional[Tuple[float, float]] = None,
                   bbox: Optional[Tuple[float, float, float, float]] = None,
                   *,
                   offset_x: int = DEFAULT_OFFSET_X,
                   offset_y: int = DEFAULT_OFFSET_Y,
                   margin_x: int = DEFAULT_MARGIN_X,
                   margin_y: int = DEFAULT_MARGIN_Y) -> Dict[str, Any]:
    """픽셀 좌표(또는 BBox 중심)가 몇 번 die 인지 알려준다.

    반환 키는 v5 `locate_die` / v6 `locate_die_v6` 와 같게 맞췄다.
    (기존 호출부를 그대로 쓸 수 있도록. 바뀐 건 아래 두 가지뿐이다)

    v6 와 다른 점
    ----------------------------------------------------------------
    1) 인덱스를 `floor((qx-x0)/pitch)` 로 구하지 않는다.
       v6 는 이미지를 통째로 회전시켜 격자를 축정렬로 만든 뒤에야
       그 식이 성립했다. 여기서는 이미지를 안 돌리므로
       기울어진 기저 V=[vx|vy] 를 그대로 역행렬로 푼다.
           ij = V^-1 (q - origin)   ->  floor
       회전이 0 이면 v6 식과 정확히 같아진다.
    2) `die_index` 의 j 는 **아래로 갈수록 커진다** (v5/v6 의 iy 는 반대).
       화면 좌표계와 부호를 맞춰 헷갈리지 않게 하려는 것이다.
       물리 좌표 `real_coord` 는 v5 처럼 y 가 위로 양수인 그대로다.

    격자 밖 / 웨이퍼 밖 좌표를 넣어도 예외를 던지지 않는다.
    격자는 무한히 뻗어 있으므로 인덱스는 항상 계산되고,
    `in_wafer` 와 die_map 등록 여부(`in_map`)로 판단하면 된다.
    """
    if (point is None) == (bbox is None):
        raise ValueError("point 또는 bbox 중 정확히 하나를 지정하세요.")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        qx = (float(x1) + float(x2)) / 2.0
        qy = (float(y1) + float(y2)) / 2.0
        input_type = "bbox"
    else:
        qx, qy = float(point[0]), float(point[1])
        input_type = "point"

    i, j = dm.index_of(qx, qy)                 # 기울기까지 반영된 인덱스
    ctr = dm.center_xy(i, j)
    cx_d = int(round(float(ctr[0])))
    cy_d = int(round(float(ctr[1])))

    x_a = cx_d - dm.die_w // 2
    y_a = cy_d - dm.die_h // 2
    x_b = x_a + dm.die_w
    y_b = y_a + dm.die_h
    crop_rect = _crop_rect(cx_d, cy_d, dm.die_w, dm.die_h,
                           offset_x, offset_y, margin_x, margin_y)

    ppu = dm.pixel_per_unit
    rx = (qx - dm.wafer_cx) / ppu
    ry = (dm.wafer_cy - qy) / ppu
    drx = (cx_d - dm.wafer_cx) / ppu
    dry = (dm.wafer_cy - cy_d) / ppu

    emode = _normalize_edge_mode(dm.edge_mode)
    entry = dm.get_die(i, j)
    if entry is not None:
        # die map 이 이미 판정해 둔 값을 그대로 쓴다 (같은 답이 두 벌 생기면 안 된다)
        is_edge_partial = bool(entry.get("is_edge_partial", False))
        is_edge_ring = bool(entry.get("is_edge_ring", False))
    else:
        # 등록 안 된 칸(웨이퍼 밖 등). 그래도 답은 준다.
        is_edge_partial = _rect_crosses_circle(x_a, y_a, x_b, y_b,
                                               dm.wafer_cx, dm.wafer_cy, dm.wafer_r)
        is_edge_ring = any((i + di, j + dj) not in dm.dies_by_index
                           for di in (-1, 0, 1) for dj in (-1, 0, 1)
                           if not (di == 0 and dj == 0))
    is_edge = _resolve_edge_flag(is_edge_partial, is_edge_ring, emode)
    in_wafer = ((qx - dm.wafer_cx) ** 2 + (qy - dm.wafer_cy) ** 2
                <= dm.wafer_r ** 2)

    return {
        "input_type": input_type,
        "query_px": (qx, qy),
        "die_index": (i, j),
        "die_center_px": (cx_d, cy_d),
        # 축정렬 근사. 기울었을 때 정확한 건 die_quad_px 다.
        "die_rect_px": (x_a, y_a, x_b, y_b),
        "die_quad_px": (
            tuple(np.round(dm.corner_xy(i, j)).astype(int)),
            tuple(np.round(dm.corner_xy(i + 1, j)).astype(int)),
            tuple(np.round(dm.corner_xy(i + 1, j + 1)).astype(int)),
            tuple(np.round(dm.corner_xy(i, j + 1)).astype(int)),
        ),
        "crop_rect_px": crop_rect,
        "real_coord": (rx, ry),
        "real_distance": math.hypot(rx, ry),
        "die_real_coord": (drx, dry),
        "wafer_center_px": (dm.wafer_cx, dm.wafer_cy),
        "corner_px": (float(dm.origin[0]), float(dm.origin[1])),
        "is_edge": is_edge,
        "is_edge_partial": is_edge_partial,
        "is_edge_ring": is_edge_ring,
        "edge_mode": emode,
        "in_wafer": bool(in_wafer),
        "in_map": entry is not None,
    }


# ====================================================================
# 7) 오버레이 (눈으로 확인하는 검증)
# ====================================================================

def save_debug_overlay(image: Union[str, Path, np.ndarray],
                       dm: ViaDieMap,
                       path: str,
                       *,
                       max_dim: int = 2000,
                       draw_grid: bool = True,
                       draw_dies: bool = True,
                       draw_panel: bool = True) -> str:
    """웨이퍼 이미지 위에 판정 결과를 그려서 저장한다.

    v6 와 달리 `dm` 이 이미지를 들고 있지 않다 (10000x10000 을 복사해 들고
    다니면 안 되므로). 그래서 이미지를 인자로 받는다.
    **build_die_map_via 에 넣은 것과 같은 이미지**여야 좌표가 맞는다.

    색상 규약
    --------
    노랑    웨이퍼 원 + 중심 십자
    주황    격자선 (기울어진 그대로 그린다 - 이미지를 안 돌리므로)
    자홍 X  격자 원점 origin = die(0,0) 의 좌상 십자점
    초록    내부 die
    청록    edge die

    큰 이미지는 max_dim 으로 줄여서 그린다. 줄인 뒤에 좌표를 곱하므로
    (그린 뒤에 줄이는 게 아니라) 1억 픽셀 캔버스를 만들지 않는다.
    """
    img = _load_bgr(image)
    canvas, s = _downscale(img, max_dim)     # s = 축소배율 (<=1)
    canvas = canvas.copy()
    H, W = canvas.shape[:2]
    lw = max(1, int(round(min(H, W) / 1400.0)))

    def P(v) -> Tuple[int, int]:
        """원본 좌표 -> 축소된 캔버스 좌표."""
        return (int(round(float(v[0]) * s)), int(round(float(v[1]) * s)))

    # --- 웨이퍼 원 ----------------------------------------------------
    c = P((dm.wafer_cx, dm.wafer_cy))
    r = max(1, int(round(dm.wafer_r * s)))
    cv2.circle(canvas, c, r, (0, 255, 255), lw * 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, c, (0, 255, 255), cv2.MARKER_CROSS,
                   max(8, int(r * 0.10)), lw * 2, cv2.LINE_AA)

    # --- 격자선 -------------------------------------------------------
    # die 로 등록된 인덱스 범위 안에서만 그린다. 화면 전체에 그으면
    # 웨이퍼 밖까지 도배돼서 오히려 안 보인다.
    if draw_grid and dm.dies:
        idx = np.array([d["index"] for d in dm.dies], dtype=np.int64)
        i0, j0 = int(idx[:, 0].min()), int(idx[:, 1].min())
        i1, j1 = int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1
        for i in range(i0, i1 + 1):
            cv2.line(canvas, P(dm.corner_xy(i, j0)), P(dm.corner_xy(i, j1)),
                     (0, 165, 255), lw, cv2.LINE_AA)
        for j in range(j0, j1 + 1):
            cv2.line(canvas, P(dm.corner_xy(i0, j)), P(dm.corner_xy(i1, j)),
                     (0, 165, 255), lw, cv2.LINE_AA)
        cv2.drawMarker(canvas, P(dm.origin), (255, 0, 255),
                       cv2.MARKER_TILTED_CROSS,
                       max(10, int(max(dm.die_w, dm.die_h) * s * 0.8)),
                       lw * 3, cv2.LINE_AA)

    # --- die -----------------------------------------------------------
    # 축정렬 rect_px 가 아니라 quad_px(진짜 네 꼭지점)로 그린다.
    # 기울었을 때 rect 로 그리면 눈으로는 맞아 보여도 실제와 어긋난다.
    if draw_dies:
        for d in dm.dies:
            q = np.array([P(v) for v in d["quad_px"]], dtype=np.int32)
            col = (255, 255, 0) if d["is_edge"] else (0, 220, 0)
            cv2.polylines(canvas, [q], True, col, lw, cv2.LINE_AA)

    # --- 정보 패널 -----------------------------------------------------
    if draw_panel:
        g = dm.diagnostics
        b = g.background_bgr or (0, 0, 0)
        n_edge = sum(1 for d in dm.dies if d["is_edge"])
        lines = [
            "wafer_via_claude   DIE MAP (no image rotation)",
            "wafer    c=(%d,%d)  r=%d   cov=%s"
            % (dm.wafer_cx, dm.wafer_cy, dm.wafer_r,
               "-" if g.wafer_coverage is None else "%.3f" % g.wafer_coverage)
            + ("  FB:%s" % g.wafer_fallback if g.wafer_fallback else ""),
            "bg BGR   (%d,%d,%d)  [%s]" % (b[0], b[1], b[2], g.background_source),
            "pitch    x=%8.3f  y=%8.3f px" % (dm.pitch_x, dm.pitch_y),
            "angle    %+.4f deg" % dm.angle_deg,
            "origin   (%.2f, %.2f)" % (dm.origin[0], dm.origin[1]),
            "vx       (%+.3f, %+.3f)" % (dm.vx[0], dm.vx[1]),
            "vy       (%+.3f, %+.3f)" % (dm.vy[0], dm.vy[1]),
            "dies     %d  (edge %d)  mode=%s" % (len(dm.dies), n_edge, dm.edge_mode),
            "quadrant %s" % ("balanced" if dm.quadrant_report.get("balanced")
                             else "SKEWED - check grid"),
            "OVERALL  %s   elapsed=%.2fs"
            % ("OK" if not g.warnings else "CHECK WARNINGS", g.elapsed_sec),
        ]
        for w in g.warnings[:3]:
            lines.append("  ! %s" % w[:60])

        fs = max(0.45, min(H, W) / 2400.0)
        lh = int(28 * fs / 0.5)
        pw = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
                     for t in lines) + 24)
        ph = lh * len(lines) + 16
        pw, ph = min(pw, W - 8), min(ph, H - 8)
        sub = canvas[8:8 + ph, 8:8 + pw]
        cv2.addWeighted(sub, 0.35, np.zeros_like(sub), 0.65, 0, sub)
        for k, t in enumerate(lines):
            _put_text(canvas, t, (18, 8 + lh * (k + 1) - 8), fs)

    return _imwrite_unicode(path, canvas)
