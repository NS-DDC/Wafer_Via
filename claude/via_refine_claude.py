# -*- coding: utf-8 -*-
"""
via_refine_claude.py
====================================================================
YOLO 가 준 십자 좌표를 **서브픽셀로 보정**한다. (색 무관)

문제
--------------------------------------------------------------------
YOLO 박스 중심은 보통 1~3 px 흔들린다. 그 상태로 pitch 를 재면
pitch 오차가 그대로 각도 오차가 되고, 그 각도로 10000x10000 웨이퍼에
격자를 외삽하면 끝에서 수십 px 이 밀린다.
그래서 **이미지를 다시 봐서** 좌표를 다듬어야 한다.

왜 색을 못 믿나
--------------------------------------------------------------------
사용자 요구: "빨간색으로 된건 die인데 색상이 고정은 아니야".
그래서 아래를 전부 **가정하지 않는다**:
  - street 가 밝은지 어두운지 (극성)
  - street 가 무채색인지 유채색인지
  - die 가 무슨 색인지
gray 로 바꾸는 것도 안 된다. die 와 street 의 **밝기가 같고 색만 다른**
경우(예: 초록 die + 보라 street) gray 에서 십자가 통째로 사라진다.

대신 쓰는 것: "국소 지배색에서 얼마나 벗어났나"
--------------------------------------------------------------------
    bg  = medianBlur(img, K)        # K 를 street 폭보다 크게 -> street 가 지워진
                                    #   "die 색 지도"만 남는다
    d   = ||Lab(img) - Lab(bg)||    # street 는 크게 벗어남, die 는 0 근처

d 는 극성이 없다(제곱/절대값). die 색도 모른다(그때그때 median 이 알려줌).
조명 기울기에도 강하다(median 이 같이 따라 움직임).
점 노이즈(dot)는 작아서 median 한 번 더 돌리면 사라진다.

보정 방법: 띠 프로파일 무게중심
--------------------------------------------------------------------
십자 중심 (cx, cy) 주변에서 x 와 y 를 따로 푼다.

  x 를 구할 때:  중심 근처 **가로 띠**(|dy| <= B)만 잘라서 열 방향으로 평균
                 -> 세로 street 자리에 봉우리가 하나 생긴다
                 -> 바닥(median)을 빼고 봉우리 무게중심 = cx
  y 를 구할 때:  같은 걸 90도 돌려서 (세로 띠 -> 행 평균)

이걸 3번 반복한다(매번 새 중심에서 다시 자름).

왜 cv2.cornerSubPix 를 안 쓰나 (실측으로 버림)
--------------------------------------------------------------------
처음엔 cornerSubPix 를 썼다. 결과가 두 갈래로 갈렸다:
30장 중 절반은 0.3 px, 나머지 절반은 10~18 px 로 발산.
이유는 십자 중심이 **saddle 이 아니기 때문**이다. street 와 die 가 만나는
오목한 모서리 4개가 중심 주위에 진짜 코너로 존재해서, cornerSubPix 가
중심 대신 그 모서리로 미끄러진다. 이동 거리가 딱 윈도우 크기만큼이었다.
띠 프로파일은 십자 구조 자체를 모델링하므로 그 함정이 없다.

회전각을 몰라도 되는 이유
--------------------------------------------------------------------
격자가 기울어 있으면 띠 안에서 세로 street 가 비스듬히 지나가 봉우리가
번진다. 하지만 띠가 중심에 대해 **대칭**이면 번짐도 대칭이라
무게중심은 그대로다. (B=9, 8도 기울기 -> 좌우로 ±1.26 px 대칭 번짐)
그래서 1차 보정에 각도가 필요 없다. 각도는 보정된 점들로 나중에 구한다.

윈도우 크기는 pitch 의 함수다. 너무 크면 옆 십자를 물고,
너무 작으면 봉우리 바닥을 못 본다. pitch 를 아직 모를 때는
raw YOLO 점들의 최근접 거리로 대충 잡으면 충분하다
(대충 잡아도 윈도우 크기라 1~2px 틀려도 결과가 안 변한다).

신뢰도
--------------------------------------------------------------------
YOLO 는 오검출도 낸다. die 한복판을 찍은 점은 보정해도 십자가 아니다.
confidence 는 **두 축 프로파일 봉우리 높이 중 약한 쪽**이다.
보정 위치의 streetness 를 쓰면 안 된다 — street 중간을 잘못 찍은 점도
streetness 는 1.0 이라 진짜 십자와 안 갈린다(실측으로 겹쳤다).
십자는 가로/세로 street 가 **둘 다** 있어야 하므로 min 이 그 조건이다.
호출부에서 자르게 하고, 자체 판단으로 몰래 버리지 않는다.

사용법
--------------------------------------------------------------------
    from via_refine_claude import refine_points, rough_pitch_from_points

    p0 = rough_pitch_from_points(yolo_pts)          # 대략적 pitch
    res = refine_points(img, yolo_pts, pitch_hint=p0)
    res.points        # (N,2) 보정 좌표
    res.confidence    # (N,)  0~1, 낮으면 오검출 의심 (border 점은 0)
    res.moved         # (N,)  보정으로 움직인 거리(px)
    res.border        # (N,)  bool, 윈도우가 이미지 밖으로 나가 못 믿는 점

    good = res.points[res.confidence >= 0.3]        # 격자 맞출 때 이것만 쓴다

실측 성능 (합성 60장, 600점, test_refine_claude.py)
--------------------------------------------------------------------
    raw YOLO   중앙값 3.051 px   p90 5.407 px
    보정 후    중앙값 0.069 px   p90 0.155 px   최악 0.290 px   (44x)
    confidence 진짜 점 하위10% 0.474  vs  오검출 상위10% 0.003
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "RefineResult",
    "streetness_map",
    "rough_pitch_from_points",
    "refine_points",
]

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
