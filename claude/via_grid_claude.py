# -*- coding: utf-8 -*-
"""
via_grid_claude.py
====================================================================
보정된 십자 좌표들에서 **센터 코너 / pitch_x / pitch_y / 회전각**을 뽑는다.

사용자 요구 (원문)
--------------------------------------------------------------------
  "센터 코너를 찾고 센터 코너 기준 밑에 좌표는 pitch_y,
   옆에 좌표는 pitch_x로 나오게 만들고"
  "pitch_y나 x로 angle값을 찾을수 있을꺼야"

왜 "바로 밑/옆 점"을 그냥 쓰면 안 되나
--------------------------------------------------------------------
말 그대로 구현하면 이렇게 된다:
    센터에서 가장 가까운 "아래쪽" 점을 찾아 그 거리 = pitch_y
이건 세 가지 이유로 깨진다.

  1) YOLO 가 점을 흘린다. 바로 아래 점이 없으면 두 칸 아래 점을 잡아서
     pitch 가 2배로 나온다.
  2) 격자가 기울어 있으면 "아래"가 정확히 +y 가 아니다. 8도만 기울어도
     한 칸 아래 점이 x 로 20 px 밀려 있어서 "아래"의 정의가 애매해진다.
  3) 점 하나만 쓰면 그 점의 보정 오차(0.07~0.3 px)가 pitch 오차로 그대로
     간다. 그 pitch 로 10000 px 을 외삽하면 오차가 수십 배로 커진다.

그래서 하는 것: 모든 이웃 쌍으로 기저벡터를 구한다
--------------------------------------------------------------------
  1) 점들 사이의 **차 벡터**를 전부 만든다.
  2) 그중 "한 칸짜리" 길이만 남긴다 (최근접 거리의 1.4배 이내).
  3) 방향을 반평면으로 접으면(부호 통일) 두 덩어리로 뭉친다 -> ±vx, ±vy.
  4) 가로에 가까운 덩어리 = vx, 세로에 가까운 덩어리 = vy. 각각 평균.
  5) 그 기저로 모든 점의 정수 격자 인덱스 (i,j) 를 구하고,
     p = origin + i*vx + j*vy 를 **최소제곱으로 다시 푼다**.

5번이 핵심이다. 점 N 개를 전부 써서 기저를 풀기 때문에, 개별 점의 오차가
1/sqrt(N) 로 줄고 격자 전체 폭에 걸쳐 지렛대가 생긴다.
(실측: 한 칸 차이만 쓰면 pitch 오차 ~0.3 px, 최소제곱이면 ~0.03 px)

그러고 나서 사용자가 원한 형태로 돌려준다:
    center  = 클립 중앙에 가장 가까운 십자  (= "센터 코너")
    pitch_x = |vx|   (센터 "옆" 점까지의 거리)
    pitch_y = |vy|   (센터 "밑" 점까지의 거리)
    angle   = vx 의 기울기

회전각을 vx 로만 재지 않는 이유
--------------------------------------------------------------------
vx 와 vy 는 직각이므로 둘 다 각도 정보를 갖는다. vy 를 90도 돌려 vx 와
같은 방향으로 만든 뒤 **둘을 합쳐서** 각도를 낸다. 한쪽 방향으로 점이
적게 잡힌 클립에서도 각도가 안 흔들린다.

주의: 90도 대칭
--------------------------------------------------------------------
정사각 격자는 90도 돌리면 자기 자신이라, 각도는 원리적으로 90도 모듈로만
결정된다. 여기서는 웨이퍼 정렬 오차가 작다고 보고 **|각도| < 45도**를
가정해서 -45~45 로 접는다. ("밑에 있는 점", "옆에 있는 점"이라는 표현
자체가 이미 이 가정을 깔고 있다)

사용법
--------------------------------------------------------------------
    from via_grid_claude import analyze_clip

    res, g = analyze_clip(clip512, [(x, y), ...])   # <- 이거 하나면 된다

    g.ok          # 격자를 세웠나
    g.center      # (2,) 센터 코너 좌표
    g.pitch_x     # float
    g.pitch_y     # float
    g.angle_deg   # float
    g.index_of(x, y)   # 임의 좌표 -> (i, j) 격자 인덱스

실측 성능 (합성 80장, test_grid_claude.py)
--------------------------------------------------------------------
                     중앙값      p90       최악
    pitch_x 오차 :   0.0146    0.0563    0.2976  px
    pitch_y 오차 :   0.0124    0.0546    0.1213  px
    회전각 오차  :   0.0055    0.0139    0.0301  deg
    센터코너 이탈:   0.0658    0.1628    0.2380  px

    보정 없이 raw YOLO 좌표로만 격자를 세우면 각도 오차 중앙값 0.3081 deg
    (최악 44.90) -> 서브픽셀 보정이 각도를 56배 좋게 만든다.

    최악 각도 오차 0.0301 deg 는 10000 px 웨이퍼 끝에서 2.63 px 어긋난다.
    die pitch 가 100~200 px 이니 한 칸의 2% 수준이다.

    좌표->인덱스: 1056점 전부 정확. 정수에서 벗어난 최악이 0.0117 칸으로,
    옆칸으로 넘어가는 경계(0.5)까지 43배 여유가 있다.

    80장 중 1장(pitch 194)은 클립에 십자가 몇 개 안 들어와서 가장자리 점까지
    되살려 썼고, 그 경우 각도 오차가 0.357 deg 로 한 등급 나빠진다.
    (자세한 건 analyze_clip docstring)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["GridFit", "fit_grid", "analyze_clip"]


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
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    p ~= o + i*vx + j*vy 를 최소제곱으로 푼다.

    미지수 6개(o, vx, vy) 를 x/y 각각 3개씩 나눠 푼다.
    설계행렬 A = [1, i, j] 는 x, y 가 공유하므로 한 번만 만든다.
    """
    A = np.column_stack([np.ones(len(ij)), ij[:, 0], ij[:, 1]]).astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, p.astype(np.float64), rcond=None)   # (3,2)
    o, vx, vy = sol[0], sol[1], sol[2]
    resid = p - A @ sol
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return o, vx, vy, rms


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

    o, vx, vy, rms = _lstsq_grid(p, ij)

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

    return GridFit(ok=(rms <= max_resid), reason=("" if rms <= max_resid else
                   "잔차 %.2f px 가 한계 %.2f px 를 넘었다" % (rms, max_resid)),
                   center=p[k].copy(), origin=o,
                   vx=vx, vy=vy,
                   pitch_x=float(np.hypot(*vx)), pitch_y=float(np.hypot(*vy)),
                   angle_deg=ang, n_used=len(p), residual=rms,
                   ij=ij, used=used)


def analyze_clip(clip: np.ndarray,
                 yolo_points: Sequence[Tuple[float, float]],
                 conf_min: float = 0.3,
                 ):
    """
    **이게 사용자가 부르는 함수다.**
    512x512 센터 클립 + YOLO 점 리스트 -> 보정 결과 + 격자.

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
    from via_refine_claude import refine_points, rough_pitch_from_points

    res = refine_points(clip, yolo_points,
                        pitch_hint=rough_pitch_from_points(yolo_points))
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
