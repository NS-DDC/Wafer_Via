# -*- coding: utf-8 -*-
"""
synth_clip_claude.py
====================================================================
Wafer 중앙 512x512 Clip 합성기 (ground truth 포함)

왜 필요한가
--------------------------------------------------------------------
실제 데이터에는 정답이 없다. YOLO 가 찍어준 점이 "몇 px 틀렸는지",
거기서 뽑은 pitch/angle 이 "몇 % / 몇 도 틀렸는지"를 숫자로 못 잰다.
그래서 **정답을 아는 이미지**를 먼저 만든다.

이 파일이 만들어 주는 것
--------------------------------------------------------------------
1) img       : 512x512 BGR 클립 (die + street 격자 + 점 노이즈 + 조명/블러)
2) ClipTruth : 그 이미지의 정답
                 - pitch_x, pitch_y  (px)
                 - angle_deg         (격자 회전각)
                 - points            (N,2) 클립 안 street 교차점 전부
                 - center_point      클립 중심에 가장 가까운 교차점
3) simulate_yolo() : 정답을 흔들어 만든 "YOLO 가 뱉었을 법한 점 리스트"
                     (jitter / 누락 / 오검출 포함)

색은 고정이 아니다
--------------------------------------------------------------------
사용자 요구: "빨간색으로 된건 die인데 색상이 고정은 아니야".
random_palette() 가 매번 die/street/dot 색을 새로 뽑는다.
단, Lab 거리로 최소 대비를 강제해서 "사람도 못 푸는 문제"는 안 만든다.
PALETTE_PRESETS 에 손으로 고른 극단 케이스도 넣어 뒀다
(흰 street + 갈색 노이즈 = v5 가 실패했던 그 조합 포함).

좌표 규약
--------------------------------------------------------------------
픽셀 (x, y) 는 **픽셀 중심** 기준 실수 좌표.
정수 인덱스 i 의 픽셀 중심은 i + 0.5 가 아니라 i 로 둔다
(OpenCV 그리기/보정 함수들과 맞추기 위해).

격자 모델
--------------------------------------------------------------------
    p(i, j) = origin + i * vx + j * vy
    vx = pitch_x * ( cos t,  sin t)
    vy = pitch_y * (-sin t,  cos t)
    t  = angle_deg (양수 = 반시계... 는 아니고 화면 좌표계라 시계방향)

즉 vx 와 vy 는 항상 직교한다. 실제 웨이퍼도 그렇다.

사용법
--------------------------------------------------------------------
    from synth_clip_claude import make_clip, simulate_yolo

    img, truth = make_clip(seed=0)
    pts = simulate_yolo(truth, seed=0)      # [(x, y), ...]

    print(truth.pitch_x, truth.pitch_y, truth.angle_deg)
    print(truth.center_point)

CLI:
    python synth_clip_claude.py --out synth_clips --n 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "ClipTruth",
    "make_clip",
    "simulate_yolo",
    "random_palette",
    "PALETTE_PRESETS",
    "save_clip",
]


# ====================================================================
# 0) 색 팔레트
# ====================================================================

# (die_bgr, street_bgr, dot_bgr, 이름)
# 손으로 고른 극단 케이스. random 만 쓰면 "우연히 안 나오는 조합"이 생긴다.
PALETTE_PRESETS: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], str]] = [
    ((36, 28, 237), (255, 255, 255), (0, 0, 0), "sample_red_white"),        # 사용자가 준 샘플 그대로
    ((255, 255, 255), (60, 90, 140), (30, 30, 30), "white_die_brown_street"),
    ((60, 90, 140), (245, 245, 240), (90, 120, 170), "brown_die_white_street"),  # v5 실패 케이스
    ((120, 120, 120), (150, 150, 150), (90, 90, 90), "lowcontrast_gray"),   # 대비 최소
    ((40, 40, 40), (200, 200, 200), (80, 80, 80), "dark_die_bright_street"),
    ((210, 200, 190), (70, 60, 55), (180, 170, 160), "bright_die_dark_street"),
    ((150, 200, 90), (200, 120, 220), (110, 160, 60), "green_die_violet_street"),
    ((80, 160, 200), (40, 60, 70), "dummy", "teal_die_dark_street"),        # dot 은 아래서 교체
]
# 위 리스트 마지막 항목의 dot 자리 정리 (오타 방지용으로 명시적으로 덮어씀)
PALETTE_PRESETS[-1] = ((80, 160, 200), (40, 60, 70), (50, 110, 140), "teal_die_dark_street")


def _bgr_to_lab(bgr: Sequence[int]) -> np.ndarray:
    """BGR 1픽셀 -> Lab float. 색 거리 재는 용도."""
    px = np.array([[list(bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float64)[0, 0]
    return lab


def _lab_dist(a: Sequence[int], b: Sequence[int]) -> float:
    return float(np.linalg.norm(_bgr_to_lab(a) - _bgr_to_lab(b)))


def random_palette(rng: np.random.Generator, min_lab_dist: float = 28.0,
                   max_try: int = 200):
    """
    die / street / dot 색을 무작위로 뽑되 die-street Lab 거리를 강제.

    min_lab_dist 를 너무 낮추면 사람 눈으로도 street 를 못 찾는 이미지가 나온다.
    28 은 "흐릿하지만 분명히 보인다" 수준. 낮추고 싶으면 인자로 조절.
    """
    for _ in range(max_try):
        die = tuple(int(v) for v in rng.integers(0, 256, 3))
        street = tuple(int(v) for v in rng.integers(0, 256, 3))
        if _lab_dist(die, street) < min_lab_dist:
            continue
        # dot 은 die 근처 색 (die 위에 얹는 텍스처라 die 와 너무 다르면 부자연)
        dot = tuple(int(np.clip(die[k] + rng.integers(-70, 71), 0, 255)) for k in range(3))
        return die, street, dot, "random"
    # 실패하면 확실한 기본값
    return (36, 28, 237), (255, 255, 255), (0, 0, 0), "fallback"


# ====================================================================
# 1) 정답 자료구조
# ====================================================================

@dataclass
class ClipTruth:
    """합성 클립 하나의 정답.

    points 는 (N,2) float32, 클립 안(마진 포함)에 들어온 street 교차점 전부.
    center_point 는 그 중 클립 중심에 가장 가까운 것 = "센터 코너"의 정답.
    """
    size: int
    pitch_x: float
    pitch_y: float
    angle_deg: float
    street_w_x: float
    street_w_y: float
    origin_px: Tuple[float, float]
    points: np.ndarray = field(repr=False)
    point_ij: np.ndarray = field(repr=False)      # (N,2) int, 각 point 의 (i, j)
    center_point: Tuple[float, float] = (0.0, 0.0)
    center_ij: Tuple[int, int] = (0, 0)
    die_bgr: Tuple[int, int, int] = (0, 0, 0)
    street_bgr: Tuple[int, int, int] = (0, 0, 0)
    dot_bgr: Tuple[int, int, int] = (0, 0, 0)
    palette_name: str = ""
    seed: int = 0

    # --- 편의 ---
    @property
    def vx(self) -> np.ndarray:
        t = math.radians(self.angle_deg)
        return np.array([self.pitch_x * math.cos(t), self.pitch_x * math.sin(t)])

    @property
    def vy(self) -> np.ndarray:
        t = math.radians(self.angle_deg)
        return np.array([-self.pitch_y * math.sin(t), self.pitch_y * math.cos(t)])

    def to_json(self) -> dict:
        d = asdict(self)
        d["points"] = np.asarray(self.points, dtype=float).tolist()
        d["point_ij"] = np.asarray(self.point_ij, dtype=int).tolist()
        return d


# ====================================================================
# 2) 클립 렌더링
# ====================================================================

def _render_grid(size: int, px: float, py: float, ang_deg: float,
                 ox: float, oy: float, wx: float, wy: float) -> np.ndarray:
    """
    street 알파맵을 만든다. 0 = die, 1 = street.

    안티에일리어싱: 경계에서 하드컷하면 서브픽셀 보정 테스트가
    "정답이 픽셀 격자에 스냅된 상태"를 재는 꼴이 된다. 그러면 보정 알고리즘
    성능을 과대평가한다. 그래서 1px 폭 선형 램프를 넣는다.
    """
    t = math.radians(ang_deg)
    ct, st = math.cos(t), math.sin(t)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - np.float32(ox)
    dy = yy - np.float32(oy)

    # 격자 좌표계로 회전
    u = dx * np.float32(ct) + dy * np.float32(st)
    v = -dx * np.float32(st) + dy * np.float32(ct)

    # 가장 가까운 격자선까지의 부호 없는 거리
    du = np.abs(u - np.round(u / np.float32(px)) * np.float32(px))
    dv = np.abs(v - np.round(v / np.float32(py)) * np.float32(py))

    ax = np.clip(np.float32(wx * 0.5) - du + 0.5, 0.0, 1.0)
    ay = np.clip(np.float32(wy * 0.5) - dv + 0.5, 0.0, 1.0)
    return np.maximum(ax, ay)


def _lattice_points(size: int, px: float, py: float, ang_deg: float,
                    ox: float, oy: float, margin: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """클립 안(마진 포함)에 들어오는 격자 교차점과 그 (i,j) 인덱스."""
    t = math.radians(ang_deg)
    ct, st = math.cos(t), math.sin(t)
    vx = np.array([px * ct, px * st])
    vy = np.array([-py * st, py * ct])
    o = np.array([ox, oy])

    # 넉넉하게 훑는다. 회전 때문에 필요한 i,j 범위가 커진다.
    n = int(math.ceil(size * 1.5 / max(min(px, py), 1.0))) + 2
    ii, jj = np.mgrid[-n:n + 1, -n:n + 1]
    ii = ii.ravel()
    jj = jj.ravel()
    pts = o[None, :] + ii[:, None] * vx[None, :] + jj[:, None] * vy[None, :]

    lo, hi = margin, size - 1 - margin
    keep = ((pts[:, 0] >= lo) & (pts[:, 0] <= hi) &
            (pts[:, 1] >= lo) & (pts[:, 1] <= hi))
    return pts[keep].astype(np.float32), np.stack([ii[keep], jj[keep]], axis=1)


def make_clip(seed: int = 0,
              size: int = 512,
              pitch: Optional[Tuple[float, float]] = None,
              angle_deg: Optional[float] = None,
              street_w: Optional[Tuple[float, float]] = None,
              palette: Optional[int] = None,
              dot_density: Optional[float] = None,
              noise_sigma: Optional[float] = None,
              blur_sigma: Optional[float] = None,
              illum: Optional[float] = None,
              point_margin: float = 8.0,
              ) -> Tuple[np.ndarray, ClipTruth]:
    """
    합성 클립 1장 + 정답.

    인자를 None 으로 두면 seed 기반으로 무작위. 값을 주면 그 값 고정.
    (테스트 하네스에서 "각도만 바꿔가며" 같은 스윕을 하려고 이렇게 뒀다.)

    palette : None = 무작위, int = PALETTE_PRESETS 인덱스
    point_margin : 이 픽셀보다 가장자리에 가까운 교차점은 정답에서 제외.
                   (클립 밖으로 잘린 십자는 YOLO 도 못 잡고 보정도 못 한다)
    """
    rng = np.random.default_rng(seed)

    if pitch is None:
        px = float(rng.uniform(80.0, 200.0))
        py = px * float(rng.uniform(0.75, 1.35))
        py = float(np.clip(py, 70.0, 220.0))
    else:
        px, py = float(pitch[0]), float(pitch[1])

    if angle_deg is None:
        angle_deg = float(rng.uniform(-8.0, 8.0))

    if street_w is None:
        wx = float(rng.uniform(0.04, 0.11) * px)
        wy = float(rng.uniform(0.04, 0.11) * py)
        wx = float(np.clip(wx, 3.0, 20.0))
        wy = float(np.clip(wy, 3.0, 20.0))
    else:
        wx, wy = float(street_w[0]), float(street_w[1])

    # 원점 위상은 무작위 — 중심에 딱 맞춰 두면 "센터 코너 찾기"가 공짜가 된다.
    ox = size * 0.5 + float(rng.uniform(-0.5, 0.5)) * px
    oy = size * 0.5 + float(rng.uniform(-0.5, 0.5)) * py

    if palette is None:
        die_bgr, street_bgr, dot_bgr, pname = random_palette(rng)
    else:
        die_bgr, street_bgr, dot_bgr, pname = PALETTE_PRESETS[int(palette) % len(PALETTE_PRESETS)]

    dot_density = 0.02 if dot_density is None else float(dot_density)
    noise_sigma = float(rng.uniform(2.0, 9.0)) if noise_sigma is None else float(noise_sigma)
    blur_sigma = float(rng.uniform(0.4, 1.4)) if blur_sigma is None else float(blur_sigma)
    illum = float(rng.uniform(0.0, 0.22)) if illum is None else float(illum)

    # --- 합성 ---
    alpha = _render_grid(size, px, py, angle_deg, ox, oy, wx, wy)[..., None]
    die = np.full((size, size, 3), np.array(die_bgr, dtype=np.float32), dtype=np.float32)
    street = np.full((size, size, 3), np.array(street_bgr, dtype=np.float32), dtype=np.float32)

    # die 위 점 노이즈 (샘플 이미지의 검은 점들)
    if dot_density > 0:
        n_dot = int(size * size * dot_density / 25.0)   # 점 하나가 대략 5x5
        cx = rng.integers(0, size, n_dot)
        cy = rng.integers(0, size, n_dot)
        dots = np.zeros((size, size), dtype=np.float32)
        for x0, y0 in zip(cx, cy):
            r = int(rng.integers(1, 3))
            cv2.circle(dots, (int(x0), int(y0)), r, 1.0, -1)
        dots = dots[..., None]
        die = die * (1.0 - dots) + np.array(dot_bgr, dtype=np.float32) * dots

    img = die * (1.0 - alpha) + street * alpha

    # 조명 기울기 (실사 이미지의 vignetting / 조명 불균일)
    # ramp 를 반드시 [0,1] 로 정규화한다. gx*cos + gy*sin 은 각도에 따라
    # [-1.41, 1.41] 까지 벌어져서, 정규화를 빼먹으면 밝기가 최대 84% 까지
    # 깎인다. (실제로 그 버그를 _self_check 의 street_hit=0.17 로 잡았다)
    if illum > 0:
        gy, gx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
        ang = float(rng.uniform(0, 2 * math.pi))
        ramp = gx * math.cos(ang) + gy * math.sin(ang)
        lo, hi = float(ramp.min()), float(ramp.max())
        ramp = (ramp - lo) / max(hi - lo, 1e-6)          # -> [0, 1]
        img = img * (1.0 + illum * (ramp - 0.5) * 2.0)[..., None]   # -> [1-illum, 1+illum]

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- 정답 ---
    pts, ij = _lattice_points(size, px, py, angle_deg, ox, oy, point_margin)
    c = np.array([(size - 1) * 0.5, (size - 1) * 0.5], dtype=np.float32)
    if len(pts) == 0:
        raise RuntimeError("교차점이 하나도 안 잡혔다. pitch 가 클립보다 크다.")
    k = int(np.argmin(np.sum((pts - c[None, :]) ** 2, axis=1)))

    truth = ClipTruth(
        size=size, pitch_x=px, pitch_y=py, angle_deg=angle_deg,
        street_w_x=wx, street_w_y=wy, origin_px=(ox, oy),
        points=pts, point_ij=ij,
        center_point=(float(pts[k, 0]), float(pts[k, 1])),
        center_ij=(int(ij[k, 0]), int(ij[k, 1])),
        die_bgr=tuple(int(v) for v in die_bgr),
        street_bgr=tuple(int(v) for v in street_bgr),
        dot_bgr=tuple(int(v) for v in dot_bgr),
        palette_name=pname, seed=int(seed),
    )
    return img, truth


# ====================================================================
# 3) YOLO 출력 흉내
# ====================================================================

def simulate_yolo(truth: ClipTruth,
                  seed: int = 0,
                  jitter_px: float = 2.5,
                  drop_rate: float = 0.06,
                  fp_rate: float = 0.04,
                  ) -> List[Tuple[float, float]]:
    """
    정답 교차점을 흔들어서 "YOLO 가 뱉었을 법한 점 리스트"를 만든다.

    jitter_px : 각 점에 더할 가우시안 위치 오차의 sigma.
                YOLO 박스 중심은 보통 1~3px 흔들린다고 보고 잡았다.
    drop_rate : 놓친 비율 (누락)
    fp_rate   : 엉뚱한 곳에 찍은 비율 (오검출) — die 한가운데에 뿌린다.

    반환은 사용자가 준다고 한 형식 그대로: [(x, y), ...]
    """
    rng = np.random.default_rng(seed + 99991)
    out: List[Tuple[float, float]] = []

    for p in np.asarray(truth.points, dtype=np.float64):
        if rng.random() < drop_rate:
            continue
        q = p + rng.normal(0.0, jitter_px, 2)
        out.append((float(q[0]), float(q[1])))

    n_fp = int(round(len(truth.points) * fp_rate))
    for _ in range(n_fp):
        out.append((float(rng.uniform(0, truth.size - 1)),
                    float(rng.uniform(0, truth.size - 1))))

    rng.shuffle(out)   # 순서에 의존하는 코드를 걸러내려고 섞는다
    return out


# ====================================================================
# 4) 저장
# ====================================================================

def save_clip(out_dir: str, name: str, img: np.ndarray, truth: ClipTruth,
              yolo_pts: Optional[Sequence[Tuple[float, float]]] = None) -> str:
    """png + json 저장. 한글 경로 대비해서 imencode 로 쓴다."""
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, name + ".png")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("imencode 실패: " + png)
    buf.tofile(png)

    meta = truth.to_json()
    if yolo_pts is not None:
        meta["yolo_points"] = [[float(a), float(b)] for a, b in yolo_pts]
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return png


# ====================================================================
# 5) 자가검증 + CLI
# ====================================================================

def _self_check(img: np.ndarray, truth: ClipTruth) -> dict:
    """
    정답 점이 정말 street 위에 있는지 확인한다.

    방법: 정답 점 픽셀의 색이 die 색보다 street 색에 가까운지 본다.
    (노이즈/블러 때문에 100% 는 아니지만 90% 아래로 떨어지면 렌더가 잘못된 것)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_die = _bgr_to_lab(truth.die_bgr).astype(np.float32)
    lab_str = _bgr_to_lab(truth.street_bgr).astype(np.float32)

    hit = 0
    for x, y in np.asarray(truth.points, dtype=np.float64):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < truth.size and 0 <= yi < truth.size):
            continue
        c = lab[yi, xi]
        if np.linalg.norm(c - lab_str) < np.linalg.norm(c - lab_die):
            hit += 1
    n = len(truth.points)
    return {"n_points": n, "on_street": hit, "ratio": (hit / n if n else 0.0)}


def _main() -> None:
    ap = argparse.ArgumentParser(description="합성 512x512 클립 생성기")
    ap.add_argument("--out", default="synth_clips")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--presets", action="store_true",
                    help="PALETTE_PRESETS 를 순서대로 사용")
    args = ap.parse_args()

    print("합성 클립 생성:", args.out)
    print("-" * 78)
    bad = 0
    for i in range(args.n):
        pal = (i % len(PALETTE_PRESETS)) if args.presets else None
        img, truth = make_clip(seed=i, size=args.size, palette=pal)
        pts = simulate_yolo(truth, seed=i)
        name = "clip_%03d_%s" % (i, truth.palette_name)
        save_clip(args.out, name, img, truth, pts)
        chk = _self_check(img, truth)
        flag = "" if chk["ratio"] >= 0.90 else "  <<< RENDER SUSPECT"
        if flag:
            bad += 1
        print("%-34s pitch=(%6.2f,%6.2f) ang=%+6.2f pts=%2d yolo=%2d street_hit=%.2f%s"
              % (name, truth.pitch_x, truth.pitch_y, truth.angle_deg,
                 chk["n_points"], len(pts), chk["ratio"], flag))
    print("-" * 78)
    print("의심 이미지 %d / %d" % (bad, args.n))


if __name__ == "__main__":
    _main()
