# -*- coding: utf-8 -*-
"""
synth_wafer_claude.py - 합성 **전체 웨이퍼** 이미지 생성기 (정답 포함)

왜 필요한가
--------------------------------------------------------------------
`synth_clip_claude.py` 는 512x512 클립만 만든다. 그건 서브픽셀 보정과
격자 추정(pitch/angle)을 재는 데는 충분하지만, 그 다음 단계인
"웨이퍼 외곽선 검출 -> die map -> 좌표->인덱스" 를 검증할 수 없다.
그래서 **원 안에 기울어진 격자가 깔린 큰 이미지**를 정답과 함께 만든다.

검증하려는 것
--------------------------------------------------------------------
1. 웨이퍼 중심/반지름을 맞게 찾는가          -> WaferTruth.wafer_*
2. 클립 좌표계 격자를 웨이퍼 좌표계로 옮길 때
   오프셋이 맞는가 (build_die_map_via 의 clip_origin)
3. 좌표 -> die 인덱스가 정답과 같은가        -> WaferTruth.index_at()

메모리
--------------------------------------------------------------------
10000x10000x3 uint8 = 300 MB 다. 32bit 파이썬에서 mgrid 로 float 배열을
통째로 만들면 바로 죽는다. 그래서 **가로 전체 x band_rows 줄**씩 나눠
그린다. 중간 float 버퍼는 band 크기만큼만 존재한다.

사용법
--------------------------------------------------------------------
    from synth_wafer_claude import make_wafer

    img, t = make_wafer(seed=0, size=4000)
    clip = img[t.clip_y0:t.clip_y0+512, t.clip_x0:t.clip_x0+512]
    # -> analyze_clip(clip, yolo) -> build_die_map_via(img, grid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from synth_clip_claude import PALETTE_PRESETS, random_palette

__all__ = ["WaferTruth", "make_wafer"]


@dataclass
class WaferTruth:
    """합성 웨이퍼 하나의 정답."""
    size: int
    wafer_cx: float
    wafer_cy: float
    wafer_r: float

    pitch_x: float
    pitch_y: float
    angle_deg: float
    origin_px: Tuple[float, float]       # die(0,0) 좌상 십자점 (웨이퍼 이미지 좌표)

    clip_x0: int                          # 센터 클립의 좌상단
    clip_y0: int
    clip_size: int

    bg_bgr: Tuple[int, int, int] = (0, 0, 0)
    die_bgr: Tuple[int, int, int] = (0, 0, 0)
    street_bgr: Tuple[int, int, int] = (0, 0, 0)
    palette_name: str = ""
    seed: int = 0

    @property
    def vx(self) -> np.ndarray:
        t = math.radians(self.angle_deg)
        return np.array([self.pitch_x * math.cos(t), self.pitch_x * math.sin(t)])

    @property
    def vy(self) -> np.ndarray:
        t = math.radians(self.angle_deg)
        return np.array([-self.pitch_y * math.sin(t), self.pitch_y * math.cos(t)])

    def index_at(self, x: float, y: float) -> Tuple[int, int]:
        """정답 인덱스. ViaDieMap.index_of 와 **같은 규약**이어야 한다
        (j 는 아래로 증가, die(i,j) = [i,i+1) x [j,j+1) 칸, floor)."""
        V = np.stack([self.vx, self.vy], axis=1)
        ij = np.linalg.solve(V, np.array([x, y], float) - np.array(self.origin_px))
        return int(math.floor(ij[0])), int(math.floor(ij[1]))

    def center_at(self, i: int, j: int) -> np.ndarray:
        return np.array(self.origin_px) + (i + 0.5) * self.vx + (j + 0.5) * self.vy


def _pick_palette(rng: np.random.Generator, palette: Optional[int]):
    if palette is None:
        return random_palette(rng)
    p = PALETTE_PRESETS[int(palette) % len(PALETTE_PRESETS)]
    return p["die"], p["street"], p["dot"], p["name"]


def make_wafer(seed: int = 0,
               size: int = 4000,
               *,
               pitch: Optional[Tuple[float, float]] = None,
               angle_deg: Optional[float] = None,
               street_w: Optional[Tuple[float, float]] = None,
               palette: Optional[int] = None,
               wafer_frac: float = 0.92,
               bg_bgr: Optional[Sequence[int]] = None,
               noise_sigma: Optional[float] = None,
               clip_size: int = 512,
               band_rows: int = 256,
               ) -> Tuple[np.ndarray, WaferTruth]:
    """원형 웨이퍼 + 기울어진 die 격자 이미지를 만든다.

    Parameters
    ----------
    size        : 한 변 픽셀. 10000 도 되지만 300MB 를 쓴다.
    wafer_frac  : 웨이퍼 지름 / 이미지 한 변.
    pitch       : (px, py). None 이면 size 에 비례해 무작위로 (die 가
                  수백 개 나오도록). 클립 512 안에 3~6 칸이 들어오게 잡는다.
    band_rows   : 한 번에 그리는 줄 수. 메모리/속도 트레이드오프.
    """
    rng = np.random.default_rng(seed)

    if pitch is None:
        px = float(rng.uniform(90.0, 170.0))
        py = px * float(rng.uniform(0.8, 1.25))
    else:
        px, py = float(pitch[0]), float(pitch[1])

    ang = float(rng.uniform(-8.0, 8.0)) if angle_deg is None else float(angle_deg)

    if street_w is None:
        wx = float(rng.uniform(0.05, 0.14)) * px
        wy = float(rng.uniform(0.05, 0.14)) * py
    else:
        wx, wy = float(street_w[0]), float(street_w[1])
    wx, wy = max(wx, 3.0), max(wy, 3.0)

    die_bgr, street_bgr, _dot, pname = _pick_palette(rng, palette)
    if bg_bgr is None:
        # 배경은 die/street 어느 쪽과도 안 겹치게. 검출이 배경색을 잘못
        # 잡으면 외곽선이 통째로 틀리므로 일부러 무작위로 흔든다.
        bg_bgr = tuple(int(v) for v in rng.integers(0, 90, 3))
    bg_bgr = tuple(int(v) for v in bg_bgr)

    sig = float(rng.uniform(2.0, 7.0)) if noise_sigma is None else float(noise_sigma)

    wcx = wcy = (size - 1) / 2.0
    wr = size * wafer_frac / 2.0

    # 격자 원점: 웨이퍼 중심 근처에 두되 정확히 중심은 피한다.
    # 중심에 딱 맞추면 "원점이 클립 정중앙"이라는 특수 상황만 테스트하게 된다.
    ox = wcx + float(rng.uniform(-0.5, 0.5)) * px
    oy = wcy + float(rng.uniform(-0.5, 0.5)) * py

    t = math.radians(ang)
    ct, st = math.cos(t), math.sin(t)

    img = np.empty((size, size, 3), np.uint8)
    die_v = np.array(die_bgr, np.float32)
    str_v = np.array(street_bgr, np.float32)
    bg_v = np.array(bg_bgr, np.float32)

    xx = np.arange(size, dtype=np.float32) - np.float32(ox)
    for y0 in range(0, size, band_rows):
        y1 = min(y0 + band_rows, size)
        yy = (np.arange(y0, y1, dtype=np.float32) - np.float32(oy))[:, None]

        u = xx[None, :] * np.float32(ct) + yy * np.float32(st)
        v = -xx[None, :] * np.float32(st) + yy * np.float32(ct)
        du = np.abs(u - np.round(u / np.float32(px)) * np.float32(px))
        dv = np.abs(v - np.round(v / np.float32(py)) * np.float32(py))
        # 1px 램프 안티에일리어싱 (하드컷하면 정답이 픽셀에 스냅돼 버린다)
        a = np.maximum(np.clip(np.float32(wx * 0.5) - du + 0.5, 0.0, 1.0),
                       np.clip(np.float32(wy * 0.5) - dv + 0.5, 0.0, 1.0))[..., None]
        band = die_v[None, None, :] * (1.0 - a) + str_v[None, None, :] * a

        # 웨이퍼 원 밖은 배경
        gx = np.arange(size, dtype=np.float32)[None, :] - np.float32(wcx)
        gy = (np.arange(y0, y1, dtype=np.float32) - np.float32(wcy))[:, None]
        inside = (gx * gx + gy * gy) <= np.float32(wr * wr)
        band = np.where(inside[..., None], band, bg_v[None, None, :])

        if sig > 0:
            band += rng.normal(0.0, sig, band.shape).astype(np.float32)
        img[y0:y1] = np.clip(band, 0, 255).astype(np.uint8)

    cs = int(clip_size)
    cx0, cy0 = (size - cs) // 2, (size - cs) // 2

    truth = WaferTruth(size=size, wafer_cx=wcx, wafer_cy=wcy, wafer_r=wr,
                       pitch_x=px, pitch_y=py, angle_deg=ang,
                       origin_px=(ox, oy),
                       clip_x0=cx0, clip_y0=cy0, clip_size=cs,
                       bg_bgr=bg_bgr, die_bgr=tuple(die_bgr),
                       street_bgr=tuple(street_bgr),
                       palette_name=pname, seed=seed)
    return img, truth


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=4000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    img, t = make_wafer(seed=a.seed, size=a.size)
    print("size=%d  wafer=(%.1f,%.1f) r=%.1f" % (t.size, t.wafer_cx, t.wafer_cy, t.wafer_r))
    print("pitch=%.3f/%.3f  angle=%+.4f  origin=(%.2f,%.2f)"
          % (t.pitch_x, t.pitch_y, t.angle_deg, t.origin_px[0], t.origin_px[1]))
    print("clip=(%d,%d)+%d  palette=%s  bg=%s"
          % (t.clip_x0, t.clip_y0, t.clip_size, t.palette_name, t.bg_bgr))
    if a.out:
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(a.out)
        print("saved", a.out)


if __name__ == "__main__":
    _main()
