# -*- coding: utf-8 -*-
"""
via_diemap_claude.py
====================================================================
**웨이퍼 외곽선** + **클립에서 뽑은 격자** -> 10000x10000 die map,
그리고 좌표 -> die index 조회.

사용자 요구 (원문)
--------------------------------------------------------------------
  "저기서 뽑은 데이터로 10000x10000 Wafer 외각선 기준으로 DM 만들고
   좌표 넣으면 die index 이런거 나오는건 유지해줘"

  "locate_die, 오버레이, Wafer 외각선 찾기, Die map 생성, 회전각 추정
   이렇게는 보존해주고 나머지는 지우고 싶어"

v6 에서 가져온 것 / 버린 것
--------------------------------------------------------------------
가져옴 : 배경색 추정 -> Lab 거리맵 -> Otsu -> 실루엣 -> 원 (외곽선 검출).
         die 순회 / edge 판정 / 4분면 검증 / crop / 오버레이 / 좌표조회.
버림   : 다채널 주기성 격자검출(_spectral_pitch, _fold, detect_grid_adaptive ...)
         과 projection/FFT 회전각 추정(estimate_grid_angle_adaptive).
         pitch 와 회전각을 이제 YOLO 클립에서 직접 얻으므로 필요가 없다.
         (그쪽이 훨씬 정확하다 - 회전각 실측 0.0055 deg vs 주기성 방식 0.1 deg 대)

**이미지를 회전시키지 않는다** (v6 와의 가장 큰 차이)
--------------------------------------------------------------------
v6 는 격자가 축과 나란해지도록 이미지 전체를 워핑한 다음
`ix = floor((x - x0) / pitch_x)` 로 인덱스를 구했다. 10000x10000 에서 이건
1억 픽셀을 재샘플링하는 것이다. 느리고, 메모리를 먹고, 무엇보다 **모든
픽셀이 보간으로 뭉개진다**.

지금은 클립에서 vx, vy 를 이미 알고 있으므로 회전할 이유가 없다:

    die(i,j) 중심 = origin + (i+0.5)*vx + (j+0.5)*vy
    좌표 -> 인덱스  = V^-1 (q - origin),  V = [vx | vy]

2x2 역행렬 한 번이면 끝이고 보간이 전혀 없다. 원본 픽셀이 그대로 남는다.

인덱스 방향 (주의)
--------------------------------------------------------------------
(i, j) 는 **vx, vy 방향**을 센다. vy 는 이미지 아래쪽을 향하므로
**j 는 아래로 갈수록 커진다** (이미지 좌표계와 같은 방향).
v5/v6 의 iy 는 위로 갈수록 커졌으니 부호가 반대다. 두 관습을 섞으면
반드시 사고가 나므로 여기서는 이미지 방향 하나로 통일했다.
물리 좌표가 필요하면 `real_coord` 를 쓰면 된다 - 그건 v5 처럼 y 가 위로 양수다.

십자점은 die 의 "모서리"다
--------------------------------------------------------------------
YOLO 가 찾는 십자(street 교차점)는 die 중심이 아니라 **네 die 가 만나는 꼭지점**
이다. 그래서 die 중심에는 +0.5 칸이 붙는다. v6 도 `x0 + ix*px + px/2` 로
같은 일을 했다 (x0,y0 가 코너였다).

사용법
--------------------------------------------------------------------
    from via_grid_claude import analyze_clip
    from via_diemap_claude import build_die_map_via, locate_die_via

    res, g = analyze_clip(clip512, yolo_points)        # 클립에서 격자
    dm = build_die_map_via(wafer_img, g)               # 웨이퍼 외곽선 + die map
    hit = locate_die_via(dm, point=(4820.0, 3110.0))   # 좌표 -> die
    hit["die_index"], hit["is_edge"], hit["real_coord"]
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
    "WaferProfile", "DieMapDiag", "ViaDieMap",
    "detect_wafer_adaptive", "clean_wafer_outside",
    "build_die_map_via", "locate_die_via",
    "clip_die", "crop_die", "save_debug_overlay",
]

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
    )


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
            "via_diemap_claude   DIE MAP (no image rotation)",
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
