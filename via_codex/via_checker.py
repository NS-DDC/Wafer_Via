# -*- coding: utf-8 -*-
"""via_checker.py - PAD 평균 밝기 기반 중앙 VIA 검사 (단일 파일, 복붙용)

이미지 네 장을 넣으면 코드와 결과 이미지를 반환합니다.

    from via_checker import check_via

    code, result, via_bin = check_via(원본, 이진화, PAD설계도, VIA설계도)

    # code    : "1"  검사 대상 PAD마다 중앙 VIA가 있음
    #           "42" 중앙 VIA 없음
    #           "-1" 입력 오류 (이때 result와 via_bin은 None)
    # result  : 원본 해상도의 표시 이미지
    # via_bin : 실제로 채택한 VIA만 흰색인 0/255 단일채널 마스크

이 버전의 의도
-------------
* 컬러 이미지를 회색조로 바꾸고 PAD 평균 밝기보다 일정 수준 어두운 부분을
  이진화해 VIA 후보로 사용합니다.
* 후보 연결성분의 실제 어두운 픽셀이 PAD 중앙 허용영역 안에 있을 때만 VIA로
  인정합니다. 연결성분 무게중심 거리는 후보 선택에 사용하지 않습니다.
* PAD 이진 마스크의 커버리지로 검사 대상을 제외하지 않습니다.
  ``PAD_PRESENT_MIN``과 ``VIA_EXCLUDE_RATIO`` 필터는 제거했습니다.
* 쏠림(VIA_OFFSET)은 불량으로 판정하지 않습니다. 중앙 허용거리 안에서
  검출된 VIA는 중심 오차가 있더라도 OK입니다. 기존 호출부 import 호환을 위해
  CODE_VIA_OFFSET 상수만 남지만 code "99"는 반환하지 않습니다.

검출 순서
---------
1. VIA 설계도에 점이 있는 PAD만 검사합니다.
2. 실측 이진 마스크가 충분하면 설계 PAD를 국소 정합합니다. 커버리지가 낮거나
   비어 있어도 검사를 건너뛰지 않고 원래 설계 PAD 중심을 사용합니다.
3. ROI를 회색조로 바꾸고 PAD 내부 평균 밝기를 구합니다.
4. ``PAD평균 - VIA_GRAY_DROP + dark_offset``보다 어두운 픽셀만 이진화합니다.
5. 최소 면적을 넘는 연결성분에서 PAD 중심에 가장 가까운 실제 픽셀을 구합니다.
   그 픽셀이 ``PAD반지름 * VIA_CENTER_SEARCH_RATIO`` 안에 있는 덩어리 중 실제
   픽셀이 중앙에 가장 가까운 덩어리를 VIA로 선택합니다.

사용자가 수정할 곳
------------------
* ``[SECTOR: VIA_DETECTION_CONFIG]``: 밝기와 중앙 위치 임계값
* ``[SECTOR: VIA_DETECTION_CORE]``: 실제 후보 마스크를 만드는 순서

``dark_offset``은 기존 호출 호환용입니다. 양수면 이진화 임계값이 올라가 더
민감해지고, 음수면 더 어두운 부분만 남아 엄격해집니다. 기본값 0부터 5~10
단위로 조절하세요.

입력 네 장은 모두 같은 해상도와 좌표계여야 합니다.
의존성: numpy, opencv-python (Python 3.9+)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["check_via", "debug_via",
           "CODE_OK", "CODE_VIA_MISSING", "CODE_VIA_OFFSET", "CODE_ERROR"]


# ============================================================================
# 결과 코드
# ============================================================================
CODE_OK = "1"            # 양품
CODE_VIA_MISSING = "42"  # VIA 없음
CODE_VIA_OFFSET = "99"   # import 호환용: 판정에서는 반환하지 않음
CODE_ERROR = "-1"        # 입력 오류

# 이 버전에서 VIA 불량 코드는 중앙 VIA 없음만 사용합니다.
CODE_PRIORITY = [CODE_VIA_MISSING]


# ============================================================================
# [SECTOR: VIA_DETECTION_CONFIG]
# VIA 검출 튜닝 값 - 사용자가 가장 먼저 수정할 곳
# ============================================================================

# 설계 PAD 가 실물보다 몇 px 작게 그려졌는지. 그 만큼 되돌려서 공칭 크기로 씁니다.
# 설계도를 실물과 같은 크기로 그렸으면 0.
DESIGN_PAD_SHRINK = 2

# VIA 연결성분의 실제 픽셀 허용거리 = PAD 등가반지름 * 이 값.
# 0.30이면 후보의 실제 어두운 픽셀이 PAD 반지름의 30% 안쪽에 있어야 VIA입니다.
# 외곽 선을 VIA로 잘못 잡으면 낮추고, 정상 VIA가 빠지면 0.05씩 올리세요.
VIA_CENTER_SEARCH_RATIO = 0.30

# 국소 정합. 중앙 검색원이 실물 PAD 중앙과 맞도록 설계 PAD를 조금 이동합니다.
# 0이면 정합하지 않습니다.
ALIGN_MAX_RATIO = 0.40    # 최대 이동량 = PAD반지름 * 이 값
ALIGN_MARGIN = 2          # 무게중심을 잴 때 설계 PAD 를 몇 px 키운 창을 볼지
ALIGN_MIN_COVER = 0.30    # 창 안 실측 픽셀이 이 비율도 안 되면 정합 생략

# 회색조 이진화 기준.
# VIA 후보 = gray < PAD영역 평균 - VIA_GRAY_DROP + dark_offset
# 값이 크면 더 어두운 부분만 남아 엄격하고, 작으면 연한 VIA도 잡습니다.
VIA_GRAY_DROP = 25.0

# 한두 픽셀 카메라 잡음만 제외하기 위한 유일한 크기 조건입니다.
VIA_MIN_AREA = 4

# 탐색 영역을 PAD 안쪽으로 몇 px 깎을지.
# 크게 깎으면 작은 VIA가 사라질 수 있으므로 기본 1px만 사용합니다.
PAD_ERODE = 1

# 설계도 잡티 제거용 최소 px. 검사 대상은 "VIA 설계도에 점이 있는 PAD" 로만 정해지므로
# 여기서는 1~3px 짜리 노이즈만 걸러내면 됩니다. 값이 작아 스케일이 바뀌어도 안전합니다.
MIN_PAD_AREA = 4
MIN_VIA_AREA = 1              # VIA 설계 점의 최소 px

# 결과 이미지 마커 색 (BGR)
COLOR_OK = (0, 220, 0)          # 초록 : 정상
COLOR_MISSING = (0, 0, 255)     # 빨강 : VIA 없음
COLOR_VIA = (255, 255, 0)       # 하늘 : 찾아낸 VIA 위치 (판정과 무관하게 항상 표기)

# [END SECTOR: VIA_DETECTION_CONFIG]


# ============================================================================
# 공개 함수
# ============================================================================
def check_via(image: Union[str, np.ndarray],
              bin_mask: Union[str, np.ndarray],
              pad_design: Union[str, np.ndarray],
              via_design: Union[str, np.ndarray],
              dark_offset: float = 0.0
              ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """이미지 4장을 받아 (코드, 결과이미지, VIA이진화) 를 돌려준다.

        code, result, via_bin = check_via(원본, 이진화, PAD설계도, VIA설계도)

    code는 "1"(양품) / "42"(중앙 VIA 없음) / "-1"(입력 오류)입니다.
    호환용 CODE_VIA_OFFSET 상수는 남지만 code "99"는 반환하지 않습니다.
    via_bin 은 검출한 VIA 만 흰색인 0/255 마스크입니다 (원본과 같은 해상도).
    "-1" 이면 result 와 via_bin 은 None 이고, 이유가 표준에러로 출력됩니다.

    bin_mask는 설계 PAD 중심을 실측 쪽으로 미세 정합할 때만 사용합니다.
    PAD 존재/커버리지 필터에는 사용하지 않으므로 비어 있거나 VIA 구멍이 커도
    해당 PAD의 VIA 검사를 건너뛰지 않습니다.

    dark_offset : 회색조 이진화 임계값을 조절합니다 (기본 0).

        임계값 = PAD영역 평균 - VIA_GRAY_DROP + dark_offset

        양수  더 밝은 부분까지 VIA 후보로 포함 -> 민감
        음수  더 어두운 부분만 VIA 후보로 포함 -> 엄격

    5~10 단위로 움직이면서 debug_via의 pad_mean, dark_threshold,
    dark_candidate_pixels 값을 함께 확인하세요.
    """
    code, src, via_bin, rows, err = _run(image, bin_mask, pad_design, via_design,
                                         dark_offset)
    if code == CODE_ERROR:
        sys.stderr.write("[via_checker] %s\n" % err)
        return CODE_ERROR, None, None
    return code, _draw(src, rows, numbering=False), via_bin


def debug_via(image: Union[str, np.ndarray],
              bin_mask: Union[str, np.ndarray],
              pad_design: Union[str, np.ndarray],
              via_design: Union[str, np.ndarray],
              quiet: bool = False,
              dark_offset: float = 0.0
              ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray],
                         List[Dict[str, Any]]]:
    """check_via 와 같은 검사를 하되, 디버깅에 필요한 것을 함께 준다.

        code, result, via_bin, rows = debug_via(원본, 이진화, PAD설계도, VIA설계도)

    앞 세 개는 check_via 와 같습니다. rows 만 추가됩니다.
    dark_offset도 check_via와 같은 뜻입니다.

    - PAD 별 수치를 표로 출력합니다 (quiet=True 로 끄기)
    - 결과 이미지에 PAD 번호를 함께 그립니다
    - rows 는 PAD 별 dict 목록입니다. 들어있는 키:
        pad_id, status, pad_center, pad_radius, pad_area, pad_coverage,
        align_shift, design_via, via_center, via_area, offset_px, offset_norm,
        pad_median, pad_mean, dark_threshold, search_radius,
        nearest_center_pixel_distance, center_zone_pixels,
        dark_candidate_pixels, color_candidate_pixels,
        via_aspect, via_radial_dev, shape_rejected, edge_rejected
      (해당 없는 항목은 None)

    align_shift 가 크게 나오면 설계도와 실물이 그만큼 어긋나 있다는 뜻입니다.
    dark_candidate_pixels는 PAD 평균 기반 이진화를 통과한 픽셀 수입니다.
    search_radius는 실제 후보 픽셀에 허용하는 PAD 중앙 거리입니다.
    nearest_center_pixel_distance는 선택된 덩어리의 실제 픽셀 최소거리이고,
    center_zone_pixels는 중앙 허용영역 안에 들어온 실제 픽셀 수입니다.
    pad_coverage는 이전 rows 호환용이며 항상 None입니다.
    shape_rejected와 edge_rejected는 예전 rows 호환용이며 항상 0입니다.
    """
    code, src, via_bin, rows, err = _run(image, bin_mask, pad_design, via_design,
                                         dark_offset)
    if code == CODE_ERROR:
        if not quiet:
            print("code=-1  ERROR: %s" % err)
        return CODE_ERROR, None, None, [{"status": "ERROR", "message": err}]
    if not quiet:
        _print_table(code, rows)
    return code, _draw(src, rows, numbering=True), via_bin, rows


# ============================================================================
# 이미지 읽기 (한글 경로 대응)
# ============================================================================
def _imread(path: str, flags: int) -> Optional[np.ndarray]:
    """cv2.imread 는 비ASCII 경로에서 실패하므로 np.fromfile 로 우회한다."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def _to_bgr(src: Union[str, np.ndarray], name: str) -> np.ndarray:
    """무엇이 들어와도 uint8 3채널 BGR 로 맞춘다."""
    img = src if isinstance(src, np.ndarray) else _imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("%s를 읽을 수 없습니다: %s" % (name, src))
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _to_mask(src: Union[str, np.ndarray], name: str) -> np.ndarray:
    """무엇이 들어와도 0/255 uint8 단일채널 마스크로 맞춘다."""
    m = src if isinstance(src, np.ndarray) else _imread(str(src), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("%s를 읽을 수 없습니다: %s" % (name, src))
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return np.where(m > 0, 255, 0).astype(np.uint8)


# ============================================================================
# 설계도 해석
# ============================================================================
def _label_pads(pad_design: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]]]:
    """PAD 설계도를 연결요소로 나눈다. 반환 (라벨맵, {pad_id: bbox})

    작은 PAD 도 그대로 남긴다. 실제 검사 대상은 뒤에서 'VIA 설계도에 점이 있는가'
    로만 걸러지므로, 여기서 크기로 자르면 작은 PAD 가 통째로 빠진다.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (pad_design > 0).astype(np.uint8), connectivity=8)

    out = np.zeros(labels.shape, np.int32)
    boxes: Dict[int, Tuple[int, int, int, int]] = {}
    next_id = 1
    for i in range(1, num):
        if int(stats[i, 4]) < MIN_PAD_AREA:
            continue
        out[labels == i] = next_id
        boxes[next_id] = (int(stats[i, 0]), int(stats[i, 1]),
                          int(stats[i, 2]), int(stats[i, 3]))
        next_id += 1
    return out, boxes


def _map_vias(via_design: np.ndarray,
              pad_label: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """VIA 설계도의 점들을 각각 어느 설계 PAD 소속인지 매핑한다.

    반환 {pad_id: (x, y)}. 어느 PAD 에도 안 들어간 점은 버린다.
    """
    H, W = pad_label.shape[:2]
    num, _, stats, cents = cv2.connectedComponentsWithStats(
        (via_design > 0).astype(np.uint8), connectivity=8)

    out: Dict[int, Tuple[float, float]] = {}
    for i in range(1, num):
        if int(stats[i, 4]) < MIN_VIA_AREA:
            continue
        vx, vy = float(cents[i][0]), float(cents[i][1])
        ix = int(np.clip(round(vx), 0, W - 1))
        iy = int(np.clip(round(vy), 0, H - 1))
        pid = int(pad_label[iy, ix])
        if pid > 0:
            out[pid] = (vx, vy)
    return out


# ============================================================================
# 검사 본체
# ============================================================================
def _run(image: Union[str, np.ndarray],
         bin_mask: Union[str, np.ndarray],
         pad_design: Union[str, np.ndarray],
         via_design: Union[str, np.ndarray],
         dark_offset: float = 0.0
         ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray],
                    List[Dict[str, Any]], str]:
    """반환 (code, 원본BGR, VIA이진화, rows, 오류메시지)"""
    try:
        src = _to_bgr(image, "원본 이미지")
        actual = _to_mask(bin_mask, "이진화 이미지")
        pdes = _to_mask(pad_design, "PAD 설계도")
        vdes = _to_mask(via_design, "VIA 설계도")
    except ValueError as e:
        return CODE_ERROR, None, None, [], str(e)

    H, W = src.shape[:2]
    for nm, m in (("이진화 이미지", actual), ("PAD 설계도", pdes), ("VIA 설계도", vdes)):
        if m.shape[:2] != (H, W):
            return CODE_ERROR, None, None, [], (
                "%s 크기%s가 원본 이미지 크기%s와 다릅니다." % (nm, m.shape[:2], (H, W)))

    # 회색조 평균 기반 이진화를 하되 결과 표시는 컬러 원본에 그립니다.
    blur = cv2.GaussianBlur(src, (3, 3), 0)

    pad_label, boxes = _label_pads(pdes)
    design_vias = _map_vias(vdes, pad_label)

    dil = None
    if DESIGN_PAD_SHRINK > 0:
        d = DESIGN_PAD_SHRINK
        dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    # 침식은 1px로 얕게만 합니다. 외곽 검은 선은 후보 연결성분 중심과
    # PAD 중심 사이의 거리로 제외합니다.
    ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * PAD_ERODE + 1, 2 * PAD_ERODE + 1))
    alk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * ALIGN_MARGIN + 1, 2 * ALIGN_MARGIN + 1))

    # 검출한 VIA 만 흰색으로 남길 마스크. PAD 마다 채택한 덩어리를 여기에 찍는다.
    via_bin = np.zeros((H, W), np.uint8)

    rows: List[Dict[str, Any]] = []
    found_status = set()

    for pid in sorted(design_vias):
        dvx, dvy = design_vias[pid]
        x, y, w, h = boxes[pid]
        mg = DESIGN_PAD_SHRINK + PAD_ERODE + 4
        x0, x1 = max(x - mg, 0), min(x + w + mg, W)
        y0, y1 = max(y - mg, 0), min(y + h + mg, H)

        roi = blur[y0:y1, x0:x1]
        act = actual[y0:y1, x0:x1]

        # 기준 형상은 '설계 PAD' 를 쓴다.
        # VIA 가 가장자리로 심하게 쏠리면 어두운 VIA 가 배경과 이어져
        # 실측 PAD 윤곽에 노치가 파이고 중심이 흔들리기 때문이다.
        # '정중앙'의 정의 자체도 설계 중심이므로 이쪽이 타당하다.
        shape = ((pad_label[y0:y1, x0:x1] == pid).astype(np.uint8)) * 255
        if dil is not None:
            shape = cv2.dilate(shape, dil)   # 설계가 작게 그려진 만큼 공칭 크기로 복원

        area = float(np.count_nonzero(shape))
        if area <= 0:
            continue
        ys, xs = np.nonzero(shape)
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(area / np.pi))

        # ---- 국소 정합 : 설계도-실물 어긋남을 흡수한다 ----
        shape, cx, cy, shift = _align(shape, act, cx, cy, radius, area, alk)

        row: Dict[str, Any] = {
            "pad_id": pid,
            "status": None,
            "pad_center": (round(cx + x0, 2), round(cy + y0, 2)),
            "pad_radius": round(radius, 2),
            "pad_area": int(area),
            "pad_coverage": None,
            "align_shift": (round(shift[0], 2), round(shift[1], 2)),
            "design_via": (round(dvx, 2), round(dvy, 2)),
            "via_center": None,
            "via_area": None,
            "offset_px": None,
            "offset_norm": None,
            "pad_median": None,
            "pad_mean": None,
            "dark_threshold": None,
            "search_radius": None,
            "dark_candidate_pixels": 0,
            "nearest_center_pixel_distance": None,
            "center_zone_pixels": 0,
            # 아래 키는 이전 debug_via rows와의 호환용입니다.
            "pad_value_median": None,
            "black_value_max": None,
            "brown_value_max": None,
            "brown_sat_min": None,
            "color_candidate_pixels": 0,
            "via_aspect": None,
            "via_radial_dev": None,
            "shape_rejected": 0,
            "edge_rejected": 0,
        }

        # ---- VIA 찾기: PAD 평균보다 어둡고 중심이 중앙에 가까운 덩어리 ----
        found = _find_via(roi, shape, cx, cy, radius, ero, row, dark_offset)
        if found is None:
            row["status"] = "VIA_MISSING"
            found_status.add(CODE_VIA_MISSING)
            rows.append(row)
            continue

        via_bin[y0:y1, x0:x1][found["mask"]] = 255

        vx, vy = found["cx"] + x0, found["cy"] + y0
        dist = float(np.hypot(found["cx"] - cx, found["cy"] - cy))
        row["via_center"] = (round(vx, 2), round(vy, 2))
        row["via_area"] = int(found["area"])
        row["offset_px"] = round(dist, 2)
        row["offset_norm"] = round(dist / radius if radius > 1e-6 else 999.0, 4)

        # 중앙 허용거리 안에서 검출됐으면 항상 OK입니다. 거리는 진단값일 뿐이며
        # VIA_OFFSET 불량 판정에는 사용하지 않습니다.
        row["status"] = "OK"
        rows.append(row)

    code = CODE_OK
    for c in CODE_PRIORITY:
        if c in found_status:
            code = c
            break
    return code, src, via_bin, rows, ""


def _align(shape: np.ndarray,
           act: np.ndarray,
           cx: float,
           cy: float,
           radius: float,
           area: float,
           alk: np.ndarray) -> Tuple[np.ndarray, float, float, Tuple[float, float]]:
    """설계 PAD 형상을 실측 마스크 쪽으로 조금 평행이동한다.

    설계도와 실물이 2~3px 어긋나면 중앙 검색원도 함께 빗나가 정상 VIA를 놓칠 수
    있습니다. 실측 PAD의 무게중심 쪽으로 설계 형상을 보정하되, 이동량은 반지름의
    ALIGN_MAX_RATIO 이내로 제한해 다른 PAD나 배경으로 이동하지 않게 합니다.
    """
    if ALIGN_MAX_RATIO <= 0:
        return shape, cx, cy, (0.0, 0.0)

    loc = (cv2.dilate(shape, alk) > 0) & (act > 0)
    if np.count_nonzero(loc) < area * ALIGN_MIN_COVER:
        return shape, cx, cy, (0.0, 0.0)

    ly, lx = np.nonzero(loc)
    lim = radius * ALIGN_MAX_RATIO
    sx = float(np.clip(float(lx.mean()) - cx, -lim, lim))
    sy = float(np.clip(float(ly.mean()) - cy, -lim, lim))
    if abs(sx) < 1e-3 and abs(sy) < 1e-3:
        return shape, cx, cy, (0.0, 0.0)

    moved = cv2.warpAffine(shape, np.float32([[1, 0, sx], [0, 1, sy]]),
                           (shape.shape[1], shape.shape[0]), flags=cv2.INTER_NEAREST)
    return moved, cx + sx, cy + sy, (sx, sy)


def _find_via(roi: np.ndarray,
              shape: np.ndarray,
              center_x: float,
              center_y: float,
              radius: float,
              ero: np.ndarray,
              row: Dict[str, Any],
              dark_offset: float) -> Optional[Dict[str, Any]]:
    """PAD 평균보다 어둡고 중심에 가까운 연결성분 하나를 VIA로 반환한다.

    판정 조건은 세 개뿐입니다.
      1) PAD 평균보다 ``VIA_GRAY_DROP - dark_offset`` 이상 어둡다.
      2) 연결성분 면적이 ``VIA_MIN_AREA`` 이상이다.
      3) 연결성분의 실제 픽셀 중 하나 이상이 PAD 중심에서
         ``PAD반지름 * VIA_CENTER_SEARCH_RATIO`` 이내다.

    색상, Black-hat, 원형도, 채움비, 최대 면적 조건은 사용하지 않습니다.
    조건을 만족하는 덩어리가 여러 개면 PAD 중심에 실제 픽셀이 가장 가까운
    덩어리를 고릅니다. 무게중심은 선택 후 표시 좌표를 구할 때만 사용합니다.
    """
    inner = cv2.erode(shape, ero)
    pad_mask = inner > 0
    if np.count_nonzero(pad_mask) < VIA_MIN_AREA:
        return None

    # [SECTOR: VIA_DETECTION_CORE]
    # 1) 회색조 변환 후 PAD 내부 평균을 기준으로 단순 이진화합니다.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pad_mean = float(np.mean(gray[pad_mask]))
    threshold = float(np.clip(
        pad_mean - VIA_GRAY_DROP + float(dark_offset), 0.0, 255.0))
    cand = ((gray < threshold) & pad_mask).astype(np.uint8)

    center_tolerance = max(float(radius * VIA_CENTER_SEARCH_RATIO), 1.0)
    dark_pixels = int(np.count_nonzero(cand))
    row["pad_median"] = round(pad_mean, 1)       # 예전 키 호환
    row["pad_mean"] = round(pad_mean, 1)
    row["dark_threshold"] = round(threshold, 1)
    row["search_radius"] = round(center_tolerance, 2)
    row["dark_candidate_pixels"] = dark_pixels
    row["color_candidate_pixels"] = dark_pixels  # 예전 키 호환
    row["pad_value_median"] = round(pad_mean, 1)  # 예전 키 호환

    num, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)

    # 2) 최소 면적을 넘는 덩어리에서 PAD 중심에 가장 가까운 '실제 픽셀'을 찾습니다.
    # 연결성분 무게중심(cents)은 best 선택에 쓰지 않고 최종 표시 좌표에만 씁니다.
    best = -1
    best_area = 0
    best_pixel_distance = float("inf")
    best_center_zone_pixels = 0
    for i in range(1, num):
        a = int(stats[i, 4])
        if a < VIA_MIN_AREA:
            continue

        ys, xs = np.nonzero(lab == i)
        pixel_distances = np.hypot(xs - center_x, ys - center_y)
        nearest_pixel_distance = float(pixel_distances.min())
        center_zone_pixels = int(np.count_nonzero(
            pixel_distances <= center_tolerance))
        if center_zone_pixels == 0:
            continue

        if (nearest_pixel_distance < best_pixel_distance or
                (abs(nearest_pixel_distance - best_pixel_distance) < 1e-6 and
                 center_zone_pixels > best_center_zone_pixels) or
                (abs(nearest_pixel_distance - best_pixel_distance) < 1e-6 and
                 center_zone_pixels == best_center_zone_pixels and a > best_area)):
            best = i
            best_area = a
            best_pixel_distance = nearest_pixel_distance
            best_center_zone_pixels = center_zone_pixels

    if best < 0:
        return None

    blob = lab == best
    cx = float(cents[best][0])
    cy = float(cents[best][1])
    row["nearest_center_pixel_distance"] = round(best_pixel_distance, 2)
    row["center_zone_pixels"] = best_center_zone_pixels

    # [END SECTOR: VIA_DETECTION_CORE]
    return {"cx": cx, "cy": cy, "area": best_area, "mask": blob}


# ============================================================================
# 결과 이미지
# ============================================================================
def _draw(src: np.ndarray, rows: List[Dict[str, Any]], numbering: bool) -> np.ndarray:
    """원본과 같은 해상도에 판정 마커를 그린다. 원본은 건드리지 않는다.

        PAD 판정 마커
          정상    : 초록 원
          VIA없음 : 빨강 X

        찾아낸 VIA (판정과 무관하게 항상)
          하늘색 원 + 중심점. 원 크기는 실제로 검출된 덩어리 크기입니다.
          어디를 VIA 로 봤는지 눈으로 바로 확인할 수 있습니다.
    """
    out = src.copy()
    for r in rows:
        px, py = int(round(r["pad_center"][0])), int(round(r["pad_center"][1]))
        rad = max(int(round(r["pad_radius"])), 3)
        st = r["status"]

        if st == "OK":
            cv2.circle(out, (px, py), rad, COLOR_OK, 1, cv2.LINE_AA)
            color = COLOR_OK
        elif st == "VIA_MISSING":
            cv2.drawMarker(out, (px, py), COLOR_MISSING, cv2.MARKER_TILTED_CROSS,
                           max(rad * 2, 7), 1, cv2.LINE_AA)
            color = COLOR_MISSING
        else:
            # 현재 내부 상태는 OK/VIA_MISSING뿐이지만 예외 상태도 빨간 X로 보입니다.
            cv2.drawMarker(out, (px, py), COLOR_MISSING, cv2.MARKER_TILTED_CROSS,
                           max(rad * 2, 7), 1, cv2.LINE_AA)
            color = COLOR_MISSING

        # 찾아낸 VIA를 실제 크기의 하늘색 원으로 표기합니다.
        if r["via_center"] is not None:
            vx = int(round(r["via_center"][0]))
            vy = int(round(r["via_center"][1]))
            vr = max(int(round(np.sqrt(max(r["via_area"], 1) / np.pi))), 2)
            cv2.circle(out, (vx, vy), vr, COLOR_VIA, 1, cv2.LINE_AA)
            cv2.circle(out, (vx, vy), 0, COLOR_VIA, -1)   # 중심점 1px

        if numbering:
            cv2.putText(out, str(r["pad_id"]), (px + rad + 1, py - rad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return out


# ============================================================================
# 디버깅용 표 출력
# ============================================================================
def _print_table(code: str, rows: List[Dict[str, Any]]) -> None:
    count: Dict[str, int] = {}
    for r in rows:
        count[r["status"]] = count.get(r["status"], 0) + 1
    tally = "  ".join("%s=%d" % (k, count[k]) for k in sorted(count))

    print("code=%s   검사대상 %d개   %s" % (code, len(rows), tally))
    if not rows:
        return

    head = ("PAD", "판정", "PAD중심", "VIA중심", "중심거리", "중심비",
            "중앙허용", "근접픽셀", "중앙px", "VIA면적", "PAD평균", "이진임계",
            "어두운px", "정합이동")
    print("%4s %-12s %-16s %-16s %8s %8s %8s %8s %7s %8s %8s %8s %9s %12s" % head)
    print("-" * 154)

    def pt(v):
        return "-" if v is None else "(%.1f,%.1f)" % (v[0], v[1])

    def num(v, f="%.2f"):
        return "-" if v is None else f % v

    for r in rows:
        flag = "NG" if r["status"] == "VIA_MISSING" else "  "
        print("%4d %-12s %-16s %-16s %8s %8s %8s %8s %7s %8s %8s %8s %9s %12s %s" % (
            r["pad_id"], r["status"], pt(r["pad_center"]), pt(r["via_center"]),
            num(r["offset_px"]), num(r["offset_norm"], "%.3f"),
            num(r["search_radius"]), num(r["nearest_center_pixel_distance"]),
            "%d" % r["center_zone_pixels"], num(r["via_area"], "%d"),
            num(r["pad_mean"], "%.1f"), num(r["dark_threshold"], "%.1f"),
            "%d" % r["dark_candidate_pixels"], pt(r["align_shift"]), flag))
