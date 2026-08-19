# -*- coding: utf-8 -*-
"""via_checker.py - 중앙의 검정/짙은 갈색 VIA 검사 (단일 파일, 복붙용)

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
* VIA 색은 검정 또는 짙고 채도가 있는 갈색으로 한정합니다.
* PAD 중앙 원형 구역에서만 VIA를 찾습니다. PAD 외곽의 검은 선, 테두리,
  반사 얼룩은 검색 대상에서 먼저 제외합니다.
* 쏠림(VIA_OFFSET)은 불량으로 판정하지 않습니다. 중앙 검색 구역 안에서
  검출된 VIA는 중심 오차가 있더라도 OK입니다. 기존 호출부 호환을 위해
  CODE_VIA_OFFSET 상수와 거리 진단값은 남지만 기본 설정에서는 code "99"를
  반환하지 않습니다.

검출 순서
---------
1. VIA 설계도에 점이 있는 PAD만 검사합니다.
2. 설계 PAD를 실측 PAD와 국소 정합한 뒤 그 중심과 등가반지름을 구합니다.
3. 중심에서 ``PAD반지름 * VIA_CENTER_SEARCH_RATIO`` 이내만 검색합니다.
4. HSV 색공간에서 검정 또는 짙은 갈색인 화소만 남깁니다.
   검정은 V(밝기) 상한, 갈색은 H(색상)+S(채도)+V(밝기) 조건으로 구분합니다.
5. PAD 바탕보다 최소 ``VIA_MIN_VALUE_DROP`` 만큼 어두운 화소만 남깁니다.
6. Black-hat으로 주변보다 국소적으로 어두운 화소만 남깁니다. 밝은 이물을
   잡던 Top-hat은 사용하지 않습니다.
7. 연결성분의 면적, 원형도, 채움비 조건을 모두 통과한 가장 큰 덩어리를
   VIA로 선택합니다. 원형 조건에서 탈락한 선을 되살리는 우회도 없습니다.

사용자가 수정할 곳
------------------
* ``[SECTOR: VIA_DETECTION_CONFIG]``: 중앙 범위와 색/모양 임계값
* ``[SECTOR: VIA_DETECTION_CORE]``: 실제 후보 마스크를 만드는 순서

``dark_offset``은 기존 호출 호환용입니다. 양수면 검정/갈색 밝기 상한이 올라가고
PAD보다 어두워야 하는 최소 차이는 줄어 더 민감해집니다. 음수면 반대로 더
엄격해집니다. 기본값 0부터 5~10 단위로 조절하세요.

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
CODE_VIA_OFFSET = "99"   # 호환용: 기본 설정에서는 반환하지 않음
CODE_ERROR = "-1"        # 입력 오류

# 호환 모드를 켰을 때 둘 다 있으면 결손이 위치오차보다 우선합니다.
CODE_PRIORITY = [CODE_VIA_MISSING, CODE_VIA_OFFSET]


# ============================================================================
# [SECTOR: VIA_DETECTION_CONFIG]
# VIA 검출 튜닝 값 - 사용자가 가장 먼저 수정할 곳
# ============================================================================

# 설계 PAD 가 실물보다 몇 px 작게 그려졌는지. 그 만큼 되돌려서 공칭 크기로 씁니다.
# 설계도를 실물과 같은 크기로 그렸으면 0.
DESIGN_PAD_SHRINK = 2

# 쏠림 불량은 기본적으로 사용하지 않습니다. 검출 중심의 거리값은 debug_via에
# 진단용으로 계속 남깁니다. 예전 동작이 꼭 필요할 때만 True로 바꾸세요.
VIA_OFFSET_DEFECT_ENABLED = False
# 아래 두 값은 VIA_OFFSET_DEFECT_ENABLED=True일 때만 사용됩니다.
OFFSET_TOL = 0.30
OFFSET_MIN_PX = 2.2

# PAD 중앙 검색 반지름 = PAD 등가반지름 * 이 값.
# 0.50이면 PAD의 가운데 절반 반지름 안에서만 찾습니다. 외곽의 검은 선이 계속
# 들어오면 낮추고, 정상 VIA 가장자리가 잘리면 0.05씩 올리세요.
VIA_CENTER_SEARCH_RATIO = 0.50

# 국소 정합. 중앙 검색원이 실물 PAD 중앙과 맞도록 설계 PAD를 조금 이동합니다.
# 0이면 정합하지 않습니다.
ALIGN_MAX_RATIO = 0.40    # 최대 이동량 = PAD반지름 * 이 값
ALIGN_MARGIN = 2          # 무게중심을 잴 때 설계 PAD 를 몇 px 키운 창을 볼지
ALIGN_MIN_COVER = 0.30    # 창 안 실측 픽셀이 이 비율도 안 되면 정합 생략

# VIA 색 조건. OpenCV HSV에서 H는 0~179, S/V는 0~255입니다.
#
# 검정: V가 절대 상한 이하.
# 갈색: H가 갈색 범위이고, S가 충분히 높고, V가 갈색 상한 이하.
# 두 경우 모두 PAD 바탕의 V 중앙값보다 VIA_MIN_VALUE_DROP 이상 어두워야 합니다.
# 따라서 밝은 반사, 파랑/초록/빨강 이물은 색이 PAD와 달라도 VIA가 아닙니다.
VIA_BLACK_VALUE_MAX = 85
VIA_BROWN_HUE_MIN = 3
VIA_BROWN_HUE_MAX = 28
VIA_BROWN_SAT_MIN = 55
VIA_BROWN_VALUE_MAX = 165
VIA_MIN_VALUE_DROP = 15

# VIA 모양 조건 - VIA 는 항상 원형이므로 원이 아닌 덩어리는 후보에서 뺍니다.
# 둘 다 크기와 무관한 무차원 값이라 이미지 배율이 바뀌어도 그대로 씁니다.
#
#   장단비   = 외접 사각형의 긴변 / 짧은변          (원 = 1.0)
#   반경편차 = 중심까지 거리의 표준편차 / 등가반지름 (꽉 찬 원 = 0.236, 크기 무관)
#
# 실측 근거 (테스트셋 VIA 746개 / 실물 노이즈 이미지)
#   진짜 VIA  : 장단비 1.00~1.75   반경편차 0.211~0.317
#   실물 노이즈: 장단비 4.0~7.0    반경편차 0.443~0.690   <- 1px 폭 긁힘 자국
# 두 분포 사이가 넓게 비어 있어서 그 가운데에 선을 그었습니다.
VIA_MAX_ASPECT = 2.2        # 이보다 길쭉하면 탈락
VIA_MAX_RADIAL_DEV = 0.45   # 이보다 흩어져 있으면 탈락 (ㄴ자·대각선·부스러기)

# Black-hat: 주변보다 국소적으로 어두운 부분만 남깁니다.
# 밝은 이물을 잡는 Top-hat은 VIA 색 전제와 맞지 않아 사용하지 않습니다.
BLACKHAT_KSIZE_RATIO = 0.85   # 커널 크기 = PAD반지름 * 이 값 (VIA 보다 크고 PAD 보다 작게)
BLACKHAT_MIN = 12.0           # 응답 절대 하한
BLACKHAT_RATIO = 0.18         # 응답 상대 하한 (PAD 밝기 중앙값 * 이 값)

# VIA 로 인정할 덩어리 크기
VIA_MIN_AREA = 4              # 최소 px
VIA_MAX_AREA_RATIO = 0.25     # 최대 = PAD 면적 * 이 값

# 탐색 영역을 PAD 안쪽으로 몇 px 깎을지.
# 이 값을 키워서 PAD 테두리 오검출을 막으려 하면 안 됩니다.
# 연한 VIA 는 화소가 4~5개뿐이라 조금만 깎아도 통째로 사라집니다.
# 테두리 문제는 아래 VIA_MIN_CLEARANCE 가 담당합니다.
PAD_ERODE = 1

# VIA 로 인정할 최소 경계 여유 = PAD반지름 * 이 값.
# 덩어리 안에서 PAD 경계로부터 가장 먼 픽셀이 이 거리보다 가까우면 VIA 가 아닙니다.
# VIA 는 PAD 안쪽에 뚫린 구멍이라 항상 경계에서 떨어져 있는 반면,
# PAD 테두리를 한 바퀴 두른 어두운 링은 전부 경계에 붙어 있습니다.
#
# 이 조건이 꼭 필요한 이유 : 테두리 링은 장단비 1.0 / 반경편차 0.1~0.2 라
# 모양만 보면 '완벽한 원'으로 보이고, 무게중심도 PAD 중심과 겹쳐 편심이 0 이
# 됩니다. 즉 VIA 가 아예 없는 PAD 가 OK 로 둔갑합니다(불량 놓침).
#
# 실측 (PAD 반지름으로 나눈 값, 덩어리 최대 경계거리)
#   진짜 VIA      최소 0.536  중앙 0.965   (테스트셋 746개)
#   테두리 링     중앙 0.254  p1  0.086    (실물 이미지 158개)
VIA_MIN_CLEARANCE = 0.30

# VIA 로 인정할 최소 채움비 = 덩어리 면적 / 그 덩어리의 최소외접원 면적.
# 꽉 찬 원은 0.6~0.9 가 나오고, 테두리를 두른 링이나 호(弧)는 0.2 아래로 떨어집니다.
#
# 위의 VIA_MIN_CLEARANCE 와 같은 문제(테두리 오검출)를 다른 각도에서 막습니다.
# 경계 여유는 '덩어리 안에서 가장 깊은 한 점' 만 보기 때문에, 테두리 전이대가
# 두껍게 잡히면 그 한 점이 임계를 아슬아슬하게 넘어 통과해 버립니다
# (실측 : 경계여유 8.00 vs 임계 7.71 로 통과한 링이 OK 로 둔갑).
# 채움비는 덩어리 '전체 모양' 을 보므로 이런 링을 확실히 걸러냅니다.
# 이미지 오른쪽 끝에서 잘려 열린 C 자가 되어도 최소외접원은 그대로 커서 통과 못 합니다.
#
# 실측 (실물 이미지 49장, PAD 1016개에서 VIA 로 판정된 수)
#   채움비 없음 187 -> 0.40 일 때 9 -> 0.45 일 때 7 (테스트셋 성적은 그대로)
VIA_MIN_FILL = 0.45

# 실물 PAD 존재 판정. 커버리지가 이 값 미만이면 PAD 없음.
# 커버리지는 full 과 excl 중 큰 값입니다 (아래 VIA_EXCLUDE_RATIO 설명 참고).
PAD_PRESENT_MIN = 0.55

# excl 커버리지에서 제외할 원의 반지름 = PAD반지름 * 이 값.
# 실물 VIA 는 이진화 마스크에 구멍으로 뚫리므로, VIA 가 있어야 할 자리를 빼고 재면
# 구멍 크기와 무관하게 PAD 존재 여부를 판정할 수 있습니다 (구멍 메우기 불필요).
VIA_EXCLUDE_RATIO = 0.75

# 설계도 잡티 제거용 최소 px. 검사 대상은 "VIA 설계도에 점이 있는 PAD" 로만 정해지므로
# 여기서는 1~3px 짜리 노이즈만 걸러내면 됩니다. 값이 작아 스케일이 바뀌어도 안전합니다.
MIN_PAD_AREA = 4
MIN_VIA_AREA = 1              # VIA 설계 점의 최소 px

# 결과 이미지 마커 색 (BGR)
COLOR_OK = (0, 220, 0)          # 초록 : 정상
COLOR_OFFSET = (0, 165, 255)    # 주황 : 쏠림
COLOR_MISSING = (0, 0, 255)     # 빨강 : VIA 없음
COLOR_ABSENT = (150, 150, 150)  # 회색 : PAD 없음
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

    기본 설정의 code는 "1"(양품) / "42"(중앙 VIA 없음) / "-1"(입력 오류)입니다.
    VIA_OFFSET_DEFECT_ENABLED=True로 되돌린 경우에만 호환 코드 "99"도 나옵니다.
    via_bin 은 검출한 VIA 만 흰색인 0/255 마스크입니다 (원본과 같은 해상도).
    "-1" 이면 result 와 via_bin 은 None 이고, 이유가 표준에러로 출력됩니다.

    dark_offset : 검정/갈색으로 허용할 V(밝기) 상한을 조절합니다 (기본 0).

        양수  더 밝은 검정/갈색까지 허용 -> 민감
        음수  더 어두운 검정/갈색만 허용 -> 엄격

    5~10 단위로 움직이면서 debug_via의 black_value_max,
    brown_value_max, color_candidate_pixels 값을 함께 확인하세요.
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
        pad_median, dark_threshold, search_radius, pad_value_median,
        black_value_max, brown_value_max, brown_sat_min, color_candidate_pixels,
        via_aspect, via_radial_dev, shape_rejected, edge_rejected
      (해당 없는 항목은 None)

    align_shift 가 크게 나오면 설계도와 실물이 그만큼 어긋나 있다는 뜻입니다.
    shape_rejected가 0이 아니면 선/비원형 후보를 탈락시킨 수입니다.
    edge_rejected가 0이 아니면 경계 여유·채움비에서 탈락한 수입니다.
    color_candidate_pixels는 중앙 검색영역에서 색 조건을 통과한 픽셀 수입니다.
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

    # 검정/갈색을 HSV로 구분하므로 흑백이 아니라 BGR 컬러를 넘깁니다.
    blur = cv2.GaussianBlur(src, (3, 3), 0)

    pad_label, boxes = _label_pads(pdes)
    design_vias = _map_vias(vdes, pad_label)

    dil = None
    if DESIGN_PAD_SHRINK > 0:
        d = DESIGN_PAD_SHRINK
        dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    # 침식은 1px로 얕게만 합니다. 외곽 검은 선은 침식량을 키우는 대신
    # VIA_CENTER_SEARCH_RATIO의 중앙 검색원으로 제외합니다.
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
            "dark_threshold": None,
            "search_radius": None,
            "pad_value_median": None,
            "black_value_max": None,
            "brown_value_max": None,
            "brown_sat_min": VIA_BROWN_SAT_MIN,
            "color_candidate_pixels": 0,
            "via_aspect": None,
            "via_radial_dev": None,
            "shape_rejected": 0,
            "edge_rejected": 0,
        }

        # ---- 실물 PAD 가 그 자리에 있는지 ----
        row["pad_coverage"] = round(
            _coverage(shape, act, area, radius, dvx - x0, dvy - y0), 3)
        if row["pad_coverage"] < PAD_PRESENT_MIN:
            # PAD 가 없으면 VIA 가 없는 게 당연하므로 VIA 불량으로 세지 않는다.
            row["status"] = "PAD_ABSENT"
            rows.append(row)
            continue

        # ---- VIA 찾기: 중앙의 검정/짙은 갈색만 허용 ----
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
        row["via_aspect"] = round(found["aspect"], 2)
        row["via_radial_dev"] = round(found["radial_dev"], 3)
        row["offset_px"] = round(dist, 2)
        row["offset_norm"] = round(dist / radius if radius > 1e-6 else 999.0, 4)

        # 기본값에서는 쏠림을 불량으로 보지 않습니다. 중앙 검색영역 안에서 색과
        # 모양 조건을 통과해 검출됐으면 OK입니다. 아래 호환 스위치를 사용자가
        # 명시적으로 켠 경우에만 예전 VIA_OFFSET 판정을 복원합니다.
        if (VIA_OFFSET_DEFECT_ENABLED and
                row["offset_norm"] > OFFSET_TOL and dist > OFFSET_MIN_PX):
            row["status"] = "VIA_OFFSET"
            found_status.add(CODE_VIA_OFFSET)
        else:
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


def _coverage(shape: np.ndarray,
              act: np.ndarray,
              area: float,
              radius: float,
              vx: float,
              vy: float) -> float:
    """실측 마스크가 설계 PAD 를 얼마나 덮는지. 두 방식 중 큰 값을 쓴다.

      full : 설계 PAD 전체 대비        -> 실물 PAD 가 설계보다 작을 때 강함
      excl : VIA 자리를 뺀 영역 대비   -> VIA 구멍이 뚫려 있을 때 강함

    실물 이미지에서는 VIA 가 이진화 마스크에 구멍으로 남는다. full 만 보면 그 구멍
    때문에 멀쩡한 PAD 가 PAD_ABSENT 로 빠지므로, VIA 가 있어야 할 자리를 제외하고
    한 번 더 잰다. 구멍을 메우지 않고도 구멍의 영향을 없앨 수 있다.
    """
    hit = (shape > 0) & (act > 0)
    full = float(np.count_nonzero(hit)) / area if area > 0 else 0.0

    ex = np.zeros(shape.shape, np.uint8)
    cv2.circle(ex, (int(round(vx)), int(round(vy))),
               max(2, int(round(radius * VIA_EXCLUDE_RATIO))), 255, -1)
    ref = (shape > 0) & (ex == 0)
    n = np.count_nonzero(ref)
    excl = float(np.count_nonzero(ref & hit)) / n if n else 0.0

    return max(full, excl)


def _roundness(ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    """덩어리가 얼마나 '꽉 찬 원' 에 가까운지 두 숫자로 잰다.

    반환 (장단비, 반경편차). 둘 다 크기와 무관한 무차원 값이라
    이미지 배율이 바뀌어도 같은 임계값을 그대로 쓸 수 있다.

      장단비   = 외접 사각형 긴변 / 짧은변
                 원이면 1.0, 길쭉할수록 커진다. 긁힘·선 자국을 잡는다.

      반경편차 = 중심까지 거리의 표준편차 / 등가반지름
                 꽉 찬 원이면 크기와 상관없이 항상 0.236 이다.
                 ㄴ자, 대각선, 흩어진 부스러기처럼 '외접 사각형은 정사각인데
                 속이 안 찬' 모양을 잡는다. 장단비가 못 잡는 것을 담당한다.

    (링 모양은 반경편차가 오히려 작아서 이 두 값으로는 안 걸러진다.
     링은 아래 _fill_ratio 가 담당한다.)
    """
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    aspect = max(w, h) / min(w, h)

    d = np.hypot(xs - xs.mean(), ys - ys.mean())
    equiv_r = np.sqrt(len(xs) / np.pi)
    dev = float(d.std() / equiv_r) if equiv_r > 1e-6 else 0.0
    return aspect, dev


def _fill_ratio(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """덩어리가 자기 최소외접원을 얼마나 채우는지. 크기와 무관한 무차원 값이다.

    꽉 찬 원은 0.6~0.9, PAD 테두리를 두른 링이나 호(弧)는 0.2 아래로 떨어진다.
    _roundness 의 두 값이 링을 '완벽한 원'으로 보는 맹점을 여기서 막는다.
    링이 이미지 끝에서 잘려 열린 C 자가 되어도 최소외접원은 그대로 크므로
    이 값은 여전히 낮게 나온다 (윤곽을 채워 보는 방식은 이 경우에 뚫린다).
    """
    _, r = cv2.minEnclosingCircle(np.stack([xs, ys], 1).astype(np.float32))
    return float(area) / (np.pi * r * r) if r > 0.5 else 1.0


def _find_via(roi: np.ndarray,
              shape: np.ndarray,
              center_x: float,
              center_y: float,
              radius: float,
              ero: np.ndarray,
              row: Dict[str, Any],
              dark_offset: float) -> Optional[Dict[str, Any]]:
    """PAD 중앙에서 검정 또는 짙은 갈색 VIA를 찾고, 없으면 None을 반환한다.

    반환 dict의 mask는 실제 판정에 사용한 연결성분만 True인 배열입니다.
    아래 ``[SECTOR: VIA_DETECTION_CORE]``가 사용자가 직접 바꿀 검출 본체입니다.

    핵심은 순서입니다. 먼저 중앙 검색원으로 외곽을 잘라낸 다음 색 조건을
    적용하므로, PAD 외곽의 검은 선은 색이 완전히 검정이어도 후보가 될 수 없습니다.
    그 뒤 Black-hat, 면적, 경계 여유, 채움비, 원형도를 모두 통과해야 합니다.
    """
    inner = cv2.erode(shape, ero)
    if np.count_nonzero(inner) < VIA_MIN_AREA * 4:
        return None

    # [SECTOR: VIA_DETECTION_CORE]
    # 1) 중앙 검색영역
    # PAD 등가반지름의 VIA_CENTER_SEARCH_RATIO 안쪽만 True입니다. 외곽 선/테두리는
    # 이 단계에서 제거됩니다. 검색 반지름은 debug_via의 search_radius에 기록됩니다.
    search_radius = max(float(radius * VIA_CENTER_SEARCH_RATIO), 1.0)
    yy, xx = np.ogrid[:shape.shape[0], :shape.shape[1]]
    center_mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= search_radius ** 2)
    search = (inner > 0) & center_mask
    row["search_radius"] = round(search_radius, 2)
    if np.count_nonzero(search) < VIA_MIN_AREA:
        return None

    # 각 화소가 PAD 경계에서 얼마나 떨어져 있는지. 뒤의 안전 필터에서 사용합니다.
    dist = cv2.distanceTransform((shape > 0).astype(np.uint8), cv2.DIST_L2, 5)
    min_clr = radius * VIA_MIN_CLEARANCE

    # 2) 검정/짙은 갈색 색상 조건
    # HSV의 V가 밝기입니다. PAD 기준 밝기는 VIA가 있을 수 있는 중앙을 제외한
    # 영역에서 구해 VIA 자체가 기준값을 어둡게 끌어내리지 못하게 합니다.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv)
    pad_mask = inner > 0
    reference = pad_mask & ~center_mask
    if np.count_nonzero(reference) < max(VIA_MIN_AREA * 4, 12):
        reference = pad_mask
    pad_value = float(np.median(value[reference]))

    black_max = float(np.clip(VIA_BLACK_VALUE_MAX + dark_offset, 0, 255))
    brown_max = float(np.clip(VIA_BROWN_VALUE_MAX + dark_offset, 0, 255))
    min_drop = max(float(VIA_MIN_VALUE_DROP - dark_offset), 0.0)

    is_black = value.astype(np.float32) <= black_max
    is_brown = ((hue >= VIA_BROWN_HUE_MIN) &
                (hue <= VIA_BROWN_HUE_MAX) &
                (sat >= VIA_BROWN_SAT_MIN) &
                (value.astype(np.float32) <= brown_max))
    darker_than_pad = value.astype(np.float32) <= (pad_value - min_drop)
    color_candidate = (is_black | is_brown) & darker_than_pad & search

    row["pad_median"] = round(pad_value, 1)       # 예전 키 호환
    row["dark_threshold"] = round(black_max, 1)   # 예전 키 호환
    row["pad_value_median"] = round(pad_value, 1)
    row["black_value_max"] = round(black_max, 1)
    row["brown_value_max"] = round(brown_max, 1)
    row["color_candidate_pixels"] = int(np.count_nonzero(color_candidate))

    # 3) Black-hat: 주변보다 국소적으로 어두운지 확인
    # PAD 바깥을 PAD 밝기로 메워 외곽의 어두운 배경이 morphology에 섞이지 않게 합니다.
    # 밝은 물체를 찾는 Top-hat은 의도적으로 쓰지 않습니다.
    filled = np.where(shape > 0, value, np.uint8(round(pad_value)))
    ks = int(np.clip(int(round(radius * BLACKHAT_KSIZE_RATIO)) | 1, 5, 31))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bh = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    bh_thr = max(BLACKHAT_MIN, pad_value * BLACKHAT_RATIO)

    cand = (color_candidate & (bh > bh_thr)).astype(np.uint8)
    # MORPH_OPEN은 작은 정상 VIA를 지울 수 있어 사용하지 않습니다. 잡티 제거는 아래
    # 연결성분 면적/원형도/채움비 조건이 담당합니다.

    num, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    max_area = max(int(np.count_nonzero(shape) * VIA_MAX_AREA_RATIO), VIA_MIN_AREA)

    # 4) 연결성분 안전 필터
    # 면적 -> 경계 여유 -> 채움비 -> 원형도 순서로 검사합니다. 외곽 검은 선처럼
    # 길쭉한 후보는 장단비/반경편차에서 탈락하며, 예전처럼 다시 되살리지 않습니다.
    best, best_area, best_shape = -1, 0, (0.0, 0.0)
    rejected = 0
    edge_rejected = 0
    for i in range(1, num):
        a = int(stats[i, 4])
        if not (VIA_MIN_AREA <= a <= max_area):
            continue
        ys, xs = np.nonzero(lab == i)

        if float(dist[ys, xs].max()) < min_clr:
            edge_rejected += 1
            continue
        if _fill_ratio(ys, xs, a) < VIA_MIN_FILL:
            edge_rejected += 1
            continue

        aspect, dev = _roundness(ys, xs)
        if aspect > VIA_MAX_ASPECT or dev > VIA_MAX_RADIAL_DEV:
            rejected += 1
            continue
        if a > best_area:
            best, best_area, best_shape = i, a, (aspect, dev)

    row["shape_rejected"] = rejected
    row["edge_rejected"] = edge_rejected
    if best < 0:
        return None

    blob = lab == best
    # 5) 어두운 정도를 가중치로 한 서브픽셀 중심. 거리값은 진단용일 뿐,
    # VIA_OFFSET_DEFECT_ENABLED=False에서는 판정 코드에 영향을 주지 않습니다.
    wgt = np.clip(pad_value - value.astype(np.float32), 0.0, None) * blob
    total = float(wgt.sum())
    if total > 1e-6:
        gy, gx = np.mgrid[0:shape.shape[0], 0:shape.shape[1]].astype(np.float32)
        cx = float((gx * wgt).sum() / total)
        cy = float((gy * wgt).sum() / total)
    else:
        cx, cy = float(cents[best][0]), float(cents[best][1])

    # [END SECTOR: VIA_DETECTION_CORE]
    return {"cx": cx, "cy": cy, "area": best_area, "mask": blob,
            "aspect": best_shape[0], "radial_dev": best_shape[1]}


# ============================================================================
# 결과 이미지
# ============================================================================
def _draw(src: np.ndarray, rows: List[Dict[str, Any]], numbering: bool) -> np.ndarray:
    """원본과 같은 해상도에 판정 마커를 그린다. 원본은 건드리지 않는다.

        PAD 판정 마커
          정상    : 초록 원
          쏠림    : 주황 원 + PAD중심 -> VIA중심 화살표
          VIA없음 : 빨강 X
          PAD없음 : 회색 원

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
        elif st == "VIA_OFFSET":
            cv2.circle(out, (px, py), rad, COLOR_OFFSET, 1, cv2.LINE_AA)
            vx = int(round(r["via_center"][0]))
            vy = int(round(r["via_center"][1]))
            cv2.arrowedLine(out, (px, py), (vx, vy), COLOR_OFFSET, 1,
                            cv2.LINE_AA, tipLength=0.35)
            color = COLOR_OFFSET
        elif st == "VIA_MISSING":
            cv2.drawMarker(out, (px, py), COLOR_MISSING, cv2.MARKER_TILTED_CROSS,
                           max(rad * 2, 7), 1, cv2.LINE_AA)
            color = COLOR_MISSING
        else:   # PAD_ABSENT
            cv2.circle(out, (px, py), rad, COLOR_ABSENT, 1, cv2.LINE_AA)
            color = COLOR_ABSENT

        # 찾아낸 VIA 를 실제 크기로 표기한다. 판정선(초록/주황)과 겹쳐도
        # 색이 달라 구분되고, 오검출이면 엉뚱한 자리에 원이 그려져 바로 보인다.
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

    head = ("PAD", "판정", "PAD중심", "VIA중심", "편심px", "편심비율",
            "중앙비", "VIA면적", "장단비", "반경편차", "모양탈락", "테두리탈락",
            "PAD_V", "검정V", "덮임", "정합이동")
    print("%4s %-12s %-16s %-16s %7s %9s %6s %8s %7s %9s %9s %10s %8s %6s %6s %12s" % head)
    print("-" * 162)

    def pt(v):
        return "-" if v is None else "(%.1f,%.1f)" % (v[0], v[1])

    def num(v, f="%.2f"):
        return "-" if v is None else f % v

    for r in rows:
        flag = "NG" if r["status"] in ("VIA_OFFSET", "VIA_MISSING") else "  "
        rej = r["shape_rejected"]
        erej = r["edge_rejected"]
        print("%4d %-12s %-16s %-16s %7s %9s %6s %8s %7s %9s %9s %10s %8s %6s %6s %12s %s" % (
            r["pad_id"], r["status"], pt(r["pad_center"]), pt(r["via_center"]),
            num(r["offset_px"]), num(r["offset_norm"], "%.4f"),
            "%.2f" % VIA_CENTER_SEARCH_RATIO, num(r["via_area"], "%d"),
            num(r["via_aspect"]), num(r["via_radial_dev"], "%.3f"),
            "%d" % rej if rej else "-", "%d" % erej if erej else "-",
            num(r["pad_median"], "%.1f"), num(r["dark_threshold"], "%.1f"),
            num(r["pad_coverage"], "%.2f"), pt(r["align_shift"]), flag))
