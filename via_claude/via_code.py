# -*- coding: utf-8 -*-
"""via_code.py - 이미 이진화된 VIA 를 받아 코드와 결과 이미지를 돌려주는 모듈

VIA 이진화를 직접 하고 계신 경우를 위한 판정 전용 모듈입니다.
VIA 를 찾지 않습니다. 이미 찾아 놓은 것을 '판정만' 합니다.

    from via_code import check_via_code

    code, result = check_via_code(VIA이진화, PAD설계도, VIA설계도)

    #  code   : "1"  양품
    #           "42" VIA 없음
    #           "-1" 입력 오류 (이유는 표준에러로 나감. 이때 result 는 None)
    #  result : 판정 마커를 그린 BGR 이미지 (입력과 같은 해상도)

    cv2.imwrite("result.png", result)

왜 그렇게 판정했는지 표로도 보고 싶으면 verbose 를 켜세요.
반환값은 그대로 (코드, 결과이미지) 둘입니다.

    code, result = check_via_code(VIA이진화, PAD설계도, VIA설계도, verbose=True)

--------------------------------------------------------------------------
결과 이미지 배경 - image 를 주면 그 위에, 안 주면 합성 배경에 그립니다
--------------------------------------------------------------------------
    code, result = check_via_code(VIA이진화, PAD설계도, VIA설계도, image=원본)

  이 모듈은 원본 사진을 받지 않는 게 기본이라(판정만 하니까) 그릴 바탕이 없습니다.
  그래서 원본을 안 주면 받은 마스크들을 회색조로 겹쳐 바탕을 만듭니다.

      검정   아무것도 없는 곳
      진회색 PAD 설계도
      중간   실측 PAD (bin_mask 를 준 경우)
      밝은회 사용자가 준 VIA 이진화

  마커는 그 위에 원색으로 그리므로 어느 쪽이든 구분됩니다.
  원본을 손에 들고 계시면 image 로 주는 편이 눈으로 보기 좋습니다.
  판정 결과는 배경과 무관하게 완전히 같습니다 (image 는 그리기에만 씁니다).

--------------------------------------------------------------------------
via_checker.py 와 무엇이 다른가
--------------------------------------------------------------------------
  via_checker : 원본 이미지에서 VIA 를 '찾는' 것까지 합니다 (4장 입력)
  via_code    : VIA 를 이미 찾아 놓은 상태에서 '판정만' 합니다 (3장 입력)

  판정 기준(중심 거리·경계 여유·면적·채움비·모양)은 두 모듈이 똑같습니다.
  그래서 via_checker 가 준 via_bin 을 이 모듈에 그대로 넣으면 같은 코드가 나옵니다.
  테스트셋 44장 / PAD 783개로 확인했습니다.

    code, result, via_bin = check_via(원본, 이진화, PAD설계도, VIA설계도)
    code2, result2 = check_via_code(via_bin, PAD설계도, VIA설계도,
                                    bin_mask=이진화, image=원본)
    assert code2 == code          # 결과 이미지도 같은 마커로 그려집니다

  bin_mask 를 빼면 PAD 누락 구분이 꺼져서 5장이 갈립니다 (아래 설명 참고).

--------------------------------------------------------------------------
입력 3장 (+ 선택 2장)
--------------------------------------------------------------------------
  1) VIA 이진화  : 직접 만드신 VIA 마스크. 흰색 = VIA (ndarray 또는 파일 경로)
  2) PAD 설계도  : PAD 가 있어야 할 자리 (이진)
  3) VIA 설계도  : VIA 가 있어야 할 자리 (이진)

  선택) bin_mask : 실측 PAD 마스크. 주는 편이 낫습니다. 바로 아래 설명 참고.
  선택) image    : 원본 사진. 결과 이미지의 바탕으로만 씁니다. 판정에는 안 씁니다.

  모두 같은 해상도·같은 좌표계여야 합니다.
  크기가 다르면 조용히 리사이즈하지 않고 "-1" 로 실패합니다.

--------------------------------------------------------------------------
선택 인자 bin_mask - 되도록 같이 주세요
--------------------------------------------------------------------------
    code = check_via_code(VIA이진화, PAD설계도, VIA설계도, bin_mask=이진화)

  실측 PAD 마스크입니다. VIA 를 찾는 데는 안 씁니다(그건 이미 하셨으니까).
  '기준점' 을 잡는 데만 씁니다. 주면 두 가지가 켜집니다.

  1) 국소 정합
     설계도와 실물은 완벽히 겹치지 않습니다. PAD 중심이 2~3px 만 어긋나도,
     'VIA 는 PAD 중앙에만 있다' 는 조건(VIA_SEARCH_RATIO)의 기준점이 통째로
     밀려서 멀쩡한 VIA 가 범위 밖으로 잘려 나갑니다(= VIA 없음, 42).
     PAD 마다 설계 형상을 실물 쪽으로 조금 평행이동시켜 그 계통 오차를 걷어냅니다.
     42 가 유독 많이 나오면 이걸 먼저 의심하세요.

  2) PAD 누락 구분
     실물 PAD 자체가 없으면 VIA 도 당연히 없습니다. 이걸 VIA 없음(42)으로 세면
     PAD 누락 검사와 중복으로 잡힙니다. bin_mask 가 있으면 커버리지를 재서
     PAD 가 없는 자리는 판정에서 빼 둡니다.

  안 주면 둘 다 꺼집니다. 그때 42 는 'VIA 가 없다' 가 아니라
  'PAD 가 없거나 VIA 가 없다' 라는 뜻이 됩니다.
  실측으로 테스트셋 44장 중 5장이 이 차이 하나로 1 대신 42 가 나왔습니다.

--------------------------------------------------------------------------
검사 대상
--------------------------------------------------------------------------
  VIA 설계도에 점이 찍힌 PAD 만 검사합니다.
  설계상 VIA 가 없는 PAD 는 아예 건드리지 않습니다.

--------------------------------------------------------------------------
판정
--------------------------------------------------------------------------
  PAD 안에서 쓸 만한 VIA 덩어리를 하나 고를 수 있으면 양품입니다.

    VIA 를 하나도 못 고름  -> 42
    그 외                   -> 1
    실물 PAD 가 없음        -> 코드에 반영 안 함 (bin_mask 를 준 경우에만 구분됨)

  쏠림(편심)은 불량으로 세지 않습니다. VIA 가 PAD 안에 있기만 하면 양품입니다.
  과검이 너무 많아서 뺐습니다. 편심 수치 자체는 결과 표의 편심비율 열로
  계속 볼 수 있으니, 필요하면 호출한 쪽에서 직접 기준을 세우세요.

  덩어리를 고를 때 거르는 조건 다섯 가지입니다.

    면적      VIA_MIN_AREA ~ PAD면적*VIA_MAX_AREA_RATIO 밖이면 탈락
    중심 거리 PAD 중심에서 반지름*VIA_SEARCH_RATIO 보다 멀면 탈락 (되살리지 않음)
    경계 여유 PAD 경계에 붙어 있으면 탈락 (되살리지 않음)
    채움비    최소외접원을 못 채우면(링·호) 탈락 (되살리지 않음)
    모양      원이 아니면 뒤로 밀어둠 (원형 후보가 없으면 그냥 씀)

  앞의 넷은 '탈락' 이고 모양만 '선호' 입니다. 이유는 아래 상수 설명에 있습니다.

  '중심 거리' 는 PAD 외곽을 스치는 검은 선·이웃 패턴을 VIA 로 오해하지 않으려고
  둔 조건입니다. VIA 는 PAD 중앙에만 뚫리므로 위치만으로 확실히 갈립니다.

--------------------------------------------------------------------------
결과 이미지 색 읽는 법
--------------------------------------------------------------------------
  PAD 판정 (PAD 중심에 그립니다)
    초록 원   정상 (VIA 가 PAD 안에 있으면 위치와 무관하게 정상)
    빨강 X    VIA 없음
    회색 원   PAD 없음 (bin_mask 를 준 경우에만 나옵니다. 코드에는 반영 안 됨)

  고른 VIA (판정과 무관하게 항상 그립니다)
    하늘색 원 + 중심점. 원 크기는 실제로 고른 덩어리 크기입니다.
    엉뚱한 자리에 하늘색 원이 있으면 주신 VIA 이진화가 잘못된 것입니다.

의존성 : numpy, opencv-python  (Python 3.9+)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["check_via_code", "CODE_OK", "CODE_VIA_MISSING", "CODE_ERROR"]


# ============================================================================
# 결과 코드
# ============================================================================
CODE_OK = "1"            # 양품
CODE_VIA_MISSING = "42"  # VIA 없음
CODE_ERROR = "-1"        # 입력 오류

# 쏠림(편심) 코드 "99" 는 없앴습니다. 과검이 너무 많았습니다.
# VIA 가 PAD 안에서 발견되기만 하면 위치와 무관하게 양품입니다.
# 편심 수치 자체는 rows 의 offset_px / offset_norm 으로 계속 볼 수 있으니,
# 필요하면 호출한 쪽에서 직접 기준을 세우세요.
# 코드가 하나뿐이라 우선순위(CODE_PRIORITY) 도 필요 없어졌습니다.


# ============================================================================
# 튜닝 값 - via_checker.py 와 같은 값입니다. 한쪽만 고치면 두 모듈 판정이 갈립니다.
# ============================================================================

# 설계 PAD 가 실물보다 몇 px 작게 그려졌는지. 그 만큼 되돌려서 공칭 크기로 씁니다.
# 설계도를 실물과 같은 크기로 그렸으면 0.
DESIGN_PAD_SHRINK = 2

# 국소 정합 (bin_mask 를 준 경우에만 동작). 0 이면 정합 안 함.
# 편심을 판정하지 않게 된 뒤에도 정합은 그대로 필요합니다. 아래 VIA_SEARCH_RATIO
# 가 'PAD 중심에서 얼마나 떨어졌나' 를 보는데, 그 중심이 정합 결과이기 때문입니다.
ALIGN_MAX_RATIO = 0.40    # 최대 이동량 = PAD반지름 * 이 값
ALIGN_MARGIN = 2          # 무게중심을 잴 때 설계 PAD 를 몇 px 키운 창을 볼지
ALIGN_MIN_COVER = 0.30    # 창 안 실측 픽셀이 이 비율도 안 되면 정합 생략

# VIA 로 인정할 덩어리 크기
VIA_MIN_AREA = 4              # 최소 px
VIA_MAX_AREA_RATIO = 0.25     # 최대 = PAD 면적 * 이 값

# VIA 를 찾을 범위 = PAD 중심에서 PAD반지름 * 이 값 이내.
# 덩어리 중심이 이 원 밖이면 무조건 탈락입니다 (되살리지 않음).
#
# 이 조건이 필요한 이유 : VIA 는 PAD 중앙에만 뚫립니다. 반대로 PAD 외곽을
# 스치는 검은 선, 이웃 패턴, 테두리 링은 크기·모양 조건을 다 통과해 버리는
# 경우가 있는데 공통적으로 PAD 중심에서 멉니다. 위치 하나로 잘라내는 것이
# 가장 확실합니다. 이진화를 직접 주시는 경우에도 같은 문제가 생깁니다.
#
# 실측 (덩어리 중심 - PAD 중심 거리를 PAD 반지름으로 나눈 값)
#   진짜 VIA   중앙 0.03  p95 0.457  p99 0.596  최대 0.661   (테스트셋 746개)
#   외곽 잡음  p1   0.771  p5  0.859               (VIA 가 없는 실물 191개)
# 둘 사이가 0.66 ~ 0.77 로 비어 있어 그 안에 선을 그었습니다.
# 몇 개가 잘렸는지는 rows 의 far_rejected 에 찍힙니다.
VIA_SEARCH_RATIO = 0.75

# VIA 로 인정할 최소 경계 여유 = PAD반지름 * 이 값.
# 덩어리 안에서 PAD 경계로부터 가장 먼 픽셀이 이 거리보다 가까우면 VIA 가 아닙니다.
#
# 이 조건이 꼭 필요한 이유 : PAD 테두리를 한 바퀴 두른 어두운 링은 장단비 1.0 /
# 반경편차 0.1~0.2 라 모양만 보면 '완벽한 원'이고, 무게중심도 PAD 중심과 겹쳐
# 편심이 0 이 됩니다. 즉 VIA 가 아예 없는 PAD 가 양품으로 둔갑합니다(불량 놓침).
# 모양으로는 못 잡고 위치로만 잡힙니다.
#
# 실측 (PAD 반지름으로 나눈 값, 덩어리 최대 경계거리)
#   진짜 VIA      최소 0.536  중앙 0.965   (테스트셋 746개)
#   테두리 링     중앙 0.254  p1  0.086    (실물 이미지 158개)
VIA_MIN_CLEARANCE = 0.30

# VIA 로 인정할 최소 채움비 = 덩어리 면적 / 그 덩어리의 최소외접원 면적.
# 꽉 찬 원은 0.6~0.9 가 나오고, 테두리를 두른 링이나 호(弧)는 0.2 아래로 떨어집니다.
#
# 경계 여유(위)와 목적은 같은데 보는 곳이 다릅니다. 경계 여유는 덩어리 안의
# '가장 깊은 한 점' 만 보기 때문에, 링이 두껍게 잡히면 그 한 점이 임계를 아슬아슬
# 하게 넘어 통과해 버립니다 (실측 : 8.00 vs 임계 7.71 로 통과). 채움비는 덩어리
# '전체 모양' 을 보므로 그 경우도 막습니다. 링이 이미지 끝에서 잘려 열린 C 자가
# 되어도 최소외접원은 그대로 크기 때문에 값이 낮게 유지됩니다.
VIA_MIN_FILL = 0.45

# VIA 모양 조건. 둘 다 크기와 무관한 무차원 값이라 배율이 바뀌어도 그대로 씁니다.
#   장단비   = 외접 사각형의 긴변 / 짧은변          (원 = 1.0)
#   반경편차 = 중심까지 거리의 표준편차 / 등가반지름 (꽉 찬 원 = 0.236, 크기 무관)
#
# 실측 : 진짜 VIA 장단비 1.00~1.75 반경편차 0.211~0.317
#        긁힘 노이즈 장단비 4.0~7.0  반경편차 0.443~0.690
VIA_MAX_ASPECT = 2.2
VIA_MAX_RADIAL_DEV = 0.45

# 실물 PAD 존재 판정 (bin_mask 를 준 경우에만 씁니다).
# 커버리지가 이 값 미만이면 PAD 가 없다고 보고 그 자리는 판정에서 뺍니다.
PAD_PRESENT_MIN = 0.55

# excl 커버리지에서 제외할 원의 반지름 = PAD반지름 * 이 값.
# 실물 VIA 는 이진화 마스크에 구멍으로 뚫리므로, VIA 가 있어야 할 자리를 빼고 재면
# 구멍 크기와 무관하게 PAD 존재 여부를 판정할 수 있습니다 (구멍 메우기 불필요).
VIA_EXCLUDE_RATIO = 0.75

# 설계도 잡티 제거용 최소 px.
MIN_PAD_AREA = 4
MIN_VIA_AREA = 1              # VIA 설계 점의 최소 px


# ============================================================================
# 결과 이미지 색 (BGR) - via_checker.py 와 같은 색입니다.
# ============================================================================
COLOR_OK = (0, 220, 0)          # 초록 : 정상
COLOR_MISSING = (0, 0, 255)     # 빨강 : VIA 없음
COLOR_ABSENT = (150, 150, 150)  # 회색 : PAD 없음
COLOR_VIA = (255, 255, 0)       # 하늘 : 고른 VIA 위치 (판정과 무관하게 항상 표기)

# 원본(image)을 안 줬을 때 마스크로 합성하는 배경의 밝기.
# 마커가 원색이라 배경은 무채색으로 낮게 깔아 둡니다.
BG_PAD_DESIGN = 45       # PAD 설계도
BG_PAD_ACTUAL = 85       # 실측 PAD (bin_mask)
BG_VIA_INPUT = 170       # 사용자가 준 VIA 이진화


# ============================================================================
# 공개 함수
# ============================================================================
def check_via_code(via_bin: Union[str, np.ndarray],
                   pad_design: Union[str, np.ndarray],
                   via_design: Union[str, np.ndarray],
                   bin_mask: Optional[Union[str, np.ndarray]] = None,
                   image: Optional[Union[str, np.ndarray]] = None,
                   verbose: bool = False
                   ) -> Tuple[str, Optional[np.ndarray]]:
    """이미 이진화된 VIA 마스크를 받아 (코드, 결과이미지) 를 돌려준다.

        code, result = check_via_code(VIA이진화, PAD설계도, VIA설계도)

    code   : "1"(양품) / "42"(VIA 없음) / "-1"(입력 오류)
             VIA 가 쏠려 있어도 PAD 안에서 찾아지기만 하면 양품("1")입니다.
    result : 판정 마커를 그린 BGR 이미지. 입력 오류일 때만 None 입니다.

    bin_mask : 실측 PAD 마스크. 주면 설계도-실물 어긋남을 국소 정합으로 흡수합니다.
               안 주면 어긋난 만큼 PAD 중심이 밀려서, 중앙 탐색 범위 밖으로
               잘려 나간 멀쩡한 VIA 가 42 로 나올 수 있습니다.
    image    : 결과 이미지의 바탕으로 쓸 원본. 판정에는 전혀 안 씁니다.
               안 주면 받은 마스크들을 회색조로 겹쳐 바탕을 만듭니다.
    verbose  : PAD 별 판정 근거를 표준출력에 찍고, 결과 이미지에 번호도 답니다.
    """
    try:
        vmask = _to_mask(via_bin, "VIA 이진화")
        pdes = _to_mask(pad_design, "PAD 설계도")
        vdes = _to_mask(via_design, "VIA 설계도")
        act = _to_mask(bin_mask, "이진화 이미지") if bin_mask is not None else None
        src = _to_bgr(image) if image is not None else None
    except ValueError as e:
        sys.stderr.write("[via_code] %s\n" % e)
        return CODE_ERROR, None

    H, W = vmask.shape[:2]
    pairs = [("PAD 설계도", pdes), ("VIA 설계도", vdes)]
    if act is not None:
        pairs.append(("이진화 이미지", act))
    if src is not None:
        pairs.append(("원본 이미지", src))
    for nm, m in pairs:
        if m.shape[:2] != (H, W):
            sys.stderr.write("[via_code] %s 크기%s가 VIA 이진화 크기%s와 다릅니다.\n"
                             % (nm, m.shape[:2], (H, W)))
            return CODE_ERROR, None

    rows = _judge(vmask, pdes, vdes, act)
    if verbose:
        _print_rows(rows)

    if src is None:
        src = _canvas(vmask, pdes, act)
    result = _draw(src, rows, numbering=verbose)

    # 불량 코드가 "42" 하나뿐이라 우선순위를 따질 것이 없다.
    found = set(r["status"] for r in rows)
    return (CODE_VIA_MISSING if "VIA_MISSING" in found else CODE_OK), result


# ============================================================================
# 이미지 읽기 (한글 경로 대응)
# ============================================================================
def _to_mask(src: Union[str, np.ndarray], name: str) -> np.ndarray:
    """무엇이 들어와도 0/255 uint8 단일채널 마스크로 맞춘다."""
    if isinstance(src, np.ndarray):
        m: Optional[np.ndarray] = src
    else:
        try:
            buf = np.fromfile(str(src), dtype=np.uint8)
        except OSError:
            buf = np.empty(0, np.uint8)
        # cv2.imread 는 비ASCII 경로에서 실패하므로 np.fromfile 로 우회한다.
        m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
    if m is None:
        raise ValueError("%s를 읽을 수 없습니다: %s" % (name, src))
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return np.where(m > 0, 255, 0).astype(np.uint8)


def _to_bgr(src: Union[str, np.ndarray]) -> np.ndarray:
    """결과 이미지 바탕용 원본을 3채널 BGR 로 맞춘다. 판정에는 쓰지 않는다."""
    if isinstance(src, np.ndarray):
        img: Optional[np.ndarray] = src
    else:
        try:
            buf = np.fromfile(str(src), dtype=np.uint8)
        except OSError:
            buf = np.empty(0, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise ValueError("원본 이미지를 읽을 수 없습니다: %s" % src)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# ============================================================================
# 설계도 해석
# ============================================================================
def _label_pads(pad_design: np.ndarray
                ) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]]]:
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
# 판정 본체
# ============================================================================
def _judge(vmask: np.ndarray,
           pdes: np.ndarray,
           vdes: np.ndarray,
           act: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    """PAD 별 판정 결과 목록을 만든다."""
    H, W = vmask.shape[:2]
    pad_label, boxes = _label_pads(pdes)
    design_vias = _map_vias(vdes, pad_label)

    dil = None
    if DESIGN_PAD_SHRINK > 0:
        d = DESIGN_PAD_SHRINK
        dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    alk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * ALIGN_MARGIN + 1, 2 * ALIGN_MARGIN + 1))

    rows: List[Dict[str, Any]] = []
    for pid in sorted(design_vias):
        dvx, dvy = design_vias[pid]
        x, y, w, h = boxes[pid]
        mg = DESIGN_PAD_SHRINK + 4
        x0, x1 = max(x - mg, 0), min(x + w + mg, W)
        y0, y1 = max(y - mg, 0), min(y + h + mg, H)

        # 기준 형상은 '설계 PAD' 를 쓴다. '정중앙' 의 정의 자체가 설계 중심이다.
        shape = ((pad_label[y0:y1, x0:x1] == pid).astype(np.uint8)) * 255
        if dil is not None:
            shape = cv2.dilate(shape, dil)   # 설계가 작게 그려진 만큼 공칭 크기로 복원

        area = float(np.count_nonzero(shape))
        if area <= 0:
            continue
        ys, xs = np.nonzero(shape)
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(area / np.pi))

        sub_act = None
        if act is not None:
            sub_act = act[y0:y1, x0:x1]
            shape, cx, cy = _align(shape, sub_act, cx, cy, radius, area, alk)

        row: Dict[str, Any] = {
            "pad_id": pid,
            "status": "VIA_MISSING",
            "pad_center": (round(cx + x0, 2), round(cy + y0, 2)),
            "pad_radius": round(radius, 2),
            "pad_coverage": None,
            "via_center": None,
            "via_area": None,
            "offset_px": None,
            "offset_norm": None,
            "edge_rejected": 0,
            "shape_rejected": 0,
            "far_rejected": 0,
        }

        # ---- 실물 PAD 가 그 자리에 있는지 (bin_mask 를 준 경우에만) ----
        if sub_act is not None:
            row["pad_coverage"] = round(
                _coverage(shape, sub_act, area, radius, dvx - x0, dvy - y0), 3)
            if row["pad_coverage"] < PAD_PRESENT_MIN:
                # PAD 가 없으면 VIA 가 없는 게 당연하므로 VIA 불량으로 세지 않는다.
                row["status"] = "PAD_ABSENT"
                rows.append(row)
                continue

        found = _pick_via(vmask[y0:y1, x0:x1], shape, cx, cy, radius, row)
        if found is None:
            rows.append(row)
            continue

        dist = float(np.hypot(found["cx"] - cx, found["cy"] - cy))
        row["via_center"] = (round(found["cx"] + x0, 2), round(found["cy"] + y0, 2))
        row["via_area"] = int(found["area"])
        row["offset_px"] = round(dist, 2)

        # 편심은 재서 보여주기만 하고 판정에는 쓰지 않는다.
        # PAD 안에서 VIA 를 찾았으면 위치와 무관하게 양품이다.
        row["offset_norm"] = round(dist / radius if radius > 1e-6 else 999.0, 4)
        row["status"] = "OK"
        rows.append(row)
    return rows


def _align(shape: np.ndarray,
           act: np.ndarray,
           cx: float,
           cy: float,
           radius: float,
           area: float,
           alk: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """설계 PAD 형상을 실측 마스크 쪽으로 조금 평행이동한다.

    쏠림을 판정하지 않게 된 뒤에도 이 단계는 그대로 필요하다.
    'VIA 는 PAD 중앙에만 있다'(VIA_SEARCH_RATIO)와 PAD 유무 판정(_coverage)이
    둘 다 PAD 중심을 기준점으로 쓰기 때문이다. 설계도와 실물이 2~3px 만
    어긋나도 PAD 반지름이 6px 수준이면 그 기준점이 통째로 밀려서, 멀쩡한 VIA 가
    탐색 범위 밖으로 잘려 나간다(= VIA 없음, 42). 그 계통 오차만 걷어낸다.

    무게중심을 쓰는 이유 :
      - VIA 가 한가운데면 구멍이 대칭이라 무게중심이 밀리지 않는다.
      - VIA 가 쏠려 있으면 무게중심이 반대편으로 밀리므로, 정합이 VIA 를
        따라가서 쏠린 위치를 '중앙' 으로 만들어 버리는 일이 생기지 않는다.
    이동량은 반지름의 ALIGN_MAX_RATIO 이내로 묶어 폭주를 막는다.
    """
    if ALIGN_MAX_RATIO <= 0:
        return shape, cx, cy

    loc = (cv2.dilate(shape, alk) > 0) & (act > 0)
    if np.count_nonzero(loc) < area * ALIGN_MIN_COVER:
        return shape, cx, cy

    ly, lx = np.nonzero(loc)
    lim = radius * ALIGN_MAX_RATIO
    sx = float(np.clip(float(lx.mean()) - cx, -lim, lim))
    sy = float(np.clip(float(ly.mean()) - cy, -lim, lim))
    if abs(sx) < 1e-3 and abs(sy) < 1e-3:
        return shape, cx, cy

    moved = cv2.warpAffine(shape, np.float32([[1, 0, sx], [0, 1, sy]]),
                           (shape.shape[1], shape.shape[0]), flags=cv2.INTER_NEAREST)
    return moved, cx + sx, cy + sy


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

    반환 (장단비, 반경편차). 둘 다 크기와 무관한 무차원 값이다.

      장단비   = 외접 사각형 긴변 / 짧은변. 원이면 1.0. 긁힘·선 자국을 잡는다.
      반경편차 = 중심까지 거리의 표준편차 / 등가반지름. 꽉 찬 원이면 항상 0.236.
                 ㄴ자·대각선·부스러기처럼 '외접 사각형은 정사각인데 속이 안 찬'
                 모양을 잡는다.

    (링 모양은 반경편차가 오히려 작아서 이 두 값으로는 안 걸러진다.
     그래서 링은 VIA_MIN_CLEARANCE 와 아래 _fill_ratio 로 따로 거른다.)
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


def _pick_via(sub: np.ndarray,
              shape: np.ndarray,
              pad_cx: float,
              pad_cy: float,
              radius: float,
              row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """PAD 한 개 안에서 사용자가 준 VIA 덩어리 중 하나를 고른다. 없으면 None.

    찾으면 {"cx", "cy", "area"} 를 준다.
    (pad_cx, pad_cy) 는 국소 정합까지 끝난 PAD 중심이다.

    거르는 순서와 성격이 다르다.
      면적      하드 탈락. 너무 작으면 잡티, 너무 크면 PAD 얼룩이다.
      중심 거리 하드 탈락. 되살리지 않는다. PAD 외곽 잡음을 막는다.
      경계 여유 하드 탈락. 되살리지 않는다.
      채움비    하드 탈락. 되살리지 않는다. 링·호를 막는다.
      모양      선호. 원형 후보가 하나도 없으면 모양을 포기하고 가장 큰 것을 쓴다.

    중심 거리는 row["far_rejected"] 에, 나머지 하드 탈락 두 가지(경계 여유·채움비)
    는 row["edge_rejected"] 에 함께 센다.
    """
    inside = (sub > 0) & (shape > 0)
    if not np.any(inside):
        return None

    # 각 화소가 PAD 경계에서 얼마나 떨어져 있는지. 테두리 링을 거를 때 쓴다.
    dist = cv2.distanceTransform((shape > 0).astype(np.uint8), cv2.DIST_L2, 5)
    min_clr = radius * VIA_MIN_CLEARANCE
    max_off = radius * VIA_SEARCH_RATIO      # VIA 는 PAD 중앙에만 있다

    num, lab, stats, cents = cv2.connectedComponentsWithStats(
        inside.astype(np.uint8), 8)
    max_area = max(int(np.count_nonzero(shape) * VIA_MAX_AREA_RATIO), VIA_MIN_AREA)

    # 모양 조건은 '누가 VIA 인가' 를 고르는 데만 쓰고 'VIA 가 없다' 고 단정하는 데는
    # 쓰지 않는다. 긁힘이 진짜 VIA 를 가로지르면 둘이 한 덩어리로 붙어 길쭉해지는데,
    # 그때 탈락시키면 멀쩡한 PAD 가 VIA_MISSING 으로 둔갑하기 때문이다.
    #
    # 중심 거리와 경계 여유는 다르다. 되살리지 않는다.
    # PAD 테두리에 딱 붙은 덩어리나 구석에 있는 덩어리는 어떤 경우에도 VIA 가
    # 아니다 (VIA 는 PAD 중앙에 뚫린 구멍이라 정의상 그렇다).
    best, best_area = -1, 0
    alt, alt_area = -1, 0
    for i in range(1, num):
        a = int(stats[i, 4])
        if not (VIA_MIN_AREA <= a <= max_area):
            continue

        # PAD 중심에서 너무 먼 덩어리 걸러내기.  <- 외곽 검은 선 대책
        # 무게중심으로 재는 이유 : 화소 단위로 미리 잘라내면 경계에 걸친 진짜
        # VIA 가 truncate 되어 아래 채움비·모양 수치가 망가진다.
        if np.hypot(float(cents[i][0]) - pad_cx,
                    float(cents[i][1]) - pad_cy) > max_off:
            row["far_rejected"] += 1
            continue

        ys, xs = np.nonzero(lab == i)

        if float(dist[ys, xs].max()) < min_clr:
            row["edge_rejected"] += 1
            continue
        if _fill_ratio(ys, xs, a) < VIA_MIN_FILL:
            row["edge_rejected"] += 1
            continue

        if a > alt_area:
            alt, alt_area = i, a
        aspect, dev = _roundness(ys, xs)
        if aspect > VIA_MAX_ASPECT or dev > VIA_MAX_RADIAL_DEV:
            row["shape_rejected"] += 1
            continue
        if a > best_area:
            best, best_area = i, a

    if best < 0:
        # 원형 후보가 없다. 모양을 포기하고 가장 큰 것을 쓴다 (없다고 하지는 않는다).
        # 단 경계에 붙은 것은 여기서도 안 되살아난다 (애초에 alt 에 안 들어갔다).
        best, best_area = alt, alt_area
    if best < 0:
        return None

    return {"cx": float(cents[best][0]), "cy": float(cents[best][1]),
            "area": best_area}


# ============================================================================
# 결과 이미지
# ============================================================================
def _canvas(vmask: np.ndarray,
            pdes: np.ndarray,
            act: Optional[np.ndarray]) -> np.ndarray:
    """원본을 안 줬을 때 마스크만으로 바탕을 만든다.

    이 모듈은 원본 사진을 안 받는 게 기본이라 그릴 바탕이 없다. 그렇다고 새까만
    화면에 마커만 띄우면 '어디에 그려진 건지' 를 알 수 없으므로, 받은 마스크를
    밝기만 다르게 겹쳐 깔아 준다. 뒤에 그릴 마커는 전부 원색이라 안 섞인다.

        검정 -> 진회색(PAD 설계) -> 중간(실측 PAD) -> 밝은회(준 VIA)  순으로 덮는다.
    """
    g = np.zeros(vmask.shape[:2], np.uint8)
    g[pdes > 0] = BG_PAD_DESIGN
    if act is not None:
        g[act > 0] = BG_PAD_ACTUAL
    g[vmask > 0] = BG_VIA_INPUT
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _draw(src: np.ndarray,
          rows: List[Dict[str, Any]],
          numbering: bool) -> np.ndarray:
    """바탕과 같은 해상도에 판정 마커를 그린다. 바탕은 건드리지 않는다.

        PAD 판정 마커
          정상    : 초록 원   (VIA 가 PAD 안에 있으면 위치와 무관하게 정상)
          VIA없음 : 빨강 X
          PAD없음 : 회색 원

        고른 VIA (판정과 무관하게 항상)
          하늘색 원 + 중심점. 원 크기는 실제로 고른 덩어리 크기입니다.
          주신 VIA 이진화에서 어느 덩어리를 골랐는지 눈으로 바로 확인할 수 있습니다.

    via_checker.py 의 _draw 와 같은 규칙·같은 색으로 그립니다.
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
        else:   # PAD_ABSENT
            cv2.circle(out, (px, py), rad, COLOR_ABSENT, 1, cv2.LINE_AA)
            color = COLOR_ABSENT

        # 고른 VIA 를 실제 크기로 표기한다. 판정선(초록)과 겹쳐도 색이 달라
        # 구분되고, 엉뚱한 덩어리를 골랐으면 엉뚱한 자리에 원이 그려져 바로 보인다.
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
# verbose 출력
# ============================================================================
def _print_rows(rows: List[Dict[str, Any]]) -> None:
    count: Dict[str, int] = {}
    for r in rows:
        count[r["status"]] = count.get(r["status"], 0) + 1
    tally = "  ".join("%s=%d" % (k, count[k]) for k in sorted(count))
    print("검사대상 %d개   %s" % (len(rows), tally))
    if not rows:
        return

    # 편심(offset)은 더 이상 판정에 쓰지 않지만 참고용으로 계속 찍습니다.
    # 중심탈락 = VIA_SEARCH_RATIO 밖이라 잘린 덩어리 수.
    head = ("PAD", "판정", "PAD중심", "VIA중심", "편심px", "편심비율",
            "VIA면적", "모양탈락", "테두리탈락", "중심탈락", "덮임")
    print("%4s %-12s %-16s %-16s %7s %9s %8s %9s %11s %9s %6s" % head)
    print("-" * 114)

    def pt(v):
        return "-" if v is None else "(%.1f,%.1f)" % (v[0], v[1])

    def num(v, f="%.2f"):
        return "-" if v is None else f % v

    for r in rows:
        flag = "NG" if r["status"] == "VIA_MISSING" else "  "
        rej, erej = r["shape_rejected"], r["edge_rejected"]
        frej = r["far_rejected"]
        print("%4d %-12s %-16s %-16s %7s %9s %8s %9s %11s %9s %6s %s" % (
            r["pad_id"], r["status"], pt(r["pad_center"]), pt(r["via_center"]),
            num(r["offset_px"]), num(r["offset_norm"], "%.4f"),
            num(r["via_area"], "%d"),
            "%d" % rej if rej else "-", "%d" % erej if erej else "-",
            "%d" % frej if frej else "-",
            num(r["pad_coverage"], "%.2f"), flag))
