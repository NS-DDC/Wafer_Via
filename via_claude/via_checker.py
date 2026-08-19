# -*- coding: utf-8 -*-
"""via_checker.py - VIA 검사 (단일 파일, 복붙용)

이미지 4장 넣으면 코드와 결과 이미지가 나옵니다. 그게 전부입니다.

    from via_checker import check_via

    code, result, via_bin = check_via(원본, 이진화, PAD설계도, VIA설계도)

    #  code    : "1"  양품
    #            "42" VIA 없음
    #            "-1" 입력 오류 (이때 result 와 via_bin 은 None)
    #  result  : 원본과 같은 해상도의 결과 이미지 (찾아낸 VIA 도 하늘색 원으로 표기)
    #  via_bin : 검출한 VIA 의 이진화 마스크 (원본과 같은 해상도, 0/255 단일채널)

쏠림(편심)은 불량으로 세지 않습니다. VIA 가 PAD 안에 있기만 하면 양품입니다.
과검이 너무 많아서 뺐습니다. 편심 수치 자체는 debug_via 의 offset_norm 으로
계속 볼 수 있으니, 필요하면 호출한 쪽에서 직접 기준을 세우세요.

PAD 표면 얼룩 때문에 엉뚱한 것을 VIA 로 잡으면 임계값을 내리세요.

    code, result, via_bin = check_via(원본, 이진화, PAD설계도, VIA설계도,
                                      dark_offset=-25)  # 색이 확실히 다른 것만 인정

디버깅할 때는 debug_via 를 쓰세요. PAD 별 수치가 표로 찍히고
결과 이미지에는 PAD 번호가 함께 그려집니다.

    code, result, via_bin, rows = debug_via(원본, 이진화, PAD설계도, VIA설계도)

    for r in rows:
        print(r["pad_id"], r["status"], r["offset_norm"])

--------------------------------------------------------------------------
입력 4장
--------------------------------------------------------------------------
  1) 원본        : 검사할 이미지          (BGR / GRAY ndarray 또는 파일 경로)
  2) 이진화      : 원본을 이진화한 실측 PAD 마스크
  3) PAD 설계도  : PAD 가 있어야 할 자리 (이진)
  4) VIA 설계도  : VIA 가 있어야 할 자리 (이진)

  네 장 모두 같은 해상도·같은 좌표계여야 합니다.
  크기가 다르면 조용히 리사이즈하지 않고 "-1" 로 실패합니다.

--------------------------------------------------------------------------
출력 via_bin - 검출한 VIA 의 이진화 마스크
--------------------------------------------------------------------------
  원본과 같은 해상도의 0/255 단일채널 이미지입니다.
  '검사 대상 PAD 안에서 VIA 로 채택한 덩어리' 만 흰색으로 남습니다.

    - VIA 를 못 찾은 PAD(VIA_MISSING) 와 PAD_ABSENT 는 아무것도 안 찍힙니다.
    - 후보였다가 면적 조건에서 탈락한 덩어리도 안 찍힙니다.
      즉 흰 픽셀 = 이 판정의 근거가 된 VIA 그 자체입니다.

  판정 근거를 눈으로 확인하거나 다음 공정에 넘길 때 쓰세요.

    code, result, via_bin = check_via(...)
    print(cv2.connectedComponents(via_bin)[0] - 1)   # 검출된 VIA 개수

--------------------------------------------------------------------------
검사 대상
--------------------------------------------------------------------------
  VIA 설계도에 점이 찍힌 PAD 만 검사합니다.
  설계상 VIA 가 없는 PAD 는 아예 건드리지 않습니다.

--------------------------------------------------------------------------
판정
--------------------------------------------------------------------------
  OK           VIA 를 찾음               -> code "1"   (쏠렸어도 양품입니다)
  VIA_MISSING  VIA 를 못 찾음            -> code "42"
  PAD_ABSENT   실물 PAD 자체가 없음      -> code 에 반영 안 함 (PAD 누락은 별개 검사 영역)

==========================================================================
VIA 를 어떻게 찾는가  <- 고치실 곳은 여기입니다
==========================================================================
소스에서는 _find_via() 안의

    # >>>>>>>>>>>>>>>>  VIA 검출 로직 시작  >>>>>>>>>>>>>>>>
    ...
    # <<<<<<<<<<<<<<<<  VIA 검출 로직 끝  <<<<<<<<<<<<<<<<

구간이 전부입니다. 그 밖은 후보 거르기와 그리기라 건드릴 일이 거의 없습니다.
아래 네 조건을 모두(AND) 통과한 화소만 VIA 후보가 됩니다.

  [1] PAD 대표색과 색이 다르다
      대표색  = PAD 안 화소들의 채널별 중앙값 (B, G, R)
      색상차  = |화소BGR - 대표색BGR| 을 세 채널 더한 값 (L1)
      임계    = 색상차중앙값 + COLOR_K * 로버스트편차 - dark_offset
      로버스트편차 = 1.4826 * MAD(색상차)   (COLOR_FLOOR 아래로는 안 내려감)

      평균·표준편차가 아니라 중앙값·MAD 를 쓰는 이유는, VIA 나 얼룩이 몇 개
      섞여 있어도 기준선이 그쪽으로 끌려가지 않게 하기 위해서입니다.
      PAD 마다 자기 색을 기준으로 재는 상대값이라 PAD 색이 라인마다 달라도,
      VIA 대비가 연해도 같은 숫자를 그대로 씁니다.

  [2] PAD 보다 어둡다
      VIA 는 항상 검은색이거나 진한 갈색입니다. 밝은 쪽으로 색이 달라진
      화소(반사·하이라이트)는 여기서 전부 빠집니다.
        조건 : 화소밝기 < PAD밝기중앙값 - VIA_DARK_MARGIN

  [3] 색이 검정 또는 갈색 계열이다
      검정은 B≈G≈R 이고 갈색은 R > G > B 입니다. 둘 다 '파랑·초록이 빨강보다
      세지 않다' 로 묶입니다.
        조건 : B <= R + VIA_WARM_TOL  그리고  G <= R + VIA_WARM_TOL

      HSV 색상각을 안 쓴 이유 : 어두운 화소는 채도·색상각이 노이즈로 요동쳐서
      새까만 VIA 가 '파란색' 으로 튀어 나옵니다. BGR 대소 비교가 훨씬 안정적입니다.

  [4] 주변에서 홀로 튄다 (Black-hat)
      Black-hat = 닫힘(closing) - 원본. 주변보다 우묵하게 파인 곳만 남습니다.
      커널은 PAD 반지름 * BLACKHAT_KSIZE_RATIO 라 VIA 보다 크고 PAD 보다 작습니다.
      PAD 테두리는 '단조 경사' 라 응답이 거의 0 이므로 자연히 걸러집니다.
      덕분에 PAD 를 크게 깎지 않아도 되고 가장자리로 쏠린 VIA 도 안 놓칩니다.

      PAD 바깥(어두운 배경)은 미리 PAD 밝기 중앙값으로 메우고 겁니다. 안 그러면
      가장자리 VIA 주변에서 closing 이 배경 어둠에 끌려 내려가 응답이 사라집니다.

--------------------------------------------------------------------------
가짜 VIA 를 어떻게 거르는가
--------------------------------------------------------------------------
위 [1]~[4] 를 통과한 화소를 연결요소로 묶은 뒤, 덩어리마다 아래를 봅니다.
'탈락' 은 되살아나지 않고 '선호' 는 대안이 없으면 되살아납니다.

  면적      탈락   VIA_MIN_AREA ~ PAD면적*VIA_MAX_AREA_RATIO 밖
  중심거리  탈락   PAD 중심에서 PAD반지름*VIA_SEARCH_RATIO 보다 먼 덩어리
  경계여유  탈락   덩어리 안 가장 깊은 점이 PAD 경계에 붙어 있음
  채움비    탈락   자기 최소외접원을 못 채움 (링·호·선)
  모양      선호   원이 아니면 뒤로 밀어둠 (원형 후보가 없으면 그냥 씀)

  * 중심거리 - VIA 는 PAD 중앙에만 뚫립니다. 그래서 아예 중앙 근처만 봅니다.
    PAD 바깥을 스치는 검은 선, 이웃 패턴, 테두리 링이 여기서 통째로 빠집니다.
    ("PAD 외곽에 검은 선이 있으면 이상하게 오류가 난다" 가 이 조건으로 막힙니다.)
    실측으로 진짜 VIA 는 중심에서 0.46*R 안쪽에 다 들어옵니다 -> 임계 0.55

  * 경계여유 - 링은 장단비 1.0 / 반경편차 0.13~0.20 이라 모양으로는 '완벽한 원'
    입니다. 모양 필터로는 절대 못 잡습니다. 위치로만 잡힙니다.
      경계여유 = max(거리변환(설계PAD)[덩어리]) / PAD반지름
    실측 : 진짜 VIA 최소 0.536 / 테두리 링 중앙 0.254 -> 임계 0.30

  * 채움비 - 경계여유는 '가장 깊은 한 점' 만 보기 때문에 전이대가 두꺼우면
    아슬아슬하게 통과합니다 (실측 : 8.00 vs 임계 7.71). 채움비는 '전체 모양' 을
    봅니다. 링이 이미지 끝에서 잘려 열린 C 자가 되어도 최소외접원은 그대로
    크므로 값이 낮게 유지됩니다.
      채움비 = 덩어리 면적 / 그 덩어리의 최소외접원 면적
    실측 : 꽉 찬 원 0.6~0.9 / 링·호 0.2 아래 -> 임계 0.45

  * 모양만 '선호' 인 이유 - 긁힘이 진짜 VIA 를 가로지르면 둘이 한 덩어리로 붙어
    길쭉해집니다. 그때 탈락시키면 멀쩡한 PAD 가 VIA_MISSING 으로 둔갑합니다.
    실측에서 이 역전이 4건 나왔고, 대안이 있을 때만 거르도록 바꿔 0건이 됐습니다.
    즉 이 필터는 판정을 나쁘게 만들 수 없습니다.

  몇 개가 어디서 걸렸는지는 rows 의 shape_rejected / edge_rejected / far_rejected
  에 각각 찍힙니다. 튜닝할 때 이 세 숫자를 먼저 보세요.

--------------------------------------------------------------------------
그 밖에 실물에서 자주 나던 오판정
--------------------------------------------------------------------------
  "VIA 구멍 때문에 PAD 가 없다고 나옴"
     실물에서는 VIA 가 이진화 마스크에 구멍으로 뚫립니다. 면적 커버리지만
     보면 이 구멍 때문에 멀쩡한 PAD 가 PAD_ABSENT 로 빠집니다.
     그래서 커버리지를 두 가지로 재고 둘 중 큰 값을 씁니다.
       full : 설계 PAD 전체 대비
       excl : VIA 가 있어야 할 자리를 빼고 계산   <- 구멍에 영향 없음
     구멍 메우기(hole filling) 는 쓰지 않습니다.

  "노이즈를 VIA 로 잡음" 이 남아 있으면
     dark_offset 을 음수로 주세요. 위 [1] 의 임계가 그만큼 올라갑니다.
       임계 = 색상차중앙값 + COLOR_K * 로버스트편차 - dark_offset
     음수면 더 엄격해지고(오검출이 줄고) 양수면 더 민감해집니다(연한 VIA 를
     더 잡습니다). 단위가 밝기(0~255)가 아니라 세 채널 합이라 대략 3배
     스케일이니 -15 정도씩 옮기면서 debug_via 의 dark_threshold 를 보세요.

  탐색 영역을 더 깎는(PAD_ERODE 를 키우는) 방식은 실패했습니다.
  연한 VIA 는 화소가 4~5개뿐이라 3px 만 깎아도 통째로 사라집니다
  (실측 : 테스트셋 OK 705 -> 661). 자세한 것은 PAD_ERODE 주석 참고.

의존성 : numpy, opencv-python  (Python 3.9+)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = ["check_via", "debug_via",
           "CODE_OK", "CODE_VIA_MISSING", "CODE_ERROR"]


# ============================================================================
# 결과 코드
# ============================================================================
CODE_OK = "1"            # 양품
CODE_VIA_MISSING = "42"  # VIA 없음
CODE_ERROR = "-1"        # 입력 오류

# 쏠림(편심) 코드 "99" 는 없앴습니다. 과검이 너무 많았습니다.
# VIA 가 PAD 안에서 발견되기만 하면 위치와 무관하게 양품입니다.
# 편심 수치는 rows 의 offset_px / offset_norm 으로 계속 나오므로,
# 다시 쓰고 싶으면 호출한 쪽에서 그 값으로 직접 기준을 세우세요.


# ============================================================================
# 튜닝 값 - 결과가 마음에 안 들면 이 숫자만 고치면 됩니다
# ============================================================================

# 설계 PAD 가 실물보다 몇 px 작게 그려졌는지. 그 만큼 되돌려서 공칭 크기로 씁니다.
# 설계도를 실물과 같은 크기로 그렸으면 0.
DESIGN_PAD_SHRINK = 2

# 국소 정합. 설계 PAD 를 실측 마스크 무게중심 쪽으로 이만큼까지 평행이동해서
# 설계도-실물 어긋남을 흡수합니다. 0 이면 정합 안 함.
#
# 쏠림 판정을 없앤 뒤에도 이게 필요한 이유 : 아래 VIA_SEARCH_RATIO 가 'PAD 중심에서
# 얼마나 가까운가' 로 후보를 자르는데, 그 중심이 어긋나 있으면 멀쩡한 VIA 가 잘립니다.
# 커버리지(PAD 존재 판정)의 기준 위치이기도 합니다.
ALIGN_MAX_RATIO = 0.40    # 최대 이동량 = PAD반지름 * 이 값
ALIGN_MARGIN = 2          # 무게중심을 잴 때 설계 PAD 를 몇 px 키운 창을 볼지
ALIGN_MIN_COVER = 0.30    # 창 안 실측 픽셀이 이 비율도 안 되면 정합 생략

# ----------------------------------------------------------------------------
# ★ VIA 검출 조건 ★  <- 검출 결과를 바꾸고 싶으면 이 네 값입니다
#   실제로 쓰이는 곳은 _find_via() 안의 "VIA 검출 로직 시작/끝" 구간 하나뿐입니다.
# ----------------------------------------------------------------------------
#
# [1] PAD 대표색과 색이 다른가
#       색상차 = |화소BGR - PAD대표색BGR| 을 세 채널 더한 값 (L1)
#       임계   = 색상차중앙값 + COLOR_K * 로버스트편차 - dark_offset
#       로버스트편차 = 1.4826 * MAD(색상차)   (COLOR_FLOOR 아래로는 안 내려감)
#
#     PAD 마다 자기 색 분포를 기준으로 재는 상대값입니다. 그래서 PAD 색이
#     라인마다 달라도, VIA 대비가 연해도 같은 숫자를 그대로 씁니다.
#     dark_offset 은 호출할 때 넘기는 인자입니다 (기본 0). 여기 상수는 잘 안 고칩니다.
#       COLOR_K 를 내리면 -> 더 민감 (연한 VIA 를 잡지만 얼룩도 같이 잡힘)
#       COLOR_K 를 올리면 -> 더 엄격
COLOR_K = 3.5
COLOR_FLOOR = 6.0             # 편차가 0 에 가까운 매끈한 PAD 에서 임계가 붕괴하는 것 방지

# [2] PAD 보다 어두운가  (VIA 는 항상 검은색이거나 진한 갈색이다)
#       조건 : 화소밝기 < PAD밝기중앙값 - VIA_DARK_MARGIN
#     밝은 쪽으로 색이 튄 화소(반사·하이라이트)를 통째로 뺍니다.
#     0 으로 두면 '조금이라도 어두우면' 이 되고, 올릴수록 진한 것만 인정합니다.
VIA_DARK_MARGIN = 3.0

# [3] 검정 또는 갈색 계열인가
#       조건 : B <= R + VIA_WARM_TOL  그리고  G <= R + VIA_WARM_TOL
#     검정은 B≈G≈R, 갈색은 R > G > B 이므로 둘 다 이 한 줄로 묶입니다.
#     파랑·초록·청록 쪽으로 강하게 치우친 이물만 빠집니다.
#     허용폭입니다. 키우면 느슨해지고(파란기 도는 것도 통과) 줄이면 빡빡해집니다.
#
#     HSV 색상각을 안 쓴 이유 : 어두운 화소는 채도·색상각이 노이즈로 요동쳐서
#     새까만 VIA 가 '파란색' 으로 튀어 나옵니다. BGR 대소 비교가 훨씬 안정적입니다.
VIA_WARM_TOL = 12.0

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

# [4] 주변에서 홀로 튀는가  (Black-hat = 닫힘 - 원본, 우묵하게 파인 곳만 남는다)
# VIA 는 항상 어두우므로 Black-hat 만 씁니다 (예전에는 Top-hat 도 같이 썼습니다).
# PAD 테두리는 '단조 경사' 라 응답이 거의 0 이므로 자연히 걸러집니다.
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

# VIA 를 찾을 범위 = PAD 중심에서 PAD반지름 * 이 값 이내.
# VIA 는 PAD 중앙에만 뚫리므로 아예 중앙 근처만 봅니다.
# 덩어리 중심이 이 원 밖이면 무조건 탈락입니다 (되살리지 않음).
#
# 이 조건이 필요한 이유 : PAD 외곽을 스치는 검은 선, 이웃 패턴, 테두리 링이
# 크기·모양 조건을 다 통과해 버리는 경우가 있습니다. 그것들은 공통적으로
# PAD 중심에서 멉니다. 위치 하나로 통째로 잘라내는 것이 가장 확실합니다.
#
# 실측 (덩어리 중심 - PAD 중심 거리를 PAD 반지름으로 나눈 값)
#   진짜 VIA   중앙 0.03  p95 0.457  p99 0.596  최대 0.661   (테스트셋 746개)
#   외곽 잡음  p1   0.771  p5  0.859               (VIA 가 없는 실물 191개)
# 둘 사이가 0.66 ~ 0.77 로 비어 있어 그 안에 선을 그었습니다.
#
# 이 값을 바꿨을 때 실측 결과 (왼쪽 = 실물에서 없는 VIA 를 찾아낸 수)
#   0.55  오검출 2/777   테스트셋 754/783   <- 쏠린 VIA 18개를 잘라먹음
#   0.75  오검출 2/777   테스트셋 772/783   <- 채택
#   0.90  오검출 179/777 테스트셋 772/783   <- 외곽 잡음이 다시 들어옴
#   제한없음 오검출 191/777
# 쏠린 VIA 까지 살리려면 올리고, 외곽 오검출이 계속 나오면 내리세요.
# 몇 개가 잘렸는지는 rows 의 far_rejected 에 찍힙니다.
VIA_SEARCH_RATIO = 0.75

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
COLOR_MISSING = (0, 0, 255)     # 빨강 : VIA 없음
COLOR_ABSENT = (150, 150, 150)  # 회색 : PAD 없음
COLOR_VIA = (255, 255, 0)       # 하늘 : 찾아낸 VIA 위치 (판정과 무관하게 항상 표기)


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

    code 는 "1"(양품) / "42"(VIA 없음) / "-1"(입력 오류) 중 하나입니다.
    VIA 가 쏠려 있어도 PAD 안에서 발견되기만 하면 양품("1")입니다.
    via_bin 은 검출한 VIA 만 흰색인 0/255 마스크입니다 (원본과 같은 해상도).
    "-1" 이면 result 와 via_bin 은 None 이고, 이유가 표준에러로 출력됩니다.

    dark_offset : VIA 를 찾는 색상차 임계값을 올리고 내립니다 (기본 0).

        임계값 = 색상차중앙값 + 3.5 * 로버스트편차 - dark_offset
        색상차 = |화소BGR - PAD대표색BGR| 을 세 채널 더한 값

        음수  색이 확실히 다른 픽셀만 VIA 로 본다 -> 표면 얼룩에 안 속음
        양수  조금만 달라도 VIA 로 본다          -> 연한 VIA 를 놓치지 않음

    단위가 밝기(0~255)가 아니라 세 채널 합이라 대략 3배 스케일입니다.
    한 번에 크게 움직이지 말고 -15 정도씩 옮기면서 debug_via 의
    dark_threshold 열을 보세요. 실제로 적용된 값이 그대로 찍힙니다.
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
    dark_offset 도 check_via 와 같은 뜻입니다.

    - PAD 별 수치를 표로 출력합니다 (quiet=True 로 끄기)
    - 결과 이미지에 PAD 번호를 함께 그립니다
    - rows 는 PAD 별 dict 목록입니다. 들어있는 키:
        pad_id, status, pad_center, pad_radius, pad_area, pad_coverage,
        align_shift, design_via, via_center, via_area, offset_px, offset_norm,
        pad_median, dark_threshold, via_aspect, via_radial_dev,
        shape_rejected, edge_rejected, far_rejected
      (해당 없는 항목은 None)

    align_shift 가 크게 나오면 설계도와 실물이 그만큼 어긋나 있다는 뜻입니다.
    offset_norm 은 편심 수치입니다. 판정에는 안 쓰지만 참고용으로 계속 나옵니다.

    후보가 어디서 걸렸는지는 아래 세 숫자로 봅니다. 튜닝할 때 여기부터 보세요.
      shape_rejected 원이 아니라 뒤로 밀어낸 수  (대안이 없으면 되살아납니다)
      edge_rejected  PAD 경계에 붙었거나 속이 빈(링·호) 탈락 수
      far_rejected   PAD 중심에서 멀어서 탈락한 수 (VIA_SEARCH_RATIO 밖)
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

    # VIA 는 'PAD 대표색과 색이 다른 곳' 으로 찾으므로 흑백이 아니라 컬러를 넘깁니다.
    blur = cv2.GaussianBlur(src, (3, 3), 0)

    pad_label, boxes = _label_pads(pdes)
    design_vias = _map_vias(vdes, pad_label)

    dil = None
    if DESIGN_PAD_SHRINK > 0:
        d = DESIGN_PAD_SHRINK
        dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    # 침식은 1px 로 얕게만 한다. 여기서 크게 깎으면 안 된다.
    # 연한 VIA 는 화소가 4~5개뿐이라 3px 만 깎아도 통째로 사라진다
    # (실측: 침식 1->3 으로 바꿨더니 테스트셋 OK 705 -> 661 로 45개가
    #  오검출 VIA_MISSING 이 됐다). 탐색 영역이 PAD 밖으로 조금 나가는 문제는
    # 깎아서가 아니라 아래 VIA_MIN_CLEARANCE 로 '위치' 를 보고 막는다.
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
            "via_aspect": None,
            "via_radial_dev": None,
            "shape_rejected": 0,
            "edge_rejected": 0,
            "far_rejected": 0,
        }

        # ---- 실물 PAD 가 그 자리에 있는지 ----
        row["pad_coverage"] = round(
            _coverage(shape, act, area, radius, dvx - x0, dvy - y0), 3)
        if row["pad_coverage"] < PAD_PRESENT_MIN:
            # PAD 가 없으면 VIA 가 없는 게 당연하므로 VIA 불량으로 세지 않는다.
            row["status"] = "PAD_ABSENT"
            rows.append(row)
            continue

        # ---- VIA 찾기 ----
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
        # 편심은 재서 보여주기만 하고 판정에는 쓰지 않는다.
        # PAD 안에서 VIA 를 찾았으면 위치와 무관하게 양품이다.
        row["offset_norm"] = round(dist / radius if radius > 1e-6 else 999.0, 4)
        row["status"] = "OK"
        rows.append(row)

    code = CODE_VIA_MISSING if CODE_VIA_MISSING in found_status else CODE_OK
    return code, src, via_bin, rows, ""


def _align(shape: np.ndarray,
           act: np.ndarray,
           cx: float,
           cy: float,
           radius: float,
           area: float,
           alk: np.ndarray) -> Tuple[np.ndarray, float, float, Tuple[float, float]]:
    """설계 PAD 형상을 실측 마스크 쪽으로 조금 평행이동한다.

    설계도와 실물은 2~3px 어긋난다. PAD 반지름이 6px 수준이면 그것만으로
    중심이 반지름의 절반 가까이 밀린다. 그 계통 오차를 걷어내는 것이 목적이다.

    여기서 잡은 중심이 VIA_SEARCH_RATIO 의 기준점이 되므로, 이게 틀어지면
    멀쩡한 VIA 가 '중심에서 멀다' 는 이유로 잘려 나간다.
    이동량은 반지름의 ALIGN_MAX_RATIO 이내로 묶어 폭주를 막는다.
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
              pad_cx: float,
              pad_cy: float,
              radius: float,
              ero: np.ndarray,
              row: Dict[str, Any],
              dark_offset: float) -> Optional[Dict[str, Any]]:
    """PAD 한 개 안에서 VIA 를 찾는다. 없으면 None.  roi 는 BGR 컬러다.

    찾으면 {"cx", "cy", "area", "mask", "aspect", "radial_dev"} 를 준다.
    mask 는 채택한 덩어리의 bool 배열(roi 와 같은 크기)로,
    호출부에서 via_bin 에 찍는 데 쓴다.

    (pad_cx, pad_cy) 는 국소 정합까지 끝난 PAD 중심이다. VIA 는 PAD 중앙에만
    뚫리므로 이 점에서 얼마나 떨어져 있는지가 후보를 자르는 기준이 된다.

    아래 "VIA 검출 로직 시작/끝" 사이가 검출의 전부다. 네 조건의 AND 로 후보
    화소를 만들고, 그 뒤는 덩어리 단위로 거르고 고르는 일뿐이다.
    각 조건의 뜻과 튜닝 방향은 파일 위쪽 "★ VIA 검출 조건 ★" 상수 블록에 있다.
    """
    inner = cv2.erode(shape, ero)
    if np.count_nonzero(inner) < VIA_MIN_AREA * 4:
        return None

    # 각 화소가 PAD 경계에서 얼마나 떨어져 있는지. 아래에서 테두리 링을 거를 때 쓴다.
    dist = cv2.distanceTransform((shape > 0).astype(np.uint8), cv2.DIST_L2, 5)
    min_clr = radius * VIA_MIN_CLEARANCE
    max_off = radius * VIA_SEARCH_RATIO      # VIA 는 PAD 중앙에만 있다

    m = inner > 0
    f = roi.astype(np.float32)
    gray = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray[m]))          # PAD 대표 밝기

    # >>>>>>>>>>>>>>>>>>>>>>>  VIA 검출 로직 시작  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 아래 네 조건을 모두(AND) 통과한 화소만 VIA 후보(cand)가 된다.
    # 조건을 더하거나 빼려면 이 구간만 고치면 된다.

    # [1] PAD 대표색과 색이 다른가
    #     대표색은 채널별 중앙값이라 얼룩이 몇 개 섞여 있어도 그쪽으로 안 끌려간다.
    base = np.median(f[m], axis=0)              # PAD 대표색 (B, G, R)
    diff = np.abs(f - base).sum(axis=2)         # 대표색과의 L1 색상차
    #     임계도 색상차 자체의 분포로 정한다. 평균/표준편차 대신 중앙값/MAD 를 쓰는
    #     이유는 VIA 나 얼룩이 기준선을 끌어올리지 못하게 하기 위해서다.
    d = diff[m]
    d_med = float(np.median(d))
    scale = max(1.4826 * float(np.median(np.abs(d - d_med))), COLOR_FLOOR)
    thr = d_med + COLOR_K * scale - dark_offset
    c_color = diff > thr

    # [2] PAD 보다 어두운가   (VIA 는 항상 검은색이거나 진한 갈색이다)
    #     밝은 쪽으로 색이 튄 화소 - 반사, 하이라이트, 밝은 이물 - 이 여기서 빠진다.
    c_dark = gray.astype(np.float32) < med - VIA_DARK_MARGIN

    # [3] 검정 또는 갈색 계열인가
    #     검정은 B≈G≈R, 갈색은 R > G > B. 둘 다 '파랑·초록이 빨강보다 세지 않다'.
    #     파랑·초록·청록으로 강하게 치우친 이물만 빠진다.
    c_warm = ((f[:, :, 0] <= f[:, :, 2] + VIA_WARM_TOL) &
              (f[:, :, 1] <= f[:, :, 2] + VIA_WARM_TOL))

    # [4] 주변에서 홀로 튀는가   (Black-hat = 닫힘 - 원본, 우묵한 곳만 남는다)
    #     PAD 바깥(어두운 배경)을 PAD 중앙값으로 메운 뒤 건다. 그대로 두면 가장자리
    #     VIA 주변에서 closing 이 배경 어둠에 끌려 내려가 응답이 사라진다.
    filled = np.where(shape > 0, gray, np.uint8(round(med)))
    ks = int(np.clip(int(round(radius * BLACKHAT_KSIZE_RATIO)) | 1, 5, 31))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bh = cv2.morphologyEx(filled, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    c_hat = bh > max(BLACKHAT_MIN, med * BLACKHAT_RATIO)

    cand = (c_color & c_dark & c_warm & c_hat & m).astype(np.uint8)
    # <<<<<<<<<<<<<<<<<<<<<<<  VIA 검출 로직 끝  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    row["pad_median"] = round(med, 1)
    row["dark_threshold"] = round(thr, 1)

    # 여기서 MORPH_OPEN 으로 잡티를 지우면 안 된다.
    # 대비가 약한 VIA(코어 밝기 = PAD 밝기의 0.45배)는 임계 아래로 내려가는 화소가
    # 4~5개뿐이고 모양도 십자/대각이라 2x2 로 열면 통째로 사라진다(= 오검출 VIA_MISSING).
    # 잡티 제거는 아래 연결성분 면적 하한(VIA_MIN_AREA)이 이미 담당한다.
    # 네 조건을 동시에 통과한 화소가 우연히 4개나 이어질 확률은 낮다.

    num, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    max_area = max(int(np.count_nonzero(shape) * VIA_MAX_AREA_RATIO), VIA_MIN_AREA)

    # VIA 는 항상 원형이다. 크기가 맞아도 모양이 원이 아니면 뒤로 밀어둔다.
    # PAD 표면의 긁힘 자국은 크기만 보면 진짜 VIA 보다 커서 그냥 두면 이기지만,
    # 1px 폭이라 장단비에서 걸린다. 그래서 원형인 것 중에서 먼저 고른다.
    #
    # 다만 모양 조건은 '누가 VIA 인가' 를 고르는 데만 쓰고
    # 'VIA 가 없다' 고 단정하는 데는 쓰지 않는다. 원형 후보가 하나도 없으면
    # 모양을 무시하고 가장 큰 것을 그냥 쓴다. 이유 :
    #   긁힘이 진짜 VIA 를 가로지르면 둘이 한 덩어리로 붙어 길쭉해진다.
    #   이때 탈락시키면 멀쩡한 PAD 가 VIA_MISSING(코드 42) 으로 둔갑한다.
    #   실측에서 이 역전이 4건 나왔고, 대안이 있을 때만 거르도록 바꿔 0건이 됐다.
    # 즉 이 필터는 판정을 뒤집지 못하고 후보 선택만 바꾼다.
    #
    # 중심 거리와 경계 여유는 이것들과 달리 '탈락' 이다. 되살리지 않는다.
    # PAD 테두리에 딱 붙은 덩어리나 구석에 있는 덩어리는 어떤 경우에도 VIA 가
    # 아니기 때문이다. (VIA 는 PAD 중앙에 뚫린 구멍이라 정의상 그렇다.)
    best, best_area, best_shape = -1, 0, (0.0, 0.0)
    alt, alt_area, alt_shape = -1, 0, (0.0, 0.0)
    rejected = 0
    edge_rejected = 0
    far_rejected = 0
    for i in range(1, num):
        a = int(stats[i, 4])
        if not (VIA_MIN_AREA <= a <= max_area):
            continue

        # PAD 중심에서 너무 먼 덩어리 걸러내기.  <- 외곽 검은 선 대책
        # PAD 바깥쪽을 지나가는 검은 배선/이웃 패턴 자국은 색·어둡기·모양이
        # 진짜 VIA 와 구분이 안 될 때가 있다. 하지만 위치는 확실히 다르다.
        # VIA 는 PAD 중앙에만 뚫리므로 여기서 먼저 자른다.
        # 무게중심으로 재는 이유 : 화소 단위로 미리 마스킹해 버리면 경계에 걸친
        # 진짜 VIA 가 잘려 나가 아래 채움비·모양 수치가 망가진다.
        if np.hypot(float(cents[i][0]) - pad_cx,
                    float(cents[i][1]) - pad_cy) > max_off:
            far_rejected += 1
            continue

        ys, xs = np.nonzero(lab == i)

        # PAD 테두리를 두른 어두운 링 걸러내기.
        # 링은 장단비 1.0 / 반경편차 0.1~0.2 라 모양으로는 '완벽한 원' 이고
        # 무게중심도 PAD 중심과 겹쳐서 편심 0 = OK 로 나온다.
        # (가) 위치 : 덩어리에서 가장 깊은 점이 PAD 경계에 가까우면 링이다.
        # (나) 채움 : 링/호는 자기 최소외접원을 거의 못 채운다. (가) 가 아슬아슬하게
        #      통과하는 두꺼운 전이대도 여기서 확실히 걸린다.
        if float(dist[ys, xs].max()) < min_clr:
            edge_rejected += 1
            continue
        if _fill_ratio(ys, xs, a) < VIA_MIN_FILL:
            edge_rejected += 1
            continue

        aspect, dev = _roundness(ys, xs)
        if a > alt_area:
            alt, alt_area, alt_shape = i, a, (aspect, dev)
        if aspect > VIA_MAX_ASPECT or dev > VIA_MAX_RADIAL_DEV:
            rejected += 1
            continue
        if a > best_area:
            best, best_area, best_shape = i, a, (aspect, dev)

    row["shape_rejected"] = rejected
    row["edge_rejected"] = edge_rejected
    row["far_rejected"] = far_rejected
    if best < 0:
        # 원형 후보가 없다. 모양을 포기하고 가장 큰 것을 쓴다 (없다고 하지는 않는다).
        # 단 경계에 붙은 것은 여기서도 안 되살아난다 (애초에 alt 에 안 들어갔다).
        best, best_area, best_shape = alt, alt_area, alt_shape
    if best < 0:
        return None

    blob = lab == best
    # 이진 중심은 픽셀 양자화 오차가 커서 작은 VIA 에서 흔들린다.
    # '색이 얼마나 다른가'를 가중치로 쓴 무게중심이 훨씬 안정적이다(서브픽셀).
    wgt = np.clip(diff - thr, 0.0, None) * blob
    total = float(wgt.sum())
    if total > 1e-6:
        gy, gx = np.mgrid[0:shape.shape[0], 0:shape.shape[1]].astype(np.float32)
        cx = float((gx * wgt).sum() / total)
        cy = float((gy * wgt).sum() / total)
    else:
        cx, cy = float(cents[best][0]), float(cents[best][1])

    return {"cx": cx, "cy": cy, "area": best_area, "mask": blob,
            "aspect": best_shape[0], "radial_dev": best_shape[1]}


# ============================================================================
# 결과 이미지
# ============================================================================
def _draw(src: np.ndarray, rows: List[Dict[str, Any]], numbering: bool) -> np.ndarray:
    """원본과 같은 해상도에 판정 마커를 그린다. 원본은 건드리지 않는다.

        PAD 판정 마커
          정상    : 초록 원   (VIA 가 PAD 안에 있으면 위치와 무관하게 정상)
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
        elif st == "VIA_MISSING":
            cv2.drawMarker(out, (px, py), COLOR_MISSING, cv2.MARKER_TILTED_CROSS,
                           max(rad * 2, 7), 1, cv2.LINE_AA)
            color = COLOR_MISSING
        else:   # PAD_ABSENT
            cv2.circle(out, (px, py), rad, COLOR_ABSENT, 1, cv2.LINE_AA)
            color = COLOR_ABSENT

        # 찾아낸 VIA 를 실제 크기로 표기한다. 판정선(초록)과 겹쳐도
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

    # 편심(offset)은 더 이상 판정에 쓰지 않지만, 참고용으로 계속 찍는다.
    # 중심탈락 = VIA_SEARCH_RATIO 밖이라 잘린 덩어리 수 (외곽 검은 선 대책이
    # 실제로 몇 개를 잡아냈는지 여기서 확인하면 된다).
    head = ("PAD", "판정", "PAD중심", "VIA중심", "편심px", "편심비율",
            "VIA면적", "장단비", "반경편차", "모양탈락", "테두리탈락", "중심탈락",
            "PAD밝기", "임계", "덮임", "정합이동")
    print("%4s %-12s %-16s %-16s %7s %9s %8s %7s %9s %9s %10s %8s %8s %6s %6s %12s" % head)
    print("-" * 164)

    def pt(v):
        return "-" if v is None else "(%.1f,%.1f)" % (v[0], v[1])

    def num(v, f="%.2f"):
        return "-" if v is None else f % v

    for r in rows:
        flag = "NG" if r["status"] == "VIA_MISSING" else "  "
        rej = r["shape_rejected"]
        erej = r["edge_rejected"]
        frej = r["far_rejected"]
        print("%4d %-12s %-16s %-16s %7s %9s %8s %7s %9s %9s %10s %8s %8s %6s %6s %12s %s" % (
            r["pad_id"], r["status"], pt(r["pad_center"]), pt(r["via_center"]),
            num(r["offset_px"]), num(r["offset_norm"], "%.4f"),
            num(r["via_area"], "%d"),
            num(r["via_aspect"]), num(r["via_radial_dev"], "%.3f"),
            "%d" % rej if rej else "-", "%d" % erej if erej else "-",
            "%d" % frej if frej else "-",
            num(r["pad_median"], "%.1f"), num(r["dark_threshold"], "%.1f"),
            num(r["pad_coverage"], "%.2f"), pt(r["align_shift"]), flag))
