# Wafer_Via / claude

YOLO 십자점 -> 격자 -> 웨이퍼 die map.

512x512 센터 클립 안의 십자 좌표(YOLO 결과)를 받아 서브픽셀 보정하고,
센터 코너 / pitch_x / pitch_y / 회전각을 뽑은 다음,
그 격자를 10000x10000 웨이퍼 전체로 늘려 die map 을 만든다.
좌표를 넣으면 die index 가 나온다.

색은 고정이 아니다. 어떤 색 조합이 와도 같은 코드로 동작한다.


## 쓰는 법 (통짜 파일 하나)

**`wafer_via_claude.py` 한 파일만 복사하면 된다.** 의존성은 `numpy`,
`opencv-python` 뿐이다.

```python
import cv2
from ultralytics import YOLO
from wafer_via_claude import build_die_map_from_yolo, locate_die_via

wafer_bgr = cv2.imread("wafer.png")              # 10000x10000
h, w = wafer_bgr.shape[:2]
center_clip_bgr = wafer_bgr[h//2-256:h//2+256,   # 정중앙 512x512
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
print(dm.aligned_image)      # 회전 보정본 (처음 볼 때 만든다)

info = locate_die_via(dm, point=(7321.0, 4180.5))
print(info["die_index"], info["die_center_px"], info["is_edge"])
```

클립이 정중앙이 아니면 `build_die_map_from_yolo(..., clip_origin=(x0, y0))`
로 알려준다. 격자를 못 세우면 **`ValueError` 를 던진다** — 조용히 틀린
die map 을 주지 않는다.


### 인자 설명

| 인자 | 뜻 |
|---|---|
| `detections` | `(N,4)` 배열. torch.Tensor 를 그냥 넘겨도 된다. `conf`/`cls` 가 뒤에 더 붙어 있어도(`(N,6)`) 앞 4열만 쓴다. |
| `detection_format` | `"xywh"` (bbox 중심, ultralytics `boxes.xywh`) / `"xyxy"` / `"xy"` (이미 중심점만 있을 때) |
| `refine` | 서브픽셀 보정. 끄면 각도 오차가 **56배** 나빠진다 (실측). 비교용 아니면 켜 둔다. |
| `refine_mode` | `"auto"` 는 보정 윈도우를 pitch 에서 자동으로 정한다(권장, 실측상 `pitch*0.30` 이 최선). 이때 **`refine_radius` 는 안 쓰인다**. `"fixed"` 면 `refine_radius` 를 그대로 쓴다. `"off"` 는 `refine=False` 와 같다. |
| `refine_noise_kernel` | 보정 전에 클립에 거는 `medianBlur` 커널(홀수, 0=안 걸음). "흰색+갈색 노이즈" 같은 점잡음용. |
| `refine_min_confidence` | 실측 분포가 진짜 점 p10 = 0.469, 오검출 p90 = 0.003 이라 `0.15` 는 둘 사이 빈 구간에 있다. |

**`boxes.xywhn`(정규화 좌표)을 넣으면 `ValueError` 로 막는다.** 값이 전부
0~1 이면 격자가 소수점 크기로 잡혀 조용히 완전히 틀린 die map 이 나오기 때문이다.


## 파이프라인

```
512x512 클립 + YOLO 검출 (N,4)
        |  detections_to_points     bbox 중심만 뽑는다
        |  refine_points            서브픽셀 보정 (색-무관)
        v
    보정된 점 + 신뢰도
        |  fit_grid                 센터 코너, pitch_x/y, 회전각, 각도 신뢰도
        v
    GridFit  (클립 좌표계)
        |  detect_wafer_adaptive    웨이퍼 외곽선 (색-무관)
        |  build_die_map_via        격자를 웨이퍼 전체로 외삽
        v
    ViaDieMap  (웨이퍼 좌표계)
        +--> dm.index_of(x, y)        -> (i, j)
        +--> locate_die_via(dm, ...)  -> die 정보 dict
        +--> save_debug_overlay(...)  -> 확인용 이미지
```


## 파일

배포용은 **`wafer_via_claude.py` 하나**다. 나머지는 개발/검증용이다.

| 파일 | 역할 |
|---|---|
| **`wafer_via_claude.py`** | **통짜 배포본 (2076줄). 이것만 복사하면 된다.** |
| `build_single_claude.py` | 아래 세 모듈을 합쳐 통짜본을 만든다 |
| `via_refine_claude.py` | YOLO 좌표 서브픽셀 보정 + 신뢰도 |
| `via_grid_claude.py` | 센터 코너 / pitch_x / pitch_y / 회전각 / 각도 신뢰도 |
| `via_diemap_claude.py` | 웨이퍼 외곽선, die map, `build_die_map_from_yolo`, `locate_die_via`, 오버레이 |
| `synth_clip_claude.py` | 합성 512 클립 생성기 (정답 포함) |
| `synth_wafer_claude.py` | 합성 웨이퍼 전체 이미지 생성기 (정답 포함) |
| `test_refine_claude.py` | 보정 정확도 측정 |
| `test_grid_claude.py` | 격자/각도 정확도 측정 |
| `test_diemap_claude.py` | 클립 -> 격자 -> die map 전체 사슬 측정 |
| `test_yolo_api_claude.py` | `build_die_map_from_yolo` 를 사용자 호출 형태 그대로 측정 |

세 모듈을 고쳤으면 `python build_single_claude.py` 로 통짜본을 다시 만든다.
두 벌이 갈라지지 않았는지는 **같은 테스트를 양쪽에 돌려서** 확인한다:

```bash
python test_yolo_api_claude.py --n 6 --size 2000                     # 세 모듈
VIA_MODULE=wafer_via_claude python test_yolo_api_claude.py --n 6 --size 2000
```

지금은 두 결과가 표시된 모든 자릿수까지 동일하다.


## 왜 색에 안 흔들리나

회색조로 바꾸지 않는다. 회색조는 밝기가 같고 색만 다른 die/street 쌍
(isoluminant) 을 통째로 지워버린다. 대신

* **외곽선**: 이미지 4코너에서 배경색을 추정하고 **배경색으로부터의 Lab 거리**
  를 본다. "검정보다 밝은가"가 아니라 "배경과 다른가"라서 흰 배경이든
  검은 배경이든 같은 코드가 돈다.
* **십자점 보정**: `medianBlur` 로 만든 로컬 배경과의 Lab 거리를
  streetness 로 쓴다. street 가 die 보다 밝든 어둡든 **크기**만 보므로
  극성(polarity) 가정이 없다.

`medianBlur` 커널은 street 폭이 아니라 **pitch 에 맞춰야** 한다.
street 폭에 맞추면 street 자체가 배경으로 흡수돼 신호가 사라진다. 실측으로 확인한 값이다.

이건 `refine_noise_kernel` 과 **다른 물건**이다. 그건 점잡음 제거용이고,
streetness 의 `bg_ksize` 는 pitch 에서 자동으로 정해진다.


## 실측 성능

### 사용자 호출 형태 그대로 (`test_yolo_api_claude.py`)

| | 2000x2000 (6장) | 10000x10000 (1장) |
|---|---|---|
| 성공 | 6 / 6 | 1 / 1 |
| 웨이퍼 중심 오차 | 0.71 px | 0.71 px |
| 웨이퍼 반지름 오차 | 0.00 px | 0.00 px |
| pitch 오차 | x 0.0142, y 0.0516 px | x 0.0012, y 0.0057 px |
| 회전각 오차 | 중앙 0.0015, 최악 0.0047 deg | 0.0007 deg |
| `x0,y0` 격자 이탈 | 0.0001 칸 | 0.0001 칸 |
| 격자 불일치 | 2.00e-03 칸 | 3.99e-03 칸 |
| `locate_die_via` 불일치 | 0 건 | 0 건 |
| die map 생성 | 0.33 s | 3.88 s (die 3633개) |

`dies_by_index` 는 `dies` 와 **같은 객체**를 가리킨다 (사본이 아니라서 한쪽을
고치면 양쪽이 같이 바뀐다). 검증됨.

`wafer_boundary` 는 `(cx, cy, r)` 로 그냥 언패킹된다. `.contour` 에는 notch
까지 포함한 실제 외곽선이 들어 있다 (10000px 에서 13112점).

### `angle_confidence` 는 지어낸 점수가 아니다

최소제곱 잔차에서 회전각의 표준편차 `sigma` 를 뽑고,

```
angle_confidence = erf( 0.05deg / (sigma * sqrt(2)) )   = P(|각도오차| < 0.05deg)
```

로 읽는다. 0.05deg 를 기준으로 삼은 이유는 반지름 5000 px 웨이퍼 끝에서
`5000*tan(0.05deg) = 4.36 px` 밀리는 값이고, die pitch 가 100~200 px 이니
한 칸의 2~4% 라서다.

합성 클립 80장으로 **예측한 sigma 와 실제 각도 오차를 대조**했다.

| 확인 | 결과 |
|---|---|
| 신뢰도 >= 0.5 인 79장 | 100% 가 0.05deg 안 (최악 0.0301) |
| 신뢰도 < 0.5 인 1장 | 정확히 그 1장이 유일한 실패(0.3567 deg) |
| 최저 신뢰도 0.330 | 그 실패 건(seed 41). **제대로 집어냈다** |
| 1 sigma 안에 든 비율 | 79% (정규분포 기대 68%) — 보수적 |
| 2 sigma / 3 sigma | 99% / 99% (기대 95% / 99.7%) |
| 순위상관 rho(sigma, 오차) | **+0.342** |

rho 가 0.342 밖에 안 된다는 건 정직하게 말해야 한다. **이건 "선별용 깃발"이지
"오차 예측기"가 아니다.** 망한 건은 확실히 걸러내지만, 멀쩡한 것들 사이의
순위는 못 매긴다. 쓰는 법은 하나다 — `angle_confidence < 0.5` 면 의심하라.

점이 3개뿐이라 잔차가 구조적으로 0 인 경우에는 "오차 0" 이라고 우기지 않고
보정의 실측 바닥값(0.155 px)을 대신 넣는다. 그래서 신뢰도가 가짜 1.0 이
아니라 0.73 으로 나온다.

### 격자 (합성 클립 80장, `test_grid_claude.py --n 80`)

| | 중앙값 | p90 | 최악 |
|---|---|---|---|
| pitch_x 오차 | 0.0146 | 0.0563 | 0.2976 px |
| pitch_y 오차 | 0.0124 | 0.0546 | 0.1213 px |
| 회전각 오차 | 0.0055 | 0.0139 | 0.0301 deg |
| 센터코너 이탈 | 0.0658 | 0.1628 | 0.2380 px |

보정 없이 raw YOLO 좌표로 격자를 세우면 각도 오차 중앙값이 0.3081 deg
(최악 44.90) 다. 서브픽셀 보정이 각도를 **56배** 좋게 만든다.

최악 각도 오차 0.0301 deg 는 10000 px 웨이퍼 끝에서 2.63 px 어긋난다.
die pitch 가 100~200 px 이니 한 칸의 2% 수준이다.

80장 중 1장(pitch 194)은 클립에 십자가 몇 개 안 들어와 가장자리 점까지
되살려 썼고, 그 경우 각도 오차가 0.3567 deg 로 한 등급 나빠진다.
이건 `GridFit.reason` 에 기록되고 `angle_confidence` 로도 걸러진다.

### 서브픽셀 보정 (`test_refine_claude.py`)

점 286개(이미지 30장) 기준, raw YOLO 중앙값 3.019 px -> 보정 후
**0.070 px** (p90 0.155, 최악 0.249). 43배.
신뢰도는 진짜 점 p10 = 0.469, 가짜 점 p90 = 0.003 으로 깨끗이 갈린다.
(팔레트가 시드마다 무작위라 실행할 때마다 소수 셋째 자리는 흔들린다.)

### 전체 사슬 (`test_diemap_claude.py --n 20 --size 2000`)

20장 전부 성공, 격자 불일치 최악 6.80e-03 칸, `locate_die_via` 불일치 0 건.
질의점 10000개 중 12개가 정답과 다른 칸으로 나오는데, 전부 die 경계에서
6.80e-03 칸 안쪽에 떨어진 점이다. 경계 위의 점은 원래 어느 칸인지 애매하므로
오차로 세면 안 된다.

### `aligned_image` 비용 (실측, 32bit 파이썬 / OpenCV 12스레드)

10000x10000x3 기준 **`warpAffine` 0.15 s, 결과 배열 286 MB**.
die map 생성이 3.9 s 이니 시간은 문제가 아니고 **메모리가** 문제다.
32bit 는 주소공간이 ~1.5 GiB 뿐이라 원본(286) + 정렬본(286) 이면 절반이 넘는다.

그래서 `aligned_image` 는 **처음 접근할 때 만드는 lazy 속성**이다.
안 건드리면 0 이고, 다 썼으면 `dm._aligned = None` 으로 놓아주면 된다.
축정렬 여부는 검증했다 — 회전 후 `vx` 의 y성분이 최악 1.85e-02 px.


## 설계 메모 (왜 이렇게 했나)

**이미지를 회전시키지 않는다.** 격자를 축정렬로 만들려고 웨이퍼를 통째로
warp 하면 10000x10000 에서 1억 픽셀을 손실 있게 재샘플링하게 된다.
vx/vy 를 알고 있으니 기울어진 기저에 그대로 die 를 놓고, 좌표->인덱스는
2x2 역행렬로 푼다. 정확하고 보간이 없다.

```
p(i, j) = origin + i*vx + j*vy
ij      = V^-1 (q - origin),  V = [vx | vy]      -> floor 하면 인덱스
```

`dm.aligned_image` 는 이 원칙과 정면으로 충돌한다. 그래서 계산에는 절대
안 쓰고 **보기용으로만** 두었다. 그 위에서 좌표를 쓰려면
`dm.aligned_transform()` 으로 좌표도 같이 옮겨야 한다.

**십자점은 die 의 모서리지 중심이 아니다.** 그래서
`center_xy = origin + (i+0.5)*vx + (j+0.5)*vy` 로 반 칸을 더한다.
`dm.x0, dm.y0` 는 `origin` 그대로 — die(0,0) 의 **좌상 십자점**이다.

**j 는 아래로 갈수록 커진다** (vy 가 아래를 향하므로). v5/v6 의 `iy` 와 반대다.
물리 좌표 `real_coord` 는 v5 와 같이 y-up 을 유지한다.

**die 는 `quad_px`(진짜 네 꼭지점)로 그린다.** `rect_px` 는 축정렬 근사라
격자가 기울면 실제와 어긋난다. 오버레이의 `quad_px` 가 `corner_xy` 와
반올림 오차(최악 0.499 px) 안에서 일치하는 것을 확인했다.

**`cv2.cornerSubPix` 는 십자점에 쓰면 안 된다.** 그건 saddle 을 찾는 도구인데
십자 중심은 saddle 이 아니다. 실측 2.048 px 로, 밴드 프로파일 무게중심
(0.069 px) 보다 30배 나빴다.

**격자 기저는 길이 문턱으로 자르면 안 된다.** 예전에는 "최근접 거리 * 1.4
이하를 한 칸"으로 봤는데, 1.4 는 **정사각** 격자의 대각선 배율 sqrt(2) 에서
온 값이다. pitch_x != pitch_y 면 깨진다.

```
실측 실패 (합성 seed 18): px=121.94, py=136.92
  최근접 중앙값 136.86 -> 문턱 191.60
  대각선 sqrt(px^2+py^2) = 183.35 < 191.60   ** 통과해 버린다 **
```

대각선 5개가 섞이자 |dx|<|dy| 라는 이유로 전부 "세로"로 분류돼 pitch_y 가
136.9 대신 61.1, 회전각이 -3.51 대신 -7.36 도로 나왔다. die map 전체가
60 칸 어긋났는데도 `ok=True` 였다. 지금은 격자 기저 축소로 바꿨다.
가장 짧은 두 독립 벡터를 잡고, 모든 차 벡터를 그 기저로 분해해
인덱스가 정확히 (±1,0)/(0,±1) 인 것만 평균낸다.

**32bit 파이썬 메모리.** 10000x10000x3 을 float32 로 올리면 1.12 GiB 로
바로 MemoryError 다. Lab 거리 맵은 가로 전체 x 512줄씩 끊어 돈다.
정규화 배율(99.5 퍼센타일)만 다운스케일본에서 구한다 - 단순 스케일 상수이고
뒤따르는 Otsu 임계도 같은 스케일 위에서 계산되므로 결과가 안 바뀐다.

**회전각은 90도 주기로만 결정된다.** 정사각 격자는 90도 돌려도 같은 격자다.
|회전| < 45도 라고 가정하고 -45..45 로 접는다.


## 테스트 실행

```bash
python test_yolo_api_claude.py --n 6 --size 2000              # 사용자 API
python test_yolo_api_claude.py --n 1 --size 10000 --aligned   # 실제 크기 + 정렬본 비용
VIA_MODULE=wafer_via_claude python test_yolo_api_claude.py    # 통짜본도 같은지

python test_refine_claude.py                      # 서브픽셀 보정
python test_grid_claude.py --n 80                 # 격자 / 회전각
python test_diemap_claude.py --n 20 --size 2000   # 전체 사슬 (빠름)
python test_diemap_claude.py --n 4 --size 10000   # 실제 크기
python test_diemap_claude.py --n 1 --size 10000 --save out   # 오버레이 저장
```

합성 데이터라 정답을 알고 있으므로 오차가 숫자로 나온다.
색 조합과 회전각은 시드마다 무작위로 바뀐다.
