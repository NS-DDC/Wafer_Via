# Wafer_Via / claude

YOLO 십자점 -> 격자 -> 웨이퍼 die map.

512x512 센터 클립 안의 십자 좌표(YOLO 결과)를 받아 서브픽셀 보정하고,
센터 코너 / pitch_x / pitch_y / 회전각을 뽑은 다음,
그 격자를 10000x10000 웨이퍼 전체로 늘려 die map 을 만든다.
좌표를 넣으면 die index 가 나온다.

색은 고정이 아니다. 어떤 색 조합이 와도 같은 코드로 동작한다.


## 파이프라인

```
512x512 클립 + YOLO 점 리스트 [(x,y), ...]
        |
        |  via_refine_claude.refine_points     서브픽셀 보정 (색-무관)
        v
    보정된 점 + 신뢰도
        |
        |  via_grid_claude.fit_grid            센터 코너, pitch_x/y, 회전각
        v
    GridFit  (클립 좌표계)
        |
        |  via_diemap_claude.build_die_map_via  + 웨이퍼 원본 이미지
        v                                       (외곽선 검출 -> 격자 확장)
    ViaDieMap  (웨이퍼 좌표계)
        |
        +--> dm.index_of(x, y)        -> (i, j)
        +--> locate_die_via(dm, ...)  -> die 정보 dict
        +--> save_debug_overlay(...)  -> 확인용 이미지
```

두 줄로 쓰면 이렇다.

```python
from via_grid_claude import analyze_clip
from via_diemap_claude import build_die_map_via, locate_die_via

res, g = analyze_clip(clip512, yolo_points)          # yolo_points = [(x,y), ...]
dm = build_die_map_via(wafer_image, g)               # 클립은 이미지 정중앙이라고 가정
info = locate_die_via(dm, point=(7321.0, 4180.5))
print(info["die_index"], info["die_center_px"], info["is_edge"])
```

클립이 정중앙이 아니면 `build_die_map_via(..., clip_origin=(x0, y0))` 로 알려준다.


## 파일

| 파일 | 역할 |
|---|---|
| `via_refine_claude.py` | YOLO 좌표 서브픽셀 보정 + 신뢰도 |
| `via_grid_claude.py` | 센터 코너 / pitch_x / pitch_y / 회전각 |
| `via_diemap_claude.py` | 웨이퍼 외곽선, die map, `locate_die_via`, 오버레이 |
| `synth_clip_claude.py` | 합성 512 클립 생성기 (정답 포함) |
| `synth_wafer_claude.py` | 합성 웨이퍼 전체 이미지 생성기 (정답 포함) |
| `test_refine_claude.py` | 보정 정확도 측정 |
| `test_grid_claude.py` | 격자/각도 정확도 측정 |
| `test_diemap_claude.py` | 클립 -> 격자 -> die map 전체 사슬 측정 |

의존성은 `numpy`, `opencv-python` 뿐이다.


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


## 실측 성능

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
되살려 썼고, 그 경우 각도 오차가 0.357 deg 로 한 등급 나빠진다.
이건 `GridFit.reason` 에 기록되므로 조용히 넘어가지 않는다.

### 서브픽셀 보정 (`test_refine_claude.py`)

점 286개(이미지 30장) 기준, raw YOLO 중앙값 3.019 px -> 보정 후
**0.070 px** (p90 0.155, 최악 0.249). 43배.
신뢰도는 진짜 점 p10 = 0.469, 가짜 점 p90 = 0.003 으로 깨끗이 갈린다.
(팔레트가 시드마다 무작위라 실행할 때마다 소수 셋째 자리는 흔들린다.)

### 전체 사슬 (`test_diemap_claude.py --n 4 --size 10000`)

| | 값 |
|---|---|
| 웨이퍼 중심 오차 | 0.71 px (정답 4999.5 를 정수로 반올림한 값) |
| 웨이퍼 반지름 오차 | 0.00 px |
| pitch 오차 | x 최악 0.039, y 최악 0.068 px |
| 회전각 오차 | 중앙값 0.0036, 최악 0.0057 deg |
| 클립->웨이퍼 원점 이탈 | 최악 0.0001 칸 |
| die 수 / 이론값 | 중앙값 0.999 |
| 격자 불일치 | 최악 1.66e-02 칸 |
| `locate_die_via` 불일치 | 0 건 |
| die map 생성 | 3.84 s (die 3159~7804개) |

2000x2000 에서는 20장 전부 성공, 격자 불일치 최악 6.80e-03 칸.

**인덱스 오답에 대해**: 질의점 2000개 중 9개가 정답과 다른 칸으로 나온다.
전부 die 경계에서 1.66e-02 칸 안쪽에 떨어진 점이다. 경계 위의 점은
원래 어느 칸인지 애매하므로 오차로 세면 안 된다.


## 설계 메모 (왜 이렇게 했나)

**이미지를 회전시키지 않는다.** 격자를 축정렬로 만들려고 웨이퍼를 통째로
warp 하면 10000x10000 에서 1억 픽셀을 손실 있게 재샘플링하게 된다.
vx/vy 를 알고 있으니 기울어진 기저에 그대로 die 를 놓고, 좌표->인덱스는
2x2 역행렬로 푼다. 정확하고 보간이 없다.

```
p(i, j) = origin + i*vx + j*vy
ij      = V^-1 (q - origin),  V = [vx | vy]      -> floor 하면 인덱스
```

**십자점은 die 의 모서리지 중심이 아니다.** 그래서
`center_xy = origin + (i+0.5)*vx + (j+0.5)*vy` 로 반 칸을 더한다.

**j 는 아래로 갈수록 커진다** (vy 가 아래를 향하므로). v5/v6 의 `iy` 와 반대다.
물리 좌표 `real_coord` 는 v5 와 같이 y-up 을 유지한다.

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
python test_refine_claude.py                      # 서브픽셀 보정
python test_grid_claude.py --n 80                 # 격자 / 회전각
python test_diemap_claude.py --n 20 --size 2000   # 전체 사슬 (빠름)
python test_diemap_claude.py --n 4 --size 10000   # 실제 크기
python test_diemap_claude.py --n 1 --size 10000 --save out   # 오버레이 저장
```

합성 데이터라 정답을 알고 있으므로 오차가 숫자로 나온다.
색 조합과 회전각은 시드마다 무작위로 바뀐다.
