# Wafer_Via die_render variant

`wafer_via_die_render.py`는 기존 `wafer_via.py`를 변경하지 않고 V5의 full-wafer `die_render` angle 측정을 추가한 단일 파일 독립 버전입니다. 기존 파이프라인도 파일 안에 모두 포함되어 있으므로 이 파일 하나만 통째로 복사하면 됩니다.

## 사용법

아래 파일 하나만 복사합니다.

```text
wafer_via_die_render.py
```

프로젝트 안의 일반 Python 파일처럼 바로 import합니다. `codex` 패키지 폴더에 넣었다면 첫 번째 방식을, 실행 파일과 같은 폴더에 복사했다면 두 번째 방식을 사용합니다.

```python
# codex 폴더에 넣은 경우
from codex.wafer_via_die_render import build_die_map_from_yolo

# 같은 폴더에 파일 하나만 복사한 경우
# from wafer_via_die_render import build_die_map_from_yolo

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,
    detections=results[0].boxes.xywh.cpu().numpy(),
    detection_format="xywh",
    clip_origin=(clip_x, clip_y),

    angle_align_method="die_render",
)
```

별도의 `wafer_via.py` import나 파일은 필요하지 않습니다. 나머지 인수와 반환 형식은 기존 `wafer_via.build_die_map_from_yolo()`와 같습니다.

## 지원 방식

```python
# V5 die_render angle (기본값)
dm = build_die_map_from_yolo(..., angle_align_method="die_render")

# 기존 512 clip YOLO robust angle (명시할 때만)
dm = build_die_map_from_yolo(..., angle_align_method="yolo")
```

`die_render`가 기본값이며, **YOLO angle로 자동 fallback하지 않습니다.** ROI를 만들 수 없어 측정이 불가능하면 보정을 포기하고 angle `0.0`(무회전)으로 두며, `angle_align_method`는 `die_render_no_signal`이 됩니다. YOLO angle은 `angle_align_method="yolo"`를 직접 지정할 때만 사용됩니다.

`dm.yolo_angle_deg`는 이때도 계속 채워지므로 비교용으로 볼 수 있지만, DM과 `aligned_image`에는 쓰이지 않습니다.

## angle 보정 켜고 끄기

`angle_align_enabled=False`로 두면 angle 측정과 회전을 모두 건너뜁니다.

```python
dm = build_die_map_from_yolo(..., angle_align_enabled=False)
```

끈 상태에서도 `aligned_image`는 그대로 채워집니다. 회전을 하지 않았으므로 **입력 이미지와 바이트 단위로 동일**합니다(변환 행렬은 항등 행렬).

```python
dm.angle_align_method   # "off"
dm.grid_angle_deg       # 0.0
dm.aligned_image        # 입력과 동일한 이미지
dm.original_to_aligned_matrix   # 항등 행렬
dm.aligned_to_original_matrix   # 항등 행렬
```

즉 켜고 끄더라도 `dm.aligned_image`를 쓰는 하위 코드는 고칠 필요가 없습니다.

## 원리

[NS-DDC/Wafer_Map_Die_V5](https://github.com/NS-DDC/Wafer_Map_Die_V5)의 `die_render` angle 방식을 현재 YOLO 기반 DM 파이프라인에 맞게 분리했습니다.

V5의 `measure_die_render_angle`을 **그대로** 옮겨 왔습니다. 계층을 더 얹지 않았습니다.

1. Full wafer 중심 ROI(`roi_ratio=0.55`)를 잘라 회색조로 만들고, 긴 변이 `max_dim=1400`을 넘으면 축소
2. 후보 angle로 ROI를 회전하며 X/Y projection의 분산(주기성)을 계산
3. `±6.0°`를 `0.15°` 간격으로 훑고(coarse), 최대 근방을 `0.02°` 간격으로 다시 훑음(fine)
4. 점수가 최대인 angle이 wafer 기울기. 그 angle로 DM과 `aligned_image`만 다시 생성

FFT 교차검증, prominence 게이트, die grid cue, 반복 보정은 **없습니다**. 한 번 측정하고 끝입니다.

YOLO 중심 보정, `(0,0)` 선택, pitch, wafer 경계, edge clipping, index와 `locate_die()`는 기존 구현을 그대로 사용합니다.

## 결과 확인

```python
print(dm.angle_align_method)  # die_render | die_render_no_signal | off | yolo
print(dm.grid_angle_deg)      # DM/aligned image에 실제 사용한 angle
print(dm.yolo_angle_deg)      # 512 clip YOLO angle 비교값 (사용되지는 않음)
print(dm.angle_confidence)    # 측정값이 나왔는지의 이진 표시 (1.0 / 0.0)
print(dm.angle_agree)         # die_render에서는 항상 False
print(dm.die_render_info)
```

| `angle_align_method` | 의미 |
|---|---|
| `die_render` | V5 projection angle로 보정함 |
| `die_render_no_signal` | ROI를 만들 수 없어 보정을 포기함 (angle `0.0`) |
| `off` | `angle_align_enabled=False`로 꺼둠 (angle `0.0`) |
| `yolo` | `angle_align_method="yolo"`를 직접 지정함 |

`die_render`는 개별 YOLO 좌표쌍이 아니라 full-wafer ROI의 전체 픽셀을 사용합니다. 따라서 최종 angle의 근거를 잘못 표시하지 않도록 `dm.angle_pairs_full`과 `dm.grid_estimate.angle_pairs_clip`은 비어 있습니다. 기존 YOLO angle에 사용된 비교용 좌표쌍은 아래처럼 확인할 수 있습니다.

```python
print(dm.yolo_angle_pairs_full)
print(dm.yolo_angle_pairs_raw_full)
```

`angle_align_method="yolo"`일 때는 기존 `angle_pairs_full`을 그대로 유지합니다.

`die_render_info` 예시 (실제 웨이퍼 `real_mips_top_p084.png` 측정값):

```python
{
    "source": "die_render",
    "angle": 1.2494,
    "confidence": 1.0,          # 품질 점수가 아니라 "측정값이 나왔는가"
    "agree": False,             # die_render에서는 항상 False
}
```

## 튜닝 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `die_render_roi_ratio` | `0.55` | wafer 반지름 대비 중앙 ROI 비율 |
| `die_render_max_dim` | `1400` | ROI 최대 크기 |
| `die_render_search_deg` | `6.0` | angle 탐색 범위 (`±`) |
| `die_render_coarse_step` | `0.15` | coarse angle 간격 |
| `die_render_fine_step` | `0.02` | fine angle 간격 |
| `angle_align_enabled` | `True` | angle 보정 전체 on/off |

10000×10000 이미지에서 처리 시간이 길면 먼저 `die_render_max_dim`을 `1000~1200`으로 낮춰 비교하십시오. 정밀도가 부족하면 `1400~1800` 범위에서 높일 수 있습니다.

## 직접 angle만 측정

```python
from codex.wafer_via_die_render import measure_die_render_angle

angle = measure_die_render_angle(
    wafer_bgr,
    dm.wafer_cx,
    dm.wafer_cy,
    dm.wafer_r,
)

print(angle)   # float 또는 None
```

## 주의사항

- Full wafer에서 실제 die/sawline 주기 패턴이 보여야 합니다.
- **탐색은 항상 최대값을 하나 돌려줍니다.** "여기엔 주기 신호가 없다"를 스스로 알아채지 못합니다. 다만 `build_die_map_from_yolo` 안에서는 YOLO die map이 먼저 만들어진 뒤에 각도를 재므로, die 격자가 전혀 없는 입력은 여기까지 오지 않습니다. 이 한계가 실제로 문제가 되는 것은 `measure_die_render_angle`을 단독으로 임의 이미지에 쓸 때입니다. `None`이 되는 경우는 ROI 자체를 만들 수 없을 때(웨이퍼가 너무 작거나 프레임 밖)뿐입니다.
- Particle, 회로 내부 패턴 또는 banding이 격자보다 강하면 다른 주기를 선택할 수 있습니다.
- **기울기가 `±6.0°`를 넘으면 조용히 틀립니다.** 예외도 `None`도 없이 범위 안에서 점수가 가장 높은 각도를 그럴듯하게 돌려줍니다(실측: −10° 입력 → −3.8155° 반환). 기울기가 클 수 있으면 `die_render_search_deg`를 먼저 키우십시오.
- 위치별 pitch/angle이 달라지는 렌즈 왜곡이나 원근 왜곡은 단일 회전으로 해결되지 않습니다.
- `wafer_via_die_render.py` 하나에 기본 YOLO 파이프라인과 `die_render` 방식이 모두 들어 있습니다. 외부 의존은 `cv2`와 `numpy`뿐이고 로컬 모듈 import가 없어, 파일 하나만 복사하면 그대로 돕니다.
- 전체 input/output 형식(인자 목록, `WaferDieMap` 필드, `dies[i]` 키, `locate_die` 반환 키 등)은 `wafer_via_die_render.py` 맨 끝 `[SECTOR: 95_IO_CONTRACT]` 주석에 있습니다.
