# Wafer notch angle pipeline

새 방향의 angle 기준은 YOLO 점, die line, projection 또는 FFT가 아니라 wafer notch 하나입니다.

## 복붙용 파일

아래 파일 하나만 복사하면 됩니다.

```text
wafer_via_notch_standalone.py
```

필요한 외부 패키지는 `numpy`, `opencv-python`입니다.

```python
from wafer_via_notch_standalone import build_die_map_from_yolo

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,
    detections=results[0].boxes.xywh.cpu().numpy(),
    detection_format="xywh",
    clip_origin=(clip_x, clip_y),

    # notch가 정렬된 뒤 위치할 기준 방향: 아래쪽 6시
    notch_reference_angle_deg=90.0,
)
```

YOLO 좌표는 center corner와 `pitch_x`, `pitch_y`를 찾는 데만 사용합니다. 최종 `grid_angle_deg`는 notch에서만 나옵니다.

## 빨간 기준점 정의

1. 색상을 고정하지 않고 이미지 네 corner의 배경과 다른 중앙 원형 contour를 찾습니다.
2. 원 외곽의 반지름 profile에서 국소적으로 안쪽으로 들어간 notch 구간을 분리합니다.
3. notch 구간 좌우 끝의 각도상 가운데를 구합니다.
4. 그 방향을 원래 wafer 외곽 원까지 연장한 점이 `notch_point_px`입니다.

이 점이 `natural_teal_bluegray_notch_zoom_red.png`에 사용자가 표시한 빨간점입니다. 홈의 가장 안쪽 점은 angle 기준으로 사용하지 않고 `notch_deepest_point_px`에 진단용으로만 보존합니다.

### 사용자 정답과 개선 결과

사용자가 표시한 기준점:

![사용자 빨간 정답점](./NaturalColorSeries/natural_teal_bluegray_notch_zoom_red.png)

개선된 검출 결과:

![개선된 빨간 기준점](./NaturalColorSeries/natural_teal_bluegray_notch_zoom.png)

사용자 표시를 원본 좌표로 역산한 값은 약 `(627.86, 1237.76)`, 개선된 검출값은 `(627.95, 1237.24)`이며 차이는 약 `0.53px`입니다.

네 가지 색상 조건의 확대 검출 결과:

![4개 notch 검출 결과](./NaturalColorSeries/notch_detection_contact_sheet.png)

이미지 좌표 angle은 다음과 같습니다.

```text
오른쪽 =   0°
아래쪽 =  90°
왼쪽   = 180°
위쪽   = 270°
```

`correction_angle_deg`는 notch를 `notch_reference_angle_deg` 방향으로 옮기는 OpenCV 회전값입니다.

## 주요 반환값

```python
print(dm.angle_align_method)             # "notch"
print(dm.grid_angle_deg)                 # DM/aligned image에 사용한 notch 보정각
print(dm.angle_confidence)

print(dm.notch_point_px)                 # 사용자 확인 빨간점: 외곽 원 위 기준 좌표
print(dm.notch_deepest_point_px)         # 홈의 최심점: 진단 전용
print(dm.notch_angle_deg)                # 중심→빨간점의 image-space angle
print(dm.notch_reference_angle_deg)      # 기본 90°
print(dm.notch_depth_px)
print(dm.notch_width_px)
print(dm.notch_result)

print(dm.pitch_x, dm.pitch_y)
print(dm.x0, dm.y0)
print(dm.dies)
print(dm.aligned_image)                  # 기본적으로 보정된 full wafer 반환
```

기존 angle 좌표쌍은 생성하지 않습니다.

```python
assert dm.angle_pairs_full == ()
assert dm.grid_estimate.angle_pairs_clip == ()
assert dm.grid_estimate.angle_candidate_count == 0
```

## notch만 먼저 확인

```python
from wafer_via_notch_standalone import (
    detect_wafer_notch,
    make_notch_overlay,
    make_notch_zoom,
)

notch = detect_wafer_notch(wafer_bgr)

print(notch.notch_point_px)
print(notch.notch_deepest_point_px)
print(notch.notch_angle_deg)
print(notch.correction_angle_deg)
print(notch.confidence)

overlay = make_notch_overlay(wafer_bgr, notch)
zoom = make_notch_zoom(wafer_bgr, notch)

cv2.imwrite("notch_overlay.png", overlay)
cv2.imwrite("notch_zoom.png", zoom)
```

오버레이 색상:

- 빨간점: 최종 angle 기준인 외곽 원 위 좌표
- 작은 초록점: notch 최심점 진단 좌표
- 초록선: wafer 중심에서 빨간점으로 향하는 angle 벡터
- 노란선: 분리된 notch contour 구간
- 하늘색 원: 추정한 원래 wafer 외곽 원

## 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `notch_reference_angle_deg` | `90.0` | 보정 후 notch가 위치할 방향 |
| `notch_max_dimension` | `2048` | 큰 이미지의 notch 검출용 축소 크기 |
| `notch_angle_samples` | `3600` | 원주 반지름 sampling 수, 기본 0.1° 간격 |
| `notch_baseline_window_deg` | `10.0` | 정상 원 외곽을 복원할 주변 각도 폭 |
| `notch_min_depth_px` | `None` | 수동 최소 notch 깊이, 원본 이미지 px |
| `notch_min_depth_ratio` | `0.006` | 자동 최소 깊이, wafer 반지름 비율 |

notch가 검출되지 않으면 기존 angle로 fallback하지 않고 `RuntimeError`를 발생시킵니다. 이는 notch만 angle 기준으로 사용하기 위한 의도된 동작입니다.

## 검증 결과

- 전체 회귀 테스트 25개 통과
- 실제 notch 샘플에 `-27°`, `13°`, `31°` 회전을 적용했을 때 angle 오차 최대 약 `0.15°`
- 정렬 후 notch 잔여 보정각 최대 약 `0.05°`
- `wafer_via_notch_standalone.py` 한 파일만 빈 폴더에 복사한 뒤 DM 생성 성공

수동 pitch는 기존과 동일합니다.

```python
dm = build_die_map_from_yolo(
    ...,
    pitch_size=(pitch_x, pitch_y),
)
```

## 파일 구분

- `wafer_via_notch_standalone.py`: 배포·복붙용 단일 파일
- `wafer_via_notch.py`: 유지보수용 조립형 pipeline
- `wafer_notch_angle.py`: notch 검출과 시각화만 분리한 모듈
- `tools/build_wafer_via_notch_standalone.py`: 단일 파일 재생성 도구
