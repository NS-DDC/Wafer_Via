# Notch 정렬 이미지와 Die Map 사용 설명서

이 문서는 `wafer_via_notch_standalone.py` 한 파일을 복사해서 사용하는 경우를 기준으로 작성했습니다.

현재 pipeline의 핵심 동작은 다음과 같습니다.

1. 512×512 center clip의 YOLO 좌표로 center corner와 `pitch_x`, `pitch_y`를 계산합니다.
2. full wafer 영상의 아래쪽 외곽 원에서 notch를 찾습니다.
3. notch가 기준 방향인 아래쪽 90°에 오도록 **full wafer 이미지를 회전**합니다.
4. Die Map은 회전된 `aligned_image` 좌표계에서 수평·수직, 즉 `grid_angle_deg=0.0`으로 생성합니다.
5. notch 검출 위치를 확인할 수 있는 전체 overlay와 확대 이미지를 함께 반환합니다.

## 1. 필요한 파일과 패키지

아래 파일 하나만 복사하면 됩니다.

```text
wafer_via_notch_standalone.py
```

필요한 외부 패키지:

```bash
pip install numpy opencv-python
```

## 2. 권장 호출 예제

```python
import cv2
import numpy as np

from wafer_via_notch_standalone import (
    build_die_map_from_yolo,
    locate_die,
)

dm = build_die_map_from_yolo(
    # 경로 또는 OpenCV BGR numpy.ndarray 모두 가능
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,

    # Ultralytics YOLO 결과
    detections=results[0].boxes.xywh.cpu().numpy(),
    detection_format="xywh",

    # center clip이 full wafer에서 시작하는 실제 좌표
    # 정확한 crop 원점을 알고 있다면 반드시 넣는 것을 권장
    clip_origin=(clip_x, clip_y),

    # YOLO 좌표 주변의 실제 십자 교차점 미세 보정
    refine=True,
    refine_mode="auto",
    refine_radius=24,
    refine_noise_kernel=5,
    refine_min_confidence=0.15,

    # 현재 영상에서 notch가 있을 것으로 예상하는 방향
    # OpenCV 이미지 좌표: 오른쪽=0°, 아래=90°, 왼쪽=180°, 위=270°
    notch_search_center_angle_deg=90.0,
    notch_search_half_width_deg=45.0,

    # 10000×10000 영상에서 매우 얕은 notch를 보려면 4096 권장
    notch_max_dimension=4096,

    # notch를 찾지 못했을 때 즉시 중단
    notch_failure_mode="error",

    # 보정 영상과 notch 진단 이미지를 반환
    return_aligned_image=True,
    return_notch_visuals=True,
)
```

`clip_origin`을 생략하면 clip이 full wafer의 정확한 가운데에 있다고 가정합니다. 실제 crop 위치가 가운데와 다르면 center corner와 `(0,0)` die 위치가 틀어질 수 있습니다.

## 3. 가장 중요한 좌표계

현재 반환되는 주 Die Map은 원본 이미지 좌표가 아니라 **회전된 `aligned_image` 좌표계**입니다.

```text
원본 wafer_image
    │
    ├─ notch 검출 결과: 원본 좌표
    │
    └─ notch 보정 회전
          │
          ├─ dm.aligned_image
          ├─ dm.x0, dm.y0
          ├─ dm.dies
          └─ locate_die(dm, ...)
             모두 aligned_image 좌표
```

```python
print(dm.coordinate_space)       # "aligned_image"
print(dm.grid_angle_deg)         # 0.0: Die Map 자체는 회전하지 않음
print(dm.image_rotation_deg)     # 원본 이미지에 적용한 회전각
print(dm.source_grid_angle_deg)  # 원본 영상에서 측정한 방향
```

`dm.grid_estimate.*_clip`과 `dm.notch_result.*_px`는 검출에 사용한 원본 clip/full-image 좌표입니다. `dm.pitch_x_points_full`, `dm.pitch_y_points_full`, `dm.x0/y0`, `dm.dies`는 보정된 이미지 좌표입니다.

## 4. 결과 이미지 저장

```python
if dm.aligned_image is not None:
    cv2.imwrite("wafer_aligned.png", dm.aligned_image)

if dm.notch_overlay_image is not None:
    cv2.imwrite("notch_overview.png", dm.notch_overlay_image)

if dm.notch_zoom_image is not None:
    cv2.imwrite("notch_zoom.png", dm.notch_zoom_image)
```

반환 이미지 의미:

| 반환값 | 좌표계 | 내용 |
|---|---|---|
| `dm.aligned_image` | aligned | notch가 90° 아래쪽에 오도록 회전된 full wafer |
| `dm.notch_overlay_image` | original | 원본 전체에서 원 fitting과 notch 검출 결과 |
| `dm.notch_zoom_image` | original | 검출한 notch 주변 확대 이미지 |

전체 overlay는 기본적으로 최대 변을 2048px로 축소해 메모리를 줄입니다. 확대 이미지는 원본 해상도의 notch ROI에서 직접 생성합니다.

```python
dm = build_die_map_from_yolo(
    ...,
    notch_visual_max_dimension=2048,
    notch_zoom_size_px=256,
    notch_zoom_scale=2.0,
)
```

## 5. Notch overlay 읽는 방법

- 하늘색 원: notch가 없다고 가정한 기준 wafer 원
- 회색 contour: 각도별로 실제 추적한 wafer 외곽 edge
- 자홍색 직선 2개: notch를 탐색한 각도 범위
- 노란 arc: notch로 선택한 실제 함몰 구간
- 빨간점: 최종 angle 계산에 사용한 외곽 원 위 기준점
- 작은 초록점: 실제 함몰의 가장 깊은 위치, 진단 전용
- 초록 화살표: wafer 중심에서 빨간 기준점으로 향하는 방향

먼저 하늘색 원과 회색 contour가 실제 wafer 외곽을 따라가는지 확인하고, 그 다음 노란 arc가 실제 notch에 표시됐는지 확인해야 합니다.

## 6. Notch 진단 반환값

```python
notch = dm.notch_result

print(notch.found)
print(notch.detection_method)             # geometry_edge_bottom_sector
print(notch.notch_point_px)               # 원본 좌표의 빨간 기준점
print(notch.notch_deepest_point_px)       # 원본 좌표의 최심점
print(notch.notch_angle_deg)
print(notch.correction_angle_deg)
print(notch.notch_depth_px)
print(notch.notch_width_deg)
print(notch.edge_support)                 # 상대 edge 강도, 0~1
print(notch.circle_fit_residual_px)       # 기준 원 fitting 잔차

print(dm.notch_point_aligned_px)          # aligned_image 좌표의 기준점
print(dm.notch_deepest_point_aligned_px)  # aligned_image 좌표의 최심점
```

`edge_support`가 낮다는 이유만으로 notch를 자동 탈락시키지는 않습니다. 실제 저대비 영상에서도 edge가 약할 수 있으므로 overlay와 깊이·폭을 함께 판단하십시오.

## 7. 좌표를 넣어 die index 찾기

보정된 `aligned_image`에서 얻은 좌표라면 그대로 사용합니다.

```python
hit = locate_die(dm, point=(x_aligned, y_aligned))

print(hit["index"])
print(hit["center_px"])
print(hit["rect_px"])
print(hit["is_edge"])
```

원본 `wafer_image` 좌표라면 affine matrix로 먼저 변환합니다.

```python
point_original = np.asarray([x_original, y_original, 1.0], dtype=np.float64)
point_aligned = dm.original_to_aligned_matrix @ point_original

hit = locate_die(
    dm,
    point=(float(point_aligned[0]), float(point_aligned[1])),
)
```

반대로 보정 좌표를 원본 좌표로 되돌릴 때:

```python
point_aligned_h = np.asarray([x_aligned, y_aligned, 1.0], dtype=np.float64)
point_original = dm.aligned_to_original_matrix @ point_aligned_h
```

## 8. 주요 Die Map 반환값

```python
print(dm.pitch_x, dm.pitch_y)
print(dm.detected_pitch_x, dm.detected_pitch_y)
print(dm.pitch_source)                    # detected 또는 manual

print(dm.x0, dm.y0)                       # aligned_image의 (0,0) origin
print(dm.dies)
print(dm.dies_by_index)
print(dm.num_dies)
print(dm.wafer_boundary)

print(dm.pitch_x_points_full)             # aligned 좌표
print(dm.pitch_y_points_full)             # aligned 좌표
print(dm.source_pitch_x_points_full)      # 원본 좌표
print(dm.source_pitch_y_points_full)      # 원본 좌표
```

수동 pitch를 사용하려면:

```python
dm = build_die_map_from_yolo(
    ...,
    pitch_size=(pitch_x, pitch_y),
)
```

## 9. Notch 미검출 정책

운영 중 잘못된 방향으로 영상을 돌리는 것을 막으려면 `"error"`를 권장합니다.

```python
dm = build_die_map_from_yolo(
    ...,
    notch_failure_mode="error",
)
# 미검출 시 RuntimeError
```

미검출 영상을 회전하지 않고 계속 처리하려면:

```python
dm = build_die_map_from_yolo(
    ...,
    notch_failure_mode="zero",
)

if not dm.notch_result.found:
    assert dm.image_rotation_deg == 0.0
    assert dm.grid_angle_deg == 0.0
```

`"zero"`는 YOLO, FFT 또는 die-render angle로 대체하지 않습니다. notch가 없으면 회전각을 0으로 유지할 뿐입니다.

## 10. 실제 데이터 문제 확인 순서

### A. 맵이 회전되어 보이는 경우

```python
print(dm.coordinate_space)   # aligned_image
print(dm.grid_angle_deg)     # 반드시 0.0
```

DM overlay는 반드시 `dm.aligned_image` 위에 그려야 합니다. 원본 `wafer_image` 위에 aligned 좌표의 `dm.dies`를 그리면 서로 맞지 않습니다.

### B. 이미지가 잘못된 방향으로 회전하는 경우

`dm.notch_overlay_image`와 `dm.notch_zoom_image`에서 노란 arc와 빨간점이 실제 notch인지 확인합니다. 잘못된 위치라면 이미지 회전 문제가 아니라 notch 오검출입니다.

### C. 기준 원이 실제 wafer와 맞지 않는 경우

자동 원 검출 임계값을 먼저 바꾸지 말고 full-image 기준 중심과 반지름 hint를 전달합니다.

```python
dm = build_die_map_from_yolo(
    ...,
    notch_wafer_center_hint_px=(wafer_cx, wafer_cy),
    notch_wafer_radius_hint_px=wafer_radius,
)
```

### D. 원은 맞지만 얕은 notch를 못 찾는 경우

```python
dm = build_die_map_from_yolo(
    ...,
    notch_max_dimension=4096,   # 필요하면 6144
    notch_min_depth_px=5.0,     # 원본 full-image px 단위
)
```

해상도를 높이면 처리 시간과 메모리 사용량이 증가합니다. `notch_min_depth_px`를 지나치게 낮추면 원주 흠집이나 edge 노이즈를 notch로 선택할 수 있으므로 overlay로 확인해야 합니다.

### E. Notch가 아래쪽에서 더 많이 벗어날 수 있는 경우

```python
dm = build_die_map_from_yolo(
    ...,
    notch_search_center_angle_deg=90.0,
    notch_search_half_width_deg=60.0,
)
```

검색 범위를 넓힐수록 다른 외곽 결함을 notch로 선택할 가능성도 커집니다.

## 11. 메모리 절약

10000×10000 BGR 이미지는 한 장만으로도 약 300MB입니다.

```python
dm = build_die_map_from_yolo(
    ...,
    return_aligned_image=False,   # 회전 영상이 필요 없을 때
    return_notch_visuals=False,   # notch 진단을 끝낸 뒤
)
```

두 값을 `False`로 해도 Die Map과 affine matrix, notch 수치 진단값은 유지됩니다.

## 12. 실제 데이터 검증 범위

저장소의 단위 테스트는 좌표 변환, 이미지 회전, 0° DM, 비검정 배경, 얕은 notch, 미검출 정책과 복붙용 단일 파일 실행을 확인합니다. 실제 장비 영상의 notch 검출 정확도를 보장하는 테스트는 아닙니다.

실제 판정에서는 최소한 아래 세 이미지를 같이 보관하는 것을 권장합니다.

```text
wafer_aligned.png
notch_overview.png
notch_zoom.png
```
