# Wafer_Via core

512×512 center clip의 YOLO 십자점 검출 결과를 이용해 다음 결과를 생성합니다.

- 실제 wafer 중심에 가장 가까운 십자점을 `(0,0)` grid origin으로 선택
- `pitch_x`, `pitch_y` 자동 측정 또는 수동 입력
- 전체 YOLO 격자 벡터 기반 robust 회전각 추정
- full wafer 외곽선 검출과 회전 die map 생성
- wafer/image 경계에서 잘린 die polygon과 index 유지
- angle 보정 이미지, 좌표 변환, `locate_die`, 진단 overlay 반환

특정 die/street의 RGB·HSV 색상을 고정하지 않습니다. 중심 미세 보정은 각 YOLO 점 주변 네 corner die의 Lab 색상과 경계 정보를 사용합니다.

이전 projection+FFT 방식인 [wafer_via_die_render.py](./wafer_via_die_render.py)는 비교·보관용 legacy입니다. 새 pipeline에서는 사용하지 않습니다.

새 notch 단독 angle 방식은 [wafer_via_notch_standalone.py](./wafer_via_notch_standalone.py)와 [README_NOTCH.md](./README_NOTCH.md)를 사용하십시오. 이 버전은 wafer/배경 색을 분류하지 않고, 아래쪽을 제외한 원주 edge로 기준 원을 fitting한 뒤 아래쪽의 연속된 함몰을 찾습니다. 보정각은 full wafer 이미지에만 적용되고, 반환 DM은 `aligned_image` 좌표계의 0° 격자입니다. YOLO/die-render angle로 fallback하지 않으며, 미검출 시 `notch_failure_mode="error"`로 예외를 내거나 `"zero"`로 보정각 0을 반환할 수 있습니다.

## 1. 권장 사용법

```python
import cv2

from codex.wafer_via import (
    build_die_map_from_yolo,
    locate_die,
    make_clip_overlay,
    make_wafer_overlay,
    transform_point_to_aligned,
    transform_point_to_original,
)

# full image 중앙에서 YOLO용 512x512 clip 생성
height, width = wafer_bgr.shape[:2]
clip_x = width // 2 - 256
clip_y = height // 2 - 256
center_clip_bgr = wafer_bgr[clip_y:clip_y + 512, clip_x:clip_x + 512]

results = model(center_clip_bgr)
detections_xywh = results[0].boxes.xywh.cpu().numpy()

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,          # full wafer BGR ndarray
    clip_image=center_clip_bgr,     # YOLO에 넣은 512x512 BGR ndarray
    detections=detections_xywh,     # shape=(N,4), [cx,cy,w,h]
    detection_format="xywh",
    clip_origin=(clip_x, clip_y),   # full image에서 clip의 왼쪽 위

    refine=True,
    refine_mode="auto",
    refine_radius=24,
    refine_noise_kernel=5,
    refine_min_confidence=0.15,

    angle_mode="robust",
    angle_inlier_tolerance_deg=2.5,

    pitch_size=None,               # None=자동, 또는 (pitch_x, pitch_y)
    include_edge=True,
    return_aligned_image=True,
)
```

`wafer_image`, `clip_image`는 경로 또는 OpenCV `numpy.ndarray`를 모두 지원합니다. 생산 코드에서는 이미지와 `clip_origin`의 대응을 명확히 하기 위해 메모리 배열과 실제 crop 원점을 전달하는 방식을 권장합니다.

정확한 이미지 중앙에서 clip했다면 `clip_origin`을 생략할 수 있습니다.

## 2. 입력 좌표 형식

권장 Ultralytics 입력은 다음 형식입니다.

```python
detections = results[0].boxes.xywh.cpu().numpy()
detection_format = "xywh"
```

- 배열 shape: `(N, 4)`
- 한 행: `[center_x, center_y, width, height]`
- 단위: YOLO 입력 clip 기준 pixel
- 원점: clip 왼쪽 위 `(0,0)`
- `x`는 오른쪽, `y`는 아래로 증가
- `xywhn`이 아닌 비정규화 `xywh`

Confidence와 class를 먼저 필터링할 수 있습니다.

```python
boxes = results[0].boxes
keep = (boxes.conf >= 0.25) & (boxes.cls == 0)
detections = boxes.xywh[keep].cpu().numpy()
```

지원 형식:

| `detection_format` | 한 행의 구조 |
|---|---|
| `point` | `[x, y]` |
| `point_conf` | `[x, y, confidence]` |
| `xywh` | `[cx, cy, width, height]` pixel |
| `xyxy` | `[x1, y1, x2, y2]` pixel |
| `xyxy_conf_class` | `[x1, y1, x2, y2, confidence, class]` |
| `yolo_txt` | `[class, cx, cy, width, height, (confidence)]` |

정규화 좌표는 `normalized=True`를 명시하십시오. 픽셀 단위 6열 데이터는 형식이 모호하므로 `detection_format`을 생략하지 않는 것이 안전합니다.

결과 구조가 불확실하면 다음 진단 함수를 사용합니다.

```python
from codex.wafer_via import inspect_yolo_results

summary = inspect_yolo_results(results, max_rows=10)
```

## 3. 좌표계와 `(0,0)` 기준

이 모듈에는 세 좌표계가 있습니다.

| 좌표계 | 대표 값 |
|---|---|
| clip 좌표 | `dm.grid_estimate.*_clip` |
| 원본 full-image 좌표 | `dm.x0`, `dm.y0`, `dm.dies` |
| angle 보정 이미지 좌표 | `dm.aligned_image` |

`dm.x0`, `dm.y0`는 이미지 중심이나 wafer 중심 그 자체가 아닙니다. 검출된 wafer 중심을 clip 좌표로 변환한 뒤, 그 위치에 가장 가까운 YOLO 십자점을 선택한 full-image 좌표입니다.

```python
print("wafer center:", (dm.wafer_cx, dm.wafer_cy))
print("grid origin:", (dm.x0, dm.y0))
```

`locate_die()`와 die map은 원본 full-image 좌표를 사용합니다. 보정 이미지의 좌표는 변환 후 사용하십시오.

```python
aligned_point = transform_point_to_aligned(dm, original_point)
original_point = transform_point_to_original(dm, aligned_point)
result = locate_die(dm, point=original_point)
```

## 4. Pitch 설정과 진단

### 자동 측정

```python
dm = build_die_map_from_yolo(..., pitch_size=None)

print(dm.pitch_source)       # "detected"
print(dm.pitch_x, dm.pitch_y)
```

### 수동 입력

```python
dm = build_die_map_from_yolo(
    ...,
    pitch_size=(81.25, 93.50),
)

print(dm.pitch_source)                    # "manual"
print(dm.pitch_x, dm.pitch_y)             # DM에 사용한 값
print(dm.detected_pitch_x, dm.detected_pitch_y)  # YOLO 자동 측정 비교값
```

수동 pitch를 사용해도 `(0,0)` 기준과 angle은 YOLO/wafer 결과에서 계산합니다.

### Pitch 계산 좌표

```python
grid = dm.grid_estimate

print(grid.pitch_x_points_clip)       # (P0, PX), 보정 후 clip 좌표
print(grid.pitch_y_points_clip)       # (P0, PY), 보정 후 clip 좌표
print(grid.pitch_x_points_raw_clip)   # 중심 보정 전 YOLO 좌표
print(grid.pitch_y_points_raw_clip)

print(dm.pitch_x_points_full)         # full-image 좌표
print(dm.pitch_y_points_full)
print(dm.pitch_x_points_raw_full)
print(dm.pitch_y_points_raw_full)
```

자동 pitch는 다음 거리와 같습니다.

```python
import math

print(math.dist(*dm.pitch_x_points_full), dm.detected_pitch_x)
print(math.dist(*dm.pitch_y_points_full), dm.detected_pitch_y)
```

`PX` 또는 `PY`가 바로 인접한 십자점이 아니라면 점 선택 문제입니다. 점은 정확하지만 wafer 외곽으로 갈수록 실제 간격이 달라지면 단일 pitch 문제가 아니라 렌즈 왜곡·원근 변화 가능성이 큽니다.

## 5. Angle 보정과 진단

기본 `angle_mode="robust"`는 전체 YOLO 점에서 가까운 X/Y 방향 벡터를 모으고 이상치를 제거합니다. 기존 두 벡터 방식은 비교용 `local` 모드로 유지됩니다.

```python
grid = dm.grid_estimate

print("final:", dm.grid_angle_deg)
print("robust:", grid.robust_angle_deg)
print("local:", grid.local_angle_deg)
print("confidence:", dm.angle_confidence)
print("used/candidate:", len(grid.angle_pairs_clip), grid.angle_candidate_count)
```

Angle에 사용된 좌표와 개별 잔차:

```python
for pair, axis, measured, residual in zip(
    grid.angle_pairs_clip,
    grid.angle_pair_axes,
    grid.angle_pair_angles_deg,
    grid.angle_pair_residuals_deg,
):
    print(axis, pair, measured, residual)

print(dm.angle_pairs_full)       # 보정 후 full-image 좌표쌍
print(dm.angle_pairs_raw_full)   # 중심 보정 전 full-image 좌표쌍
```

튜닝 기준:

- 기본값: `angle_inlier_tolerance_deg=2.5`
- 좌표가 정밀하면 `1.0~1.5`
- 노이즈가 강하면 `3.0~4.0`
- 너무 크게 설정하면 잘못된 좌표쌍도 포함될 수 있음
- 위치에 따라 angle이 계속 변하면 homography 또는 렌즈 왜곡 보정 필요

기존 방식과 비교하려면 다음 옵션을 사용합니다.

```python
dm_local = build_die_map_from_yolo(..., angle_mode="local")
```

## 6. 중심 미세 보정

기본 `refine_mode="auto"` 처리:

1. YOLO 점 주변 ROI의 네 corner patch에서 die 대표 Lab 색상 계산
2. 네 색상과 모두 다른 vertical/horizontal street 후보 계산
3. Median filter와 projection으로 점 노이즈 억제
4. Lab gradient 경계쌍 결과와 결합
5. `refine_min_confidence` 미만이면 YOLO 원좌표 유지

```python
dm = build_die_map_from_yolo(
    ...,
    refine=True,
    refine_mode="auto",        # auto | corner_color | gradient
    refine_radius=24,
    refine_noise_kernel=5,
    refine_min_confidence=0.15,
)

grid = dm.grid_estimate
print(grid.raw_points_clip)
print(grid.points_clip)
print(grid.refinement_confidences)
```

실제 교차점이 ROI 밖이면 `refine_radius`를 키우십시오. 보정 후가 더 부정확하면 `refine=False` 결과와 비교합니다.

## 7. 진단 Overlay

### Clip overlay

```python
clip_overlay = make_clip_overlay(center_clip_bgr, dm.grid_estimate)
cv2.imwrite("clip_grid_debug.png", clip_overlay)
```

표시 의미:

- 흰색 빈 원: 보정 전 YOLO 중심
- 노란 점: 보정 후 실제 사용 좌표
- 회색 선: 중심 보정 이동량
- `P0`: `(0,0)` 기준 십자점
- `PX`, `PY`: pitch 계산점
- 청록/자홍 선: robust angle에 사용된 X/Y 좌표쌍
- `N=사용 개수/전체 후보`: angle 이상치 제거 결과

### Full wafer overlay

```python
wafer_overlay = make_wafer_overlay(
    wafer_bgr,
    dm,
    draw_dies=True,
    thickness=1,
)
cv2.imwrite("wafer_die_map.png", wafer_overlay)
```

## 8. Wafer 경계, edge die와 이미지 밖 index

`include_edge=True`가 기본값입니다. Die 중심이 wafer나 이미지 밖이어도 polygon 일부가 wafer에 걸리면 index를 생성합니다.

개별 die entry:

| key | 의미 |
|---|---|
| `index` | `(ix, iy)`, X+ 오른쪽, Y+ 위쪽 |
| `center_px` | 자르지 않은 die 중심 |
| `polygon_px` | index용 전체 polygon, 이미지 밖 좌표 유지 |
| `wafer_polygon_px` | wafer 외곽으로 자른 polygon |
| `visible_polygon_px` | wafer와 이미지 범위로 자른 polygon |
| `rect_px` | 전체 `polygon_px` bounding rectangle |
| `crop_rect_px` | 이미지 안에서 crop 가능한 rectangle |
| `full_area_px` | 자르기 전 면적 |
| `wafer_area_px` | wafer 경계로 자른 면적 |
| `visible_area_px` | 이미지에서 실제 보이는 면적 |
| `is_edge_partial` | 일부가 wafer 밖인지 여부 |
| `is_image_partial` | 일부가 이미지 밖인지 여부 |
| `is_outside_image` | 이미지에 보이는 부분이 없는지 여부 |

```python
die = dm.get_die(ix=2, iy=-3)
if die is not None:
    print(die["polygon_px"])
    print(die["wafer_polygon_px"])
    print(die["visible_polygon_px"])
    print(die["crop_rect_px"])
```

`polygon_px`는 index/격자 계산용으로 자르지 않습니다. 표시와 crop에는 `visible_polygon_px`, `crop_rect_px`를 사용합니다. `include_edge=False`이면 wafer 경계에 걸린 partial die를 제외합니다.

## 9. `locate_die()`

Point 또는 bbox 중심으로 die index를 찾습니다.

```python
result = locate_die(dm, point=(5499, 4700))

# 또는
result = locate_die(dm, bbox=(4880, 5080, 4980, 5180))
```

주요 반환값:

```python
print(result["die_index"])
print(result["die_center_px"])
print(result["die_polygon_px"])
print(result["wafer_polygon_px"])
print(result["visible_polygon_px"])
print(result["crop_rect_px"])
print(result["in_wafer"])
print(result["is_edge"])
```

Query가 wafer 또는 이미지 밖이어도 격자 수식으로 `die_index`를 계산합니다. 실제 wafer 내부 여부는 `in_wafer`로 확인하십시오.

## 10. 주요 반환 객체

### `WaferDieMap`

| 필드 | 의미 |
|---|---|
| `wafer_cx`, `wafer_cy`, `wafer_r` | 검출된 wafer 중심과 반지름 |
| `x0`, `y0` | `(0,0)` grid origin의 full-image 좌표 |
| `pitch_x`, `pitch_y` | DM에 실제 사용한 pitch |
| `detected_pitch_x`, `detected_pitch_y` | YOLO에서 자동 측정한 pitch |
| `pitch_source` | `detected`, `manual`, `direct` |
| `grid_angle_deg` | DM과 aligned image에 사용한 angle |
| `angle_confidence` | 선택된 angle 방식의 신뢰도 |
| `grid_estimate` | clip 기반 중심/pitch/angle 상세 진단 |
| `wafer_boundary` | full-image wafer contour |
| `dies`, `dies_by_index` | die entry 목록과 index dictionary |
| `aligned_image` | angle 보정된 full BGR 이미지 또는 `None` |
| `original_to_aligned_matrix` | 원본 → 보정 좌표 affine matrix |
| `aligned_to_original_matrix` | 보정 → 원본 좌표 affine matrix |
| `num_dies` | 생성된 die 수 |

### Angle 보정 이미지

```python
if dm.aligned_image is not None:
    cv2.imwrite("wafer_aligned.png", dm.aligned_image)
```

10000×10000 BGR 이미지는 한 장당 약 300MB입니다. 메모리가 부족하면 다음 옵션을 사용합니다.

```python
dm = build_die_map_from_yolo(..., return_aligned_image=False)
```

좌표 변환 matrix는 이 경우에도 반환됩니다.

## 11. 문제 확인 순서

1. `results[0].orig_shape == center_clip_bgr.shape[:2]`인지 확인
2. `detections_xywh`를 clip에 직접 그려 YOLO 중심 확인
3. `make_clip_overlay()`에서 흰 원 → 노란 점 이동 확인
4. `P0/PX/PY`가 바로 인접한 십자점인지 확인
5. `robust_angle_deg`, `local_angle_deg`, pair residual 비교
6. `clip_origin`이 실제 crop 왼쪽 위 좌표인지 확인
7. `dm.wafer_cx/cy`와 `dm.x0/y0`를 full image에서 확인
8. 외곽으로 갈수록 pitch/angle이 변하면 렌즈 왜곡 또는 homography 검토

## 12. 파일 복사와 테스트

별도 패키지 설치 없이 사용하려면 [wafer_via.py](./wafer_via.py) 파일 전체를 프로젝트에 복사하십시오. `build_die_map_from_yolo()` 함수 하나만 복사하면 내부 helper와 dataclass가 없어 동작하지 않습니다.

필수 라이브러리:

```bash
pip install numpy opencv-python
```

테스트:

```bash
python -B -m unittest discover -s tests -v
```

현재 회귀 테스트는 다음을 포함합니다.

- Ultralytics/YOLO 좌표 형식
- 다양한 die 색상과 강한 노이즈 중심 보정
- 한 중심점이 틀린 경우 robust angle 복원
- 수동 pitch override
- image center와 wafer center가 다른 경우 `(0,0)` 선택
- wafer/image 경계 die clipping과 index 유지
- 10000×10000 die-map geometry와 `locate_die`
- angle 보정 이미지와 좌표 왕복 변환
