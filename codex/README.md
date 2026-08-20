# Wafer_Via core

학습된 YOLO가 512×512 중앙 clip에서 찾은 여러 십자점으로 다음 정보만 생성합니다.

- 중앙에 가장 가까운 `center_corner`
- 중앙 코너의 옆 점으로 `pitch_x`
- 중앙 코너의 아래 점으로 `pitch_y`
- 두 축 벡터로 회전 보정각 `angle_deg`
- 원본 wafer 외곽 contour
- 원본(예: 10000×10000) 좌표계의 회전 die map
- `locate_die`와 clip/full-wafer overlay

die/street의 RGB·HSV 색상 임계값은 사용하지 않습니다. YOLO 좌표가 기준이며 `refine=True`일 때만 Lab 경계의 쌍을 이용해 street 중앙을 미세 보정합니다.

## 빠른 사용법

```python
import cv2
from codex.wafer_via import build_die_map_from_yolo, locate_die, make_wafer_overlay

dm = build_die_map_from_yolo(
    wafer_image="wafer_10000.png",
    clip_image="center_clip_512.png",
    detections="center_clip_512.txt",  # YOLO: class cx cy w h (normalized)
    refine=False,                       # 모델 bbox 중심을 그대로 쓰는 기본값
)

print(dm.x0, dm.y0)                     # full image 기준 center corner
print(dm.pitch_x, dm.pitch_y)
print(dm.grid_angle_deg, dm.angle_confidence)
cv2.imwrite("wafer_aligned.png", dm.aligned_image)  # angle 보정된 full image

result = locate_die(dm, point=(5499, 4700))
print(result["die_index"], result["die_polygon_px"])

overlay = make_wafer_overlay("wafer_10000.png", dm)
cv2.imwrite("wafer_overlay.png", overlay)
```

Ultralytics의 `boxes.xyxy`도 바로 전달할 수 있습니다.

```python
dm = build_die_map_from_yolo(
    wafer_image,
    clip_image,
    results[0].boxes.xyxy.cpu().numpy(),
    detection_format="xyxy",
)
```

이미지 경로 대신 OpenCV BGR `numpy.ndarray`를 그대로 전달할 수 있습니다.

```python
dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,       # uint8 ndarray, 예: (10000, 10000, 3)
    clip_image=center_clip_bgr,  # uint8 ndarray, 예: (512, 512, 3)
    detections=yolo_array,       # Nx2/Nx3/Nx4/Nx5/Nx6 ndarray 또는 list
)
```

`auto`가 지원하는 메모리 좌표 형식:

- `x, y`
- `x, y, confidence`
- `x1, y1, x2, y2`
- `class, cx, cy, w, h[, confidence]` (정규화 YOLO)
- `x1, y1, x2, y2, confidence, class` (Ultralytics)

픽셀 단위의 6열 `class,cx,cy,w,h,confidence`는 다른 6열 형식과 모호하므로 `detection_format="yolo_txt", normalized=False`를 지정합니다.

중앙이 아닌 위치에서 clip했다면 `clip_origin=(full_x, full_y)`를 명시합니다. 생략하면 full image의 정확한 중앙 clip으로 계산합니다.

## 결과값 한눈에 보기

`build_die_map_from_yolo()`의 반환값은 `WaferDieMap` 객체입니다. 이 객체 하나에 wafer 외곽선, 선택된 십자점, pitch, angle, 전체 die map이 모두 들어 있습니다.

```python
dm = build_die_map_from_yolo(...)

print(type(dm))                # WaferDieMap
print(dm.pitch_x, dm.pitch_y)  # X/Y die pitch
print(dm.grid_angle_deg)       # grid 회전각
print(dm.num_dies)             # 생성된 die 개수
```

결과는 다음과 같은 구조입니다.

```text
WaferDieMap (dm)
├─ wafer 중심/반지름
├─ center corner, pitch, angle
├─ grid_estimate       # 512 clip에서 선택한 점과 각도 계산 결과
├─ wafer_boundary      # full image에서 검출한 wafer contour
├─ aligned_image       # angle 보정된 full wafer 이미지
├─ 좌표 변환 matrix    # original ↔ aligned
├─ dies                # 전체 die entry list
└─ dies_by_index       # (ix, iy)로 빠르게 조회하는 dictionary
```

### 1. `WaferDieMap` 주요 필드

| 필드 | 형식 | 의미 |
|---|---|---|
| `wafer_cx`, `wafer_cy` | `int` | full image에서 검출한 wafer 중심 픽셀 |
| `wafer_r` | `int` | wafer 외곽 contour의 최소 외접원 반지름 |
| `x0`, `y0` | `float` | full image 좌표계의 center corner/grid origin |
| `pitch_x`, `pitch_y` | `float` | 옆 점과 아래 점으로 계산한 실제 grid 간격(px) |
| `die_w`, `die_h` | `int` | `round(pitch_x)`, `round(pitch_y)` 호환 필드 |
| `grid_angle_deg` | `float` | 오른쪽 grid축의 영상 기준 기울기(deg) |
| `rotation_deg` | `float` | 기존 API 호환용 각도 필드. 현재는 `grid_angle_deg`와 같은 값 |
| `angle_confidence` | `float` | X축 각도와 Y축 각도의 합의 정도, `0.0~1.0` |
| `pixel_per_unit` | `float` | 픽셀 좌표를 실좌표로 환산할 때 사용하는 `px/unit` |
| `image_shape` | `(H, W)` | full wafer 이미지 크기 |
| `edge_mode` | `str` | `is_edge` 판정 기준: `circle`, `ring`, `both` |
| `dies` | `list[dict]` | wafer 안에 생성된 모든 die 정보 |
| `dies_by_index` | `dict` | `(ix, iy)`를 key로 하는 die 조회 dictionary |
| `num_dies` | `int` property | `len(dm.dies)` |
| `grid_estimate` | `GridEstimate` | 512 clip 좌표에서 계산한 상세 grid 결과 |
| `wafer_boundary` | `WaferBoundary` | 검출된 full-image wafer 외곽선 결과 |
| `aligned_image` | `ndarray \| None` | `grid_angle_deg`만큼 보정된 full wafer BGR 이미지 |
| `original_to_aligned_matrix` | `ndarray` | 원본 좌표를 보정 이미지 좌표로 옮기는 2×3 affine matrix |
| `aligned_to_original_matrix` | `ndarray` | 보정 이미지 좌표를 원본 좌표로 되돌리는 2×3 affine matrix |
| `axis_x`, `axis_y` | `(float, float)` property | 회전된 grid의 단위 X/Y 벡터 |

die map과 `locate_die()`는 기존 호환성을 위해 원본 이미지 좌표계를 유지합니다. `aligned_image`는 추가 반환 결과이며, 두 좌표계가 섞이지 않도록 변환 matrix와 point 변환 함수를 함께 제공합니다.

### 2. 앵글 보정 이미지와 좌표 변환 결과

기본 설정에서는 angle이 보정된 full wafer 이미지가 `dm.aligned_image`에 들어갑니다.

```python
dm = build_die_map_from_yolo(
    wafer_bgr,
    center_clip_bgr,
    yolo_data,
    return_aligned_image=True,  # 기본값
)

assert dm.aligned_image is not None
print(dm.aligned_image.shape)  # 원본과 동일한 (H, W, 3)
cv2.imwrite("wafer_aligned.png", dm.aligned_image)
```

보정 방식:

- 검출한 wafer 중심 `(wafer_cx, wafer_cy)`을 회전 중심으로 사용합니다.
- `grid_angle_deg`를 OpenCV 회전 보정각으로 적용합니다.
- 출력 크기는 원본과 동일합니다.
- 기본 보간법은 `cv2.INTER_CUBIC`입니다.
- 회전으로 새로 생긴 이미지 바깥 영역은 기본 검정 `(0,0,0)`입니다.
- angle이 0이어도 반환 이미지가 원본과 같은 객체가 되지 않도록 복사본을 반환합니다.

10000×10000 BGR 이미지의 추가 300MB 메모리가 부담되면 이미지 생성만 끌 수 있습니다. 좌표 변환 matrix는 계속 반환됩니다.

```python
dm = build_die_map_from_yolo(
    wafer_bgr,
    center_clip_bgr,
    yolo_data,
    return_aligned_image=False,
)

assert dm.aligned_image is None
assert dm.original_to_aligned_matrix is not None
```

원본 이미지의 검사 좌표를 보정 이미지에서 표시하려면 다음 함수를 사용합니다.

```python
from codex.wafer_via import (
    transform_point_to_aligned,
    transform_point_to_original,
)

original_point = (5499.0, 4700.0)
aligned_point = transform_point_to_aligned(dm, original_point)
restored_point = transform_point_to_original(dm, aligned_point)
```

`locate_die()`는 원본 좌표를 받습니다. 보정 이미지에서 새로 검출한 좌표라면 먼저 원본 좌표로 되돌립니다.

```python
aligned_defect_point = (5503.2, 4691.8)
original_defect_point = transform_point_to_original(dm, aligned_defect_point)
result = locate_die(dm, point=original_defect_point)
```

직접 이미지만 보정하고 싶다면 다음 저수준 함수도 사용할 수 있습니다.

```python
aligned, forward_matrix, inverse_matrix = align_wafer_image(
    wafer_bgr,
    center_px=(dm.wafer_cx, dm.wafer_cy),
    angle_deg=dm.grid_angle_deg,
)
```

### 3. `GridEstimate`: 512 clip 분석 결과

```python
grid = dm.grid_estimate
if grid is not None:
    print(grid.center_corner_clip)
    print(grid.side_corner_clip)
    print(grid.below_corner_clip)
    print(grid.to_dict())
```

| 필드 | 의미 |
|---|---|
| `points_clip` | confidence 필터링과 중복 제거 후 사용한 전체 YOLO 십자점 |
| `center_corner_clip` | 512 clip 중심에 가장 가까워 center corner로 선택된 점 |
| `side_corner_clip` | center corner의 같은 row에서 선택한 가장 가까운 옆 점 |
| `below_corner_clip` | center corner의 같은 column에서 선택한 가장 가까운 아래 점 |
| `pitch_x` | center → side 벡터 길이 |
| `pitch_y` | center → below 벡터 길이 |
| `angle_x_deg` | center → side 벡터로 계산한 각도 |
| `angle_y_deg` | center → below 벡터로 계산한 각도 |
| `angle_deg` | `angle_x_deg`와 `angle_y_deg`를 결합한 최종 각도 |
| `angle_confidence` | 두 각도가 얼마나 일치하는지 나타내는 값 |
| `refined` | Lab 경계 기반 미세 보정을 적용했는지 여부 |

`angle_confidence`는 YOLO 모델 confidence 평균이 아닙니다. 두 직교축에서 구한 회전각의 일치도를 나타냅니다.

### 4. `WaferBoundary`: wafer 외곽선 결과

```python
boundary = dm.wafer_boundary
if boundary is not None:
    print(boundary.center_px)
    print(boundary.radius_px)
    print(boundary.bbox_px)
    print(boundary.method)
```

| 필드 | 의미 |
|---|---|
| `center_px` | contour 최소 외접원의 중심 |
| `radius_px` | contour 최소 외접원의 반지름 |
| `contour_px` | OpenCV contour 배열. full image 좌표 기준 |
| `area_px` | contour 면적(px²) |
| `bbox_px` | wafer contour의 `(x1, y1, x2, y2)` bounding box |
| `method` | 선택된 외곽선 후보 방식: Lab border distance 또는 grayscale Otsu 계열 |

정확한 die 포함 여부는 단순 원이 아니라 `contour_px`를 기준으로 계산합니다. `wafer_r`은 표시와 호환을 위한 외접원 값입니다.

### 5. 개별 die entry 결과

```python
die = dm.get_die(ix=2, iy=-3)
if die is not None:
    print(die)
```

| key | 의미 |
|---|---|
| `index` | `(ix, iy)`. X+는 오른쪽, Y+는 위쪽 |
| `center_px` | die 중심의 full-image 픽셀 좌표 |
| `polygon_px` | 회전이 반영된 die의 네 꼭짓점. 실제 die geometry 확인에 사용 |
| `rect_px` | `polygon_px` 전체를 포함하는 축 정렬 bounding rectangle |
| `crop_rect_px` | 현재는 `rect_px`와 동일한 crop 후보 영역 |
| `real_coord` | wafer 중심 기준 die 중심의 실좌표 |
| `is_edge_partial` | die polygon 일부가 wafer contour 밖에 있는지 여부 |
| `is_edge_ring` | 주변 8개 index 중 하나라도 없어 grid 최외곽인지 여부 |
| `is_edge` | `edge_mode`에 따라 선택된 최종 edge 판정 |

회전각이 0이 아니면 `rect_px`에는 polygon 바깥 영역도 포함됩니다. 정확한 die 영역이 필요하면 `polygon_px`를 mask 또는 perspective crop에 사용합니다.

### 6. `locate_die()` 반환 dictionary

```python
result = locate_die(dm, point=(5499, 4700))
# 또는
result = locate_die(dm, bbox=(4880, 5080, 4980, 5180))
```

| key | 의미 |
|---|---|
| `input_type` | 입력 방식: `point` 또는 `bbox` |
| `query_px` | 실제 index 계산에 사용한 좌표. bbox 입력이면 bbox 중심 |
| `die_index` | 계산된 `(ix, iy)` |
| `die_center_px` | 해당 index die의 중심 |
| `die_polygon_px` | 회전된 die의 네 꼭짓점 |
| `die_rect_px` | die polygon을 포함하는 축 정렬 rectangle |
| `crop_rect_px` | 현재 crop 후보 rectangle |
| `real_coord` | query 좌표의 wafer 중심 기준 실좌표 |
| `real_distance` | wafer 중심부터 query까지의 실좌표 거리 |
| `die_real_coord` | query가 아니라 die 중심의 실좌표 |
| `wafer_center_px` | 검출된 wafer 중심 |
| `corner_px` | full-image center corner/grid origin |
| `pitch_x`, `pitch_y` | map 생성에 사용한 pitch |
| `angle_deg` | map 생성에 사용한 회전각 |
| `is_edge_partial` | die 일부가 contour 밖인지 여부 |
| `is_edge_ring` | grid 최외곽인지 여부 |
| `is_edge` | `edge_mode` 기준 최종 edge 여부 |
| `edge_mode` | 현재 edge 판정 방식 |
| `in_wafer` | query 점 자체가 wafer contour 안인지 여부 |

`locate_die()`는 query가 wafer 밖이어도 해석적으로 index와 polygon을 계산합니다. 실제 wafer 내부인지는 반드시 `in_wafer`로 확인합니다.

실좌표 계산식은 다음과 같습니다.

```python
real_x = (query_x - wafer_cx) / pixel_per_unit
real_y = (wafer_cy - query_y) / pixel_per_unit  # 영상 위쪽이 +Y
```

### 7. 오버레이 반환값

```python
clip_overlay = make_clip_overlay(center_clip_bgr, dm.grid_estimate)
wafer_overlay = make_wafer_overlay(wafer_bgr, dm)

print(type(clip_overlay))   # numpy.ndarray
print(type(wafer_overlay))  # numpy.ndarray
```

- 두 함수 모두 OpenCV BGR `numpy.ndarray`를 반환합니다.
- `make_clip_overlay()`는 center/side/below 선택과 pitch/angle을 표시합니다.
- `make_wafer_overlay()`는 wafer contour, center corner, wafer center, die polygon을 표시합니다.
- 저장하지 않고 다음 영상처리 단계에 그대로 전달해도 됩니다.
- `make_wafer_overlay()`는 full image 복사본을 만들기 때문에 10000×10000 BGR 기준 약 300MB가 추가로 필요합니다.

### 8. 테스트에서 확인한 실제 결과 예시

아래 값은 반지름 4700px의 10000×10000 합성 wafer, `pitch_x=90`, `pitch_y=92`, `angle=3.25°` 조건의 검증 결과입니다. 실제 이미지의 결과값은 YOLO 좌표와 wafer 외곽선에 따라 달라집니다.

```python
{
    "wafer_center_px": (5000, 5000),
    "wafer_r": 4700,
    "corner_px": (5000.0, 5000.0),
    "pitch_x": 90.0,
    "pitch_y": 92.0,
    "angle_deg": 3.25,
    "num_dies": 8376,
}
```

index `(2, -3)` die 중심을 다시 `locate_die()`에 입력한 결과입니다.

```python
{
    "input_type": "point",
    "query_px": (5212.0, 5242.0),
    "die_index": (2, -3),
    "die_center_px": (5212, 5242),
    "die_rect_px": (5164, 5193, 5260, 5291),
    "real_coord": (6.625, -7.5625),
    "real_distance": 10.053956,
    "is_edge": False,
    "in_wafer": True,
}
```

### 9. 결과 승인 전에 확인할 항목

1. `center_corner_clip`이 실제 512 clip 중앙의 원하는 십자점인지 확인합니다.
2. `side_corner_clip`과 `below_corner_clip`이 바로 인접한 점인지 확인합니다.
3. `angle_x_deg`와 `angle_y_deg` 차이가 허용 범위 안인지 확인합니다.
4. `pitch_x`, `pitch_y`가 장비의 예상 die 크기와 일치하는지 확인합니다.
5. clip overlay에서 세 기준점과 화살표 방향을 확인합니다.
6. wafer overlay에서 외곽 contour와 회전 die polygon 정합을 확인합니다.
7. 검사 좌표 조회 시 `die_index`뿐 아니라 `in_wafer`, `is_edge`도 함께 확인합니다.

## 각도와 좌표 규칙

- `angle_deg > 0`: 오른쪽 이웃으로 갈수록 영상 Y가 증가하는 기울기입니다.
- 같은 값은 `cv2.getRotationMatrix2D(..., angle_deg, 1)`로 수평 보정할 때 쓸 수 있습니다.
- die lattice와 `locate_die()`는 원본 좌표를 유지합니다. `aligned_image`에서 얻은 좌표만 `transform_point_to_original()`로 되돌립니다.
- index는 기존 규칙처럼 `X+ = 오른쪽`, `Y+ = 위쪽`입니다.

## 구현 위치(Edit map)

- `[SECTOR: 10_YOLO_COORDINATES]`: YOLO txt/point/bbox 파싱
- `[SECTOR: 20_COLOR_INVARIANT_REFINEMENT]`: 선택형 Lab 미세 보정
- `[SECTOR: 30_GRID_ESTIMATION]`: center/side/below, pitch, angle
- `[SECTOR: 40_WAFER_BOUNDARY]`: wafer 외곽 contour
- `[SECTOR: 50_DIE_MAP]`: 회전 die map 생성
- `[SECTOR: 60_LOCATE_DIE]`: point/bbox → die index
- `[SECTOR: 65_ANGLE_ALIGNED_IMAGE]`: angle 보정 이미지와 원본↔보정 좌표 변환
- `[SECTOR: 70_OVERLAY]`: clip/wafer overlay
- `[SECTOR: 80_PIPELINE]`: end-to-end API
- `[SECTOR: 90_USAGE_REFERENCE]`: 단일 파일 복사용 상세 한국어 주석 예제

## 검증

```powershell
python -m py_compile codex\wafer_via.py tests\test_wafer_via.py
python -m unittest discover -s tests -v
```
