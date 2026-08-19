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

## 각도와 좌표 규칙

- `angle_deg > 0`: 오른쪽 이웃으로 갈수록 영상 Y가 증가하는 기울기입니다.
- 같은 값은 `cv2.getRotationMatrix2D(..., angle_deg, 1)`로 수평 보정할 때 쓸 수 있습니다.
- 이미지는 실제로 회전시키지 않습니다. die lattice를 원본 좌표에서 회전해 만들므로 YOLO/검사 좌표를 다시 변환할 필요가 없습니다.
- index는 기존 규칙처럼 `X+ = 오른쪽`, `Y+ = 위쪽`입니다.

## 구현 위치(Edit map)

- `[SECTOR: 10_YOLO_COORDINATES]`: YOLO txt/point/bbox 파싱
- `[SECTOR: 20_COLOR_INVARIANT_REFINEMENT]`: 선택형 Lab 미세 보정
- `[SECTOR: 30_GRID_ESTIMATION]`: center/side/below, pitch, angle
- `[SECTOR: 40_WAFER_BOUNDARY]`: wafer 외곽 contour
- `[SECTOR: 50_DIE_MAP]`: 회전 die map 생성
- `[SECTOR: 60_LOCATE_DIE]`: point/bbox → die index
- `[SECTOR: 70_OVERLAY]`: clip/wafer overlay
- `[SECTOR: 80_PIPELINE]`: end-to-end API

## 검증

```powershell
python -m py_compile codex\wafer_via.py tests\test_wafer_via.py
python -m unittest discover -s tests -v
```
