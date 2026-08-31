# 고정 ROI 반원 notch angle 보정 사용법

## 목적

장비에서 wafer notch가 항상 비슷한 좌표에 나타날 때, 사용자가 지정한 원본
이미지 ROI 안에서만 U자 반원을 찾고 그 방향으로 full wafer 이미지와 die-map을
회전 보정합니다. ROI 밖의 외곽선, die street, 장식성 원형 edge는 angle 후보가
될 수 없습니다.

복사할 파일은 아래 하나입니다.

```text
wafer_via_notch_standalone.py
```

외부 패키지는 `numpy`, `opencv-python`만 필요합니다.

## 가장 간단한 호출

```python
from wafer_via_notch_standalone import build_die_map_from_yolo

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,             # full wafer BGR ndarray 또는 경로
    clip_image=center_clip_bgr,         # YOLO를 수행한 중앙 clip
    detections=yolo_points,
    detection_format="point",
    clip_origin=(clip_x, clip_y),

    # 원본 full wafer 이미지 좌표입니다.
    notch_roi_center_px=(5000, 9650),
    notch_roi_half_size_px=(600, 600),

    # 검출된 notch를 정렬 후 6시 방향에 둡니다.
    notch_reference_angle_deg=90.0,

    # 잘못된 angle로 계속 진행하지 않도록 운영에서는 error를 권장합니다.
    notch_failure_mode="error",
)
```

`image5`의 10000×10000 샘플은 `(5000, 9650)` 부근이지만 실제 장비 이미지에서는
오버레이를 확인해 좌표를 조정해야 합니다. ROI 좌표와 크기는 축소 분석 이미지가
아니라 항상 원본 `wafer_image` 픽셀 단위입니다.

## notch 크기를 알고 있을 때

장비에서 notch 크기도 일정하면 반지름 범위를 추가해 오검출을 더 줄일 수 있습니다.

```python
dm = build_die_map_from_yolo(
    ...,
    notch_roi_center_px=(5000, 9650),
    notch_roi_half_size_px=(600, 600),
    notch_semicircle_radius_range_px=(50, 150),
    notch_semicircle_min_score=0.55,
    notch_failure_mode="error",
)
```

반지름 범위도 원본 이미지 픽셀 단위입니다. 실제 크기를 모르면 이 옵션은 생략하고
자동 범위를 사용하십시오.

## angle 보정 원리

1. 전체 영상의 LAB 색 변화량으로 wafer 외곽 원을 fitting합니다.
2. 사용자가 준 ROI 밖의 notch 후보를 제거합니다.
3. ROI 안에서 작은 원 후보를 찾습니다.
4. wafer 중심을 향하는 약 180° inward arc의 edge 연속성·대칭성을 검사합니다.
5. 여러 후보의 실제 arc 점을 robust circle-fit하여 잔차가 낮은 반원을 선택합니다.
6. `wafer 중심 → 반원 중심` 방향을 `notch_angle_deg`로 사용합니다.
7. notch가 `notch_reference_angle_deg`에 오도록 full wafer 이미지를 회전합니다.
8. 같은 affine matrix로 YOLO 기준점과 die-map 좌표를 변환합니다.

이미지 좌표 angle은 오른쪽 `0°`, 아래쪽 `90°`, 왼쪽 `180°`, 위쪽 `270°`입니다.

```text
correction_angle_deg = notch_angle_deg - notch_reference_angle_deg
```

반환된 `dm.aligned_image`에는 이 보정각이 적용됩니다. 반환 die-map은 이미 정렬된
좌표계이므로 `dm.grid_angle_deg == 0.0`입니다.

## 반드시 확인할 결과

```python
print(dm.notch_result.found)
print(dm.notch_angle_deg)
print(dm.notch_correction_angle_deg)
print(dm.image_rotation_deg)

print(dm.notch_roi_center_px)
print(dm.notch_roi_bounds_px)
print(dm.notch_semicircle_center_px)
print(dm.notch_semicircle_radius_px)
print(dm.notch_semicircle_score)
print(dm.notch_semicircle_fit_residual_px)

print(dm.coordinate_space)   # "aligned_image"
print(dm.grid_angle_deg)     # 0.0
```

`semicircle_score`가 높더라도 오버레이 확인을 생략하면 안 됩니다. 실제 arc fitting이
좋을수록 `semicircle_fit_residual_px`가 작습니다. 카메라 해상도와 blur가 달라지므로
고정 합격값은 실제 양품 데이터로 정해야 합니다.

## 오버레이 저장과 판독

```python
import cv2

cv2.imwrite("notch_roi_overview.png", dm.notch_overlay_image)
cv2.imwrite("notch_roi_zoom.png", dm.notch_zoom_image)
cv2.imwrite("wafer_aligned.png", dm.aligned_image)
```

- 자홍색 사각형: 실제 연산을 허용한 ROI
- 자홍색 십자: 사용자가 입력한 예상 반원 중심
- 하늘색 반원 arc와 점: robust fitting된 반원과 중심
- 노란 arc: angle 계산에 사용한 inward edge 구간
- 빨간점: wafer 외곽 원 위의 최종 notch 방향점
- 초록선: wafer 중심에서 notch 방향으로 향하는 angle 벡터
- 하늘색 큰 원: fitting된 wafer 외곽 원

하늘색 notch 표시는 전체 원을 크게 그리지 않고, 실제 계산에 사용한 inward 반원
arc만 표시합니다.

![ROI 반원 notch 검출 예시](sample_img/notch_roi_semicircle_preview.png)

## notch만 먼저 검증

YOLO와 die-map을 실행하기 전에 notch 검출만 독립적으로 확인할 수 있습니다.

```python
from wafer_via_notch_standalone import detect_wafer_notch, make_notch_overlay

result = detect_wafer_notch(
    wafer_bgr,
    notch_roi_center_px=(5000, 9650),
    notch_roi_half_size_px=(600, 600),
    failure_mode="error",
)

print(result.notch_angle_deg)
print(result.correction_angle_deg)
print(result.semicircle_fit_residual_px)

overlay = make_notch_overlay(wafer_bgr, result, max_dimension=2048)
cv2.imwrite("notch_roi_check.png", overlay)
```

## 미검출 정책

운영에서는 다음 설정을 권장합니다.

```python
notch_failure_mode="error"
```

반원을 못 찾으면 `RuntimeError`를 발생시켜 잘못된 wafer angle로 다음 검사를 진행하지
않습니다. `"zero"`는 테스트나 notch 없는 이미지용이며 미검출 시 회전각을 `0°`로
두고 계속 진행합니다.

## 실제 장비 적용 전 점검

`image5` 결과는 제공된 테스트 이미지에 대한 기하 검증입니다. 실제 카메라의 반사,
blur, 잘림, 배경 변화까지 보장하는 생산 검증은 아닙니다.

1. 자홍색 ROI가 실제 notch 전체를 포함하는지 확인합니다.
2. 하늘색 반원 arc가 U자 내부 경계를 따라가는지 확인합니다.
3. 빨간점이 실제 notch 중심 방향에 놓이는지 확인합니다.
4. 여러 정상 wafer에서 angle과 fit residual의 분포를 기록합니다.
5. 생산 합격 기준은 그 정상 데이터 분포로 정합니다.
