# notch-only angle 리팩터링 및 출력 성능 측정

## 변경 범위

현재 권장 복붙 파일 `wafer_via_notch_standalone.py`에서 angle의 유일한 입력은
`detect_wafer_notch()` 결과입니다. YOLO 좌표는 중심 corner와 X/Y pitch 선택에만
사용합니다.

생성 파일에서 제거한 항목은 다음과 같습니다.

- YOLO 좌표쌍 기반 robust angle 추정
- 중심점 인접 두 벡터 기반 local angle 추정
- robust/local 선택 옵션과 angle pair 진단 필드
- 예전 YOLO angle end-to-end builder 및 사용 예제

그 결과 기본 복붙 파일은 4,842줄에서 3,966줄로 876줄 줄었습니다.

수동 ROI의 반원/반타원 notch와 ROI 배경 분할 로직은
`wafer_notch_angle.py` 원본 모듈로 옮겼습니다. 따라서 생성기를 다시 실행해도
기능이 사라지지 않습니다.

## 기본 출력

```python
dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,
    detections=yolo_points,
    notch_roi_center_px=(5000, 9650),
    return_aligned_image=True,
    return_notch_visuals=False,  # 기본값
)

assert dm.aligned_image is not None
assert dm.notch_overlay_image is None
assert dm.notch_zoom_image is None
```

진단이 필요한 경우에만 `return_notch_visuals=True`를 지정합니다.

## 10000×10000 실측

- 입력: `image5/wafer_pale_green_wide_shallow_10000x10000.png`
- ROI 중심: `(5000, 9650)`
- ROI 반크기: `(600, 600)`
- notch 가로 반폭 범위: `(30, 80)` px
- 출력 overlay 제한: 5000×5000
- 측정: warm-up 후 각 모드를 4회 교차 실행, 중앙값
- 디스크 읽기/저장은 측정에서 제외

| 모드 | 전체 builder 중앙값 |
|---|---:|
| angle 보정 이미지 1장만 | 3.528630초 |
| 보정 이미지 + overlay + zoom | 3.567353초 |
| 절감 | 0.038722초 |

보정 이미지만 반환하면 전체 builder 지연시간은 약 **1.09% 감소**하고,
동일 시간 기준 처리율은 약 **1.10% 증가**했습니다.

이미지 단계만 분리한 5회 중앙값은 다음과 같습니다.

| 단계 | 중앙값 |
|---|---:|
| notch 검출 | 0.887618초 |
| 10000×10000 affine 회전 | 0.318078초 |
| 5000×5000 overlay | 0.030259초 |
| 1024×1024 zoom | 0.002473초 |

overlay와 zoom을 생략하면 이 분리 측정에서는 약 0.032732초, 2.64%의
지연시간이 줄었습니다. 전체 builder에는 12,609개 die 생성 비용도 포함되므로
상대 속도 차이가 약 1.09%로 작아집니다.

메모리 이득은 더 큽니다. uint8 BGR 기준 보정 이미지는 300MB이며,
5000×5000 overlay와 1024×1024 zoom은 합계 약 78.1MB입니다. 기본 모드에서는
이 추가 진단 이미지 배열을 만들지 않습니다.

수치는 현재 PC와 합성 테스트 이미지 기준입니다. 실제 장비 이미지에서는 notch
검출 난이도, OpenCV thread 수, die 개수에 따라 달라질 수 있습니다.
