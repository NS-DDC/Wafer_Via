# V5 외곽 원 + 가변 배경 Notch 진단 버전

## 목적

기존 `wafer_notch_angle.py`, `wafer_via_notch.py`,
`wafer_via_notch_standalone.py`는 변경하지 않았습니다.

새 파일 [`wafer_notch_v5_adaptive_background.py`](./wafer_notch_v5_adaptive_background.py)는
V5에서 실제로 잘 동작했던 다음 기하 로직을 별도로 유지합니다.

1. 가장 큰 Wafer contour 탐색
2. `cv2.minEnclosingCircle`로 외곽 원 계산
3. 아래쪽 sector의 외곽을 방사형으로 스캔
4. 외곽이 안쪽으로 파인 연속 구간을 notch로 선택
5. 기준 90° 선과 검출된 notch 선 사이의 정렬 잔여각 계산

차이점은 `gray > 20`으로 검정 배경만 제거하지 않고, 영상 테두리에서 실제
배경색을 LAB 색공간으로 학습한다는 점입니다. 따라서 검정·파랑·분홍처럼
배경색이 달라도 동일한 Wafer 마스크를 만들 수 있습니다.

이 파일은 로컬 모듈을 import하지 않는 **단일 복붙용 파일**입니다.

## 제공 샘플 실측 결과

테스트 원본:

```text
E:\mirero\wafer_via\Make_Sample\wafer_edge_noise_natural_v2_black.png
E:\mirero\wafer_via\Make_Sample\wafer_edge_noise_natural_v2_blue.png
E:\mirero\wafer_via\Make_Sample\wafer_edge_noise_natural_v2_pink.png
```

기존 검정 배경 전용 함수의 결과:

| 배경 | found | 중심 | 반지름 | 결과 |
|---|---:|---:|---:|---|
| black | True | `(1023, 1023)` | `963 px` | 정상 |
| blue | False | `(1024, 1024)` | `1447 px` | 화면 전체를 Wafer로 판단 |
| pink | False | `(1024, 1024)` | `1447 px` | 화면 전체를 Wafer로 판단 |

새 adaptive-background 파일의 결과:

| 배경 | found | Wafer 중심 | 반지름 | 외곽 원 위 notch 점 | notch angle | 잔여각 |
|---|---:|---:|---:|---:|---:|---:|
| black | True | `(1023.46, 1023.48)` | `963.46 px` | `(1023.91, 1986.94)` | `89.9733°` | `-0.0267°` |
| blue | True | `(1023.46, 1023.48)` | `963.46 px` | `(1023.91, 1986.94)` | `89.9733°` | `-0.0267°` |
| pink | True | `(1023.46, 1023.48)` | `963.46 px` | `(1023.91, 1986.94)` | `89.9733°` | `-0.0267°` |

세 영상은 배경색만 다르며, 새 버전에서는 모든 기하 좌표가 동일하게
나왔습니다.

![adaptive background notch sample results](sample_img/adaptive_background_notch_samples_contact_sheet.png)

## Python 사용법

```python
import cv2
from wafer_notch_v5_adaptive_background import (
    draw_aligned_wafer_notch_guide,
)

guide = draw_aligned_wafer_notch_guide(
    dm.aligned_image,             # BGR ndarray 또는 이미지 경로
    reference_angle_deg=90.0,     # 정상 notch 방향: 아래쪽
    failure_mode="zero",          # 미검출이어도 외곽 원은 반환
)

print(guide.found)
print(guide.wafer_center_px)
print(guide.wafer_radius_px)
print(guide.notch_center_px)       # 실제 파임 내부의 깊이 가중 중심
print(guide.notch_point_px)        # 외곽 원 위 최종 방향점
print(guide.notch_left_px)
print(guide.notch_right_px)
print(guide.notch_angle_deg)
print(guide.residual_angle_deg)    # aligned image에 남은 각도 오차
print(guide.background_palette_bgr)
print(guide.segmentation_threshold_lab)

# overlay_image는 입력과 같은 크기의 writable BGR 복사본입니다.
# 사용자가 판단한 정답점을 직접 추가할 수 있습니다.
manual_image = guide.overlay_image
answer_xy = (1024, 1987)
cv2.circle(manual_image, answer_xy, 16, (255, 0, 0), -1, cv2.LINE_AA)
cv2.putText(
    manual_image,
    "MY ANSWER",
    (answer_xy[0] + 24, answer_xy[1]),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2,
    cv2.LINE_AA,
)
cv2.imwrite("adaptive_notch_manual.png", manual_image)
```

함수는 입력 영상을 회전하지 않으며 DM도 생성하지 않습니다. 기존 파이프라인과
독립적으로 결과를 확인하고 정답을 표시하기 위한 함수입니다.

## 명령행 사용법

```powershell
cd E:\mirero\Wafer_V7_Codex\codex

python wafer_notch_v5_adaptive_background.py `
  E:\mirero\wafer_via\Make_Sample\wafer_edge_noise_natural_v2_blue.png `
  --output adaptive_blue_overlay.png `
  --failure-mode zero
```

## 반환값

| 필드 | 의미 |
|---|---|
| `overlay_image` | 직접 추가로 그릴 수 있는 원본 크기 BGR 이미지 |
| `found` | notch 검출 여부 |
| `wafer_center_px` | aligned image 좌표의 Wafer 중심 |
| `wafer_radius_px` | V5 `minEnclosingCircle` 반지름 |
| `notch_center_px` | notch 안쪽 깊이 가중 중심 |
| `notch_point_px` | 중심 방향을 외곽 원까지 투영한 빨간 기준점 |
| `notch_left_px`, `notch_right_px` | 검출된 notch 구간 양 끝 |
| `notch_angle_deg` | 영상 좌표 각도, 오른쪽 0°, 아래쪽 90° |
| `residual_angle_deg` | 검출각과 `reference_angle_deg`의 차이 |
| `notch_depth_px` | notch 최대 깊이 |
| `notch_width_deg` | notch 후보의 각도 폭 |
| `background_palette_bgr` | 테두리에서 학습한 배경색 목록 |
| `segmentation_threshold_lab` | 배경과 Wafer를 나눈 LAB 거리 임계값 |
| `analysis_scale` | 큰 영상을 분석할 때 적용한 축소 비율 |

모든 좌표와 길이는 축소 분석 여부와 관계없이 원본 aligned image 기준입니다.

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `max_analysis_dimension` | `3072` | 대형 영상 분석 축소 최대 변 |
| `border_band_px` | `16` | 배경을 학습할 원본 영상 테두리 폭 |
| `background_palette_size` | `3` | 테두리의 배경색 군집 최대 개수 |
| `background_distance_threshold_lab` | `None` | LAB 임계값 수동 지정, 기본은 자동 |
| `background_noise_margin_lab` | `6.0` | 테두리 색 분산에 더하는 여유값 |
| `min_background_distance_lab` | `8.0` | 자동 임계값의 절대 하한 |
| `wafer_morph_kernel` | `25` | V5 외곽 원 마스크 close/open 크기 |
| `silhouette_open_kernel` | `3` | notch 실루엣 노이즈 제거 크기 |
| `angle_samples` | `14400` | 원주 각도 샘플 수, 0.025° 간격 |
| `min_notch_depth_px` | `4.0` | 원본 해상도 기준 최소 파임 깊이 |
| `failure_mode` | `"zero"` | `"zero"`는 원만 반환, `"error"`는 예외 |

## 확인할 점

- 영상 테두리에 실제 배경이 보여야 합니다. Wafer나 지그가 테두리 전체를
  가리면 배경색을 학습할 수 없습니다.
- 실제 장비 영상에서 테두리 배경색이 매우 불균일하면
  `background_palette_size`를 늘릴 수 있습니다.
- Wafer 색과 배경색이 거의 같으면 `background_distance_threshold_lab`를
  낮춰야 할 수 있습니다.
- 본 샘플 검증은 제공된 세 이미지에 대한 결과입니다. 다른 장비 영상은
  반환 overlay에서 하늘색 외곽 원과 빨간 notch 점을 다시 확인해야 합니다.

## 검증

- 제공된 2048×2048 black/blue/pink 샘플 3개 실제 실행 성공
- 배경색 3종에서 중심·반지름·notch 좌표·각도 결과 동일
- 별도 합성 회귀 테스트 추가
- 새 파일 하나만 빈 폴더에 복사한 후 함수 호출 성공
- 기존 notch/DM 소스 파일 변경 없음
