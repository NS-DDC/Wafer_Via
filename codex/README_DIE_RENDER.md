# Wafer_Via die_render variant

`wafer_via_die_render.py`는 기존 `wafer_via.py`를 변경하지 않고 V5의 full-wafer `die_render` angle 측정만 추가한 별도 버전입니다.

## 사용법

두 파일을 같은 폴더에 둡니다.

```text
codex/
├─ wafer_via.py
└─ wafer_via_die_render.py
```

Import 대상만 별도 버전으로 변경합니다.

```python
from codex.wafer_via_die_render import build_die_map_from_yolo

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,
    detections=results[0].boxes.xywh.cpu().numpy(),
    detection_format="xywh",
    clip_origin=(clip_x, clip_y),

    angle_align_method="die_render",
)
```

나머지 인수와 반환 형식은 기존 `wafer_via.build_die_map_from_yolo()`와 같습니다.

## 지원 방식

```python
# V5 full-wafer projection + FFT
dm = build_die_map_from_yolo(..., angle_align_method="die_render")

# 기존 512 clip YOLO robust angle
dm = build_die_map_from_yolo(..., angle_align_method="yolo")
```

`die_render`가 기본값입니다. Full wafer에서 angle 신호를 찾지 못하면 기본적으로 기존 YOLO angle로 fallback합니다.

```python
dm = build_die_map_from_yolo(
    ...,
    angle_align_method="die_render",
    die_render_fallback_to_yolo=True,
)
```

## 원리

[NS-DDC/Wafer_Map_Die_V5](https://github.com/NS-DDC/Wafer_Map_Die_V5)의 `die_render` angle 방식을 현재 YOLO 기반 DM 파이프라인에 맞게 분리했습니다.

1. Full wafer 중심 ROI를 최대 1400px로 축소
2. Otsu 이진화 후 후보 angle별 X/Y projection variance 계산
3. Coarse scan → fine scan → 포물선 보간
4. 2D FFT 스펙트럼 angle과 독립 교차검증
5. 불일치하면 FFT 주변 탐색과 ±44° full scan 후보 비교
6. 잔여 angle이 작아질 때까지 최대 3회 반복
7. 결정된 full-wafer angle로 DM과 `aligned_image`만 다시 생성

YOLO 중심 보정, `(0,0)` 선택, pitch, wafer 경계, edge clipping, index와 `locate_die()`는 기존 구현을 그대로 사용합니다.

## 결과 확인

```python
print(dm.angle_align_method)  # die_render | yolo_fallback | yolo
print(dm.grid_angle_deg)      # DM/aligned image에 실제 사용한 angle
print(dm.yolo_angle_deg)      # 512 clip YOLO angle 비교값
print(dm.angle_confidence)
print(dm.angle_agree)         # projection과 FFT 합의 여부
print(dm.die_render_info)
```

`die_render`는 개별 YOLO 좌표쌍이 아니라 full-wafer ROI의 전체 픽셀을 사용합니다. 따라서 최종 angle의 근거를 잘못 표시하지 않도록 `dm.angle_pairs_full`과 `dm.grid_estimate.angle_pairs_clip`은 비어 있습니다. 기존 YOLO angle에 사용된 비교용 좌표쌍은 아래처럼 확인할 수 있습니다.

```python
print(dm.yolo_angle_pairs_full)
print(dm.yolo_angle_pairs_raw_full)
```

`yolo_fallback` 또는 `angle_align_method="yolo"`일 때는 기존 `angle_pairs_full`을 그대로 유지합니다.

`die_render_info` 예시:

```python
{
    "source": "die_render",
    "total_angle": 2.3869,
    "projection": 2.3869,
    "fft": 2.2958,
    "confidence": 0.97,
    "agree": True,
    "iteration_deltas": (2.3869, 0.0001),
    "final_residual": 0.0001,
}
```

## 튜닝 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `die_render_roi_ratio` | `0.55` | wafer 반지름 대비 중앙 ROI 비율 |
| `die_render_max_dim` | `1400` | projection ROI 최대 크기 |
| `die_render_fft_max_dim` | `1024` | FFT ROI 최대 크기 |
| `die_render_search_deg` | `6.0` | 첫 projection 탐색 범위 |
| `die_render_coarse_step` | `0.15` | coarse angle 간격 |
| `die_render_fine_step` | `0.02` | fine angle 간격 |
| `die_render_agree_tol_deg` | `0.40` | projection/FFT 합의 허용값 |
| `die_render_full_scan_deg` | `44.0` | 불일치 시 광역 탐색 범위 |
| `die_render_max_iter` | `3` | 반복 수렴 최대 횟수 |
| `die_render_min_angle_deg` | `0.01` | 이보다 작은 잔차에서 종료 |

10000×10000 이미지에서 처리 시간이 길면 먼저 `die_render_max_dim`을 `1000~1200`으로 낮춰 비교하십시오. 정밀도가 부족하면 `1400~1800` 범위에서 높일 수 있습니다.

## 직접 angle만 측정

```python
from codex.wafer_via_die_render import measure_wafer_angle_die_render

info = measure_wafer_angle_die_render(
    wafer_bgr,
    wafer_cx=dm.wafer_cx,
    wafer_cy=dm.wafer_cy,
    wafer_r=dm.wafer_r,
)

print(info)
```

## 주의사항

- Full wafer에서 실제 die/sawline 주기 패턴이 보여야 합니다.
- Particle, 회로 내부 패턴 또는 banding이 격자보다 강하면 projection/FFT가 다른 주기를 선택할 수 있습니다.
- `angle_agree=False`이면 `projection`, `fft`, `yolo_angle_deg`를 함께 비교하십시오.
- 위치별 pitch/angle이 달라지는 렌즈 왜곡이나 원근 왜곡은 단일 회전으로 해결되지 않습니다.
- 별도 버전은 `wafer_via.py`를 import하므로 두 파일을 함께 배포해야 합니다.
