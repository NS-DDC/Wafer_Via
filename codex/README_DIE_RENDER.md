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
# V5 die grid angle (기본값)
dm = build_die_map_from_yolo(..., angle_align_method="die_render")

# 기존 512 clip YOLO robust angle (명시할 때만)
dm = build_die_map_from_yolo(..., angle_align_method="yolo")
```

`die_render`가 기본값이며, **YOLO angle로 자동 fallback하지 않습니다.** Full wafer에서 angle 신호를 찾지 못하면 보정을 포기하고 angle `0.0`(무회전)으로 두며, `angle_align_method`는 `die_render_no_signal`이 됩니다. YOLO angle은 `angle_align_method="yolo"`를 직접 지정할 때만 사용됩니다.

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

1. Full wafer 중심 ROI를 최대 1400px로 축소
2. Otsu 이진화 후 후보 angle별 X/Y projection variance 계산
3. Coarse scan → fine scan → 포물선 보간
4. 2D FFT 스펙트럼 angle과 독립 교차검증
5. 불일치하면 FFT 주변 탐색과 ±44° full scan 후보 비교
6. 격자 신호 세기(prominence)가 `15.0` 미만이면 격자가 없다고 보고 중단
7. V5의 `measure_die_grid_angle`(die grid cue)로 angle을 한 번 더 독립 측정
8. **projection** 잔차가 작아질 때까지 최대 3회 반복 (V5 `align_wafer_by_die_render`와 동일)
9. 최종 angle은 grid cue 값을 채택하고, 그 angle로 DM과 `aligned_image`만 다시 생성

### 반복이 projection으로 도는 이유

V5의 `align_wafer_by_die_render`는 `measure_wafer_angle_robust`(projection+FFT)로만 반복하고, grid cue는 반복 밖에서 한 번만 씁니다. 이 파일도 같은 구조입니다.

grid cue를 반복에 넣으면 안 됩니다. grid cue는 **완벽히 축에 정렬된 이미지를 약 `-0.16°`로 잘못 읽는 사각지대**가 있습니다. 반복은 이미지를 매번 그 사각지대 쪽으로 더 밀어 넣으므로, 오차가 상쇄되지 않고 누적됩니다. 실제로 정답이 `2.4°`인 합성 웨이퍼에서 grid로 반복하면 `[2.4, -0.16, -0.16]`이 되어 `2.08°`로 끝납니다(오차 `0.32°`). projection으로 반복하면 `[2.387, 0.0]`으로 수렴해 정답과 정확히 일치합니다.

반대로 **최종 보고값은 grid cue**입니다. grid cue는 입력 이미지의 절대 기울기를 한 번에 읽으므로, 반복은 projection 추정을 수렴시키는 용도일 뿐입니다.

YOLO 중심 보정, `(0,0)` 선택, pitch, wafer 경계, edge clipping, index와 `locate_die()`는 기존 구현을 그대로 사용합니다.

## 결과 확인

```python
print(dm.angle_align_method)  # die_render | die_render_no_signal | off | yolo
print(dm.grid_angle_deg)      # DM/aligned image에 실제 사용한 angle
print(dm.yolo_angle_deg)      # 512 clip YOLO angle 비교값 (사용되지는 않음)
print(dm.angle_confidence)
print(dm.angle_agree)         # projection과 FFT 합의 여부
print(dm.die_render_info)
```

| `angle_align_method` | 의미 |
|---|---|
| `die_render` | V5 die grid angle로 보정함 |
| `die_render_no_signal` | 격자 신호가 없어 보정을 포기함 (angle `0.0`) |
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
    "angle": 1.2400,            # 최종 채택값 = grid
    "grid": 1.2400,             # V5 die grid cue
    "grid_agree": True,         # projection이 tol(0.25) 안에서 동의하는가
    "projection": 1.2498,       # projection 반복 결과
    "fft": 1.2442,
    "prominence": 179.6,        # 격자 신호 세기 (게이트 기준 15.0)
    "confidence": 0.97,
    "agree": True,              # projection과 FFT 합의 여부
    "total_angle": 1.2400,
    "iteration_deltas": (1.2498, 0.0000),
    "final_residual": 0.0000,
}
```

`grid_agree`는 **보고 전용**입니다. `False`여도 grid 값을 그대로 채택합니다(아래 참고). 두 신호가 갈라졌다는 사실을 숨기지 않기 위한 진단 플래그입니다.

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
| `die_render_min_angle_deg` | `0.01` | 이보다 작은 projection 잔차에서 종료 |
| `die_render_min_prominence` | `15.0` | 이보다 약한 격자 신호는 격자 없음으로 판정 |
| `die_render_use_grid_cue` | `True` | V5 die grid cue를 함께 측정할지 |
| `die_render_prefer_grid_angle` | `True` | 최종 angle로 grid cue 값을 채택할지 |
| `die_render_grid_cue_tol_deg` | `0.25` | `grid_agree` 판정 허용값 (채택 여부와는 무관) |
| `angle_align_enabled` | `True` | angle 보정 전체 on/off |

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
- **`prefer_grid_angle=True`(기본값)의 감수 비용**: grid cue가 어긋나도 거부하지 않습니다. 대비를 `0.06`배로 낮추고 블러 `3.0`, 노이즈 `26`을 준 열화 이미지에서는 grid가 projection과 최대 `1.33°`까지 벌어졌고, 기본값은 그래도 grid를 따랐습니다. 이때 `grid_agree`는 `False`로 남으므로, 열화가 심한 입력을 다룬다면 `grid_agree`를 확인하거나 `die_render_prefer_grid_angle=False`로 projection을 쓰십시오.
- 위치별 pitch/angle이 달라지는 렌즈 왜곡이나 원근 왜곡은 단일 회전으로 해결되지 않습니다.
- `wafer_via_die_render.py` 하나에 기본 YOLO 파이프라인과 `die_render` 방식이 모두 들어 있습니다.
