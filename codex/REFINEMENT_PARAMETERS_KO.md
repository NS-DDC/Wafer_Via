# 중심 미세 보정 파라미터 가이드

이 문서는 V7의 `wafer_via.py` 구현을 기준으로 합니다. 보정은 외부 모델이 검출한 각 bbox의 중심 주변을 검사해 **실제로 격자 계산에 사용할 점**을 정하는 단계입니다. 모델을 다시 실행하거나 검출 박스를 새로 만들지 않습니다. 기본값은 계속 `refine=True`, `refine_mode="auto"`이며, 새 `cross_lines` 모드는 명시적으로 선택합니다.

모든 좌표·반경·폭은 **모델 입력 clip의 pixel** 기준입니다. 원본 wafer 좌표나 정규화 좌표를 그대로 넣어 보정 범위를 조정하면 안 됩니다. `raw_points_clip`에는 입력 형식 변환, 검출 confidence 필터링, 중복 제거를 마친 모델 중심이 들어갑니다.

## 1. 새 모드 사용 예

다음 값은 API 기본값을 사용한 시작 예시입니다. 실제 장비 이미지에서 최적이라고 검증된 조합을 뜻하지 않습니다.

```python
from codex.wafer_via import build_die_map_from_yolo

dm = build_die_map_from_yolo(
    wafer_image=wafer_bgr,
    clip_image=center_clip_bgr,
    detections=detections_xywh,       # clip pixel 좌표 [cx, cy, w, h]
    detection_format="xywh",
    normalized=False,
    clip_origin=(clip_x, clip_y),    # 원본 wafer에서 clip의 실제 왼쪽 위
    refine=True,
    refine_mode="cross_lines",
    refine_radius=18,
    refine_noise_kernel=5,
    refine_max_street_width=None,
    refine_min_confidence=0.15,
)
```

`cross_lines`는 ROI 네 모서리의 색을 die의 기준색으로 사용하지 않습니다. 교차로의 위·아래 및 왼쪽·오른쪽으로 이어지는 경계와 띠 내부의 일관성을 검사합니다. **색 계산 자체를 모두 제거한 모드는 아닙니다.** Lab 경계를 구하고, 서로 이어져야 할 street 내부 색이 일치하는지 확인합니다. 흰색 street와 흰색 die가 접해 한쪽 팔의 경계 하나가 보이지 않아도, 반대 팔에서 그 경계를 확인할 수 있는 경우를 허용합니다.

## 2. 여덟 파라미터의 기본값과 적용 범위

`build_die_map_from_yolo()`와 `estimate_grid_from_yolo()`는 아래 이름을 사용합니다. 단일 점 함수 `refine_cross_point()`의 대응 인자는 뒤 표에 정리했습니다.

| 파라미터 | 기본값·단위 | 입력 처리·유효 범위 | 적용 및 값 변화의 의미 |
|---|---|---|---|
| `refine` | `True`, bool | bool을 전달합니다. 내부에서는 truthy/falsy로 분기합니다. | `False`면 영상 보정만 생략하고 모델 중심으로 격자를 계산합니다. 검출 필터링·중복 제거·pitch/angle 계산·wafer 외곽 검출은 계속 실행합니다. |
| `refine_mode` | `"auto"`, 문자열 | `auto`, `corner_color`, `gradient`, `cross_lines`. 보정 호출 시 다른 값은 `ValueError`. | `auto`는 색·경계 후보를 결합하는 기존 모드입니다. 새 모드로 자동 전환하는 옵션이 아닙니다. |
| `refine_radius` | `18`, px | `r = max(6, int(value))`. 명시적 상한은 없으며 정수를 권장합니다. | ROI 한 변은 `2r+1`: 18이면 37×37. 키우면 더 넓은 문맥과 후보를 보지만 다른 패턴·이미지 경계를 포함할 수 있습니다. `cross_lines`는 추가 팔 길이 조건 때문에 **r=6~11에서는 항상 원점 유지**, r≥12부터 후보 검사가 가능합니다. |
| `refine_max_street_width` | `None`, px | `int(value or max(8, r))`. `None`과 `0`은 자동값. 경계쌍 폭은 최소 3px입니다. | 네 모드 모두 경계쌍 폭의 상한으로 사용합니다. 자동값은 r=18이면 18px. 키우면 넓은 street뿐 아니라 넓은 내부 패턴도 후보가 됩니다. 별도 범위 검증은 없고 음수나 1~2는 유효한 쌍을 만들지 못합니다. |
| `refine_corner_patch_ratio` | `0.22`, 비율 | `cross_lines` 이외에서는 `0.05 ≤ value ≤ 0.33`, 아니면 `ValueError`. | `auto`·`corner_color`의 ROI 모서리 대표색 영역 크기입니다. 키우면 대표색 계산 픽셀이 늘지만 street/무늬가 섞일 수 있습니다. 줄이면 국소 색에 민감합니다. `gradient`의 반환 좌표에는 영향이 없고, `cross_lines`에서는 계산·범위 검증 모두 생략합니다. |
| `refine_corner_reference_weight` | `0.70`, 비율 | `cross_lines` 이외에서는 `0 ≤ value ≤ 1`, 아니면 `ValueError`. | `auto`에서 색 후보에 주는 기본 가중치입니다. 1에 가까울수록 색 후보, 0에 가까울수록 경계 후보 쪽입니다. 두 후보의 유효성·1px 이내 일치 검사는 0/1에서도 유지됩니다. 다른 모드의 반환 좌표에는 영향이 없습니다. `cross_lines`에서는 범위 검증도 생략합니다. |
| `refine_noise_kernel` | `5`, px | `int` 변환 → 최소 1 → 짝수는 다음 홀수 → ROI에 들어가는 최대 홀수로 제한. | `auto`·`corner_color`·`cross_lines`의 BGR ROI median filter 크기입니다. 1이면 이 median 처리를 생략하고, 4는 5로 처리됩니다. 크게 하면 점 노이즈를 줄일 수 있지만 얇은 street/경계를 지울 수 있습니다. `gradient` 결과에는 영향이 없습니다. |
| `refine_min_confidence` | `0.15`, 내부 점수 | `0 ≤ value ≤ 1`, 아니면 `ValueError`. `refine=False`일 때도 검사합니다. | 모든 모드의 마지막 채택 기준입니다. `candidate_confidence >= value`인 후보를 사용하고, 미만이면 원래 모델 점을 사용합니다. 키우면 보수적, 줄이면 낮은 점수 후보도 허용합니다. 앞단에서 거부한 후보를 되살리지는 못합니다. |

`refine=False`이면 점별 보정 함수를 호출하지 않으므로 `refine_mode`와 대부분의 보정 전용 인자는 사용하지 않습니다. 다만 위와 같이 estimator의 `refine_min_confidence` 검사는 먼저 실행됩니다. 잘못된 타입이나 `NaN`/무한대를 반경·kernel 등에 넣는 동작은 지원 계약이 아니며, 내부 `int()` 변환 등의 예외가 발생할 수 있습니다.

`gradient`는 현재 구현상 색 후보 함수도 계산한 다음 경계 결과를 반환합니다. 따라서 corner/noise 값이 **결과 좌표에 영향을 주지 않는 것**과 해당 연산을 아예 실행하지 않는 것은 다릅니다. `cross_lines`는 이 색 후보 경로 전에 분기하여 두 corner 인자를 완전히 사용하지 않습니다.

| 단일 점 `refine_cross_point()` 이름 | 상위 API 이름 |
|---|---|
| `search_radius` | `refine_radius` |
| `mode` | `refine_mode` |
| `max_street_width` | `refine_max_street_width` |
| `corner_patch_ratio` | `refine_corner_patch_ratio` |
| `corner_reference_weight` | `refine_corner_reference_weight` |
| `noise_kernel` | `refine_noise_kernel` |

단일 점 함수는 `(candidate_point, confidence)`를 반환합니다. 여기에 `refine` 또는 `refine_min_confidence` 인자는 없으며, 마지막 confidence 채택 여부는 상위 API가 결정합니다.

## 3. 모드별 실제 차이

| 모드 | 후보 생성 | 이동을 허용하는 핵심 조건 | 반환 후보 |
|---|---|---|---|
| `gradient` | Lab 경계 에너지의 세로·가로 projection에서 두 경계 선택 | X/Y 모두 유효하며 가장 강한 경계쌍이 모델 점을 포함. 경쟁 패턴·ROI 끝 경계 등은 거부 | 두 경계 중간점 |
| `corner_color` | 네 모서리 대표색과 다른 띠의 중심 + 독립 경계 후보 | 색·경계 후보 모두 유효하고 두 2D 좌표의 거리가 1px 이하 | 색 후보 중심 |
| `auto` | `corner_color`와 같은 두 후보 | 위 조건을 통과한 뒤 confidence와 설정 가중치로 결합 | 색·경계 후보의 가중 평균 |
| `cross_lines` | 여러 작은 회전각에서 상하/좌우 팔의 경계쌍과 연속성 검사 | 각 방향의 양쪽 경계·양쪽 팔에 필요한 지지가 있고 내부 색/질감이 일관되며 강한 경쟁 교차점이 없음 | 선택한 교차점 후보를 원래 ROI 좌표로 역변환 |

`auto`의 색·경계 가중치는 각각 `corner_reference_weight × colour_confidence`, `(1 - corner_reference_weight) × gradient_confidence`입니다. `0.70`은 색 좌표를 항상 70% 반영한다는 뜻이 아닙니다. 각 후보 confidence가 다르면 실제 비율도 달라집니다.

색 모드의 corner patch 한 변은 `round(ROI변 × corner_patch_ratio)`를 계산한 뒤 최소 3px, 최대 `ROI변 // 3`으로 제한합니다. 기본 r=18, 비율 0.22이면 한 변은 8px입니다. 소스 영상 median kernel은 설정값을 따르지만, 색 거리 응답의 추가 median kernel은 최대 5입니다. 따라서 5→7 변경은 소스 영상 필터를 바꾸며 응답 필터를 7×7로 만들지는 않습니다.

## 4. `cross_lines` 내부의 고정 제한

아래 값은 현재 구현에 고정되어 있으며 여덟 파라미터로 직접 바꿀 수 없습니다. `refine_min_confidence`를 낮춰도 이 검사를 우회하지 않습니다.

1. **ROI와 원신호**: 반경 r의 전체 정사각형 ROI가 이미지 안에 있어야 합니다. ROI가 잘리거나, 원래 Lab 영상의 채널별 `95백분위 - 5백분위` 중 최댓값이 6 미만이면 모델 점·confidence 0을 반환합니다. 이 조건은 모든 모드 공통입니다.
2. **국소 각도 탐색**: ROI를 −12°부터 +12°까지 2° 간격, 총 13개 각도로 검사합니다. 각도 보정 후보를 원래 ROI로 되돌리므로 출력은 여전히 원본 clip 좌표입니다. 이는 전체 격자의 `max_rotation_deg` 또는 `angle_mode`와 다른 단계입니다. ±12° 밖에서의 복구를 보장하지 않습니다.
3. **노이즈 처리**: 설정한 median filter 뒤에 고정 3×3 Gaussian, σ=0.65를 적용합니다. `noise_kernel=1`이어도 이 Gaussian은 남습니다.
4. **팔 영역**: 각 축에서 ROI 중심으로부터 `ceil(0.55r)`~`floor(0.88r)` 떨어진 상하/좌우 구간을 봅니다. r=18이면 중심에서 10~15px 구간입니다. 두 끝 차가 3 미만이면 후보가 없습니다. 이 때문에 r=6~11은 실패합니다.
5. **경계 후보**: 각 축의 경계 peak는 최대 16개를 사용합니다. 폭은 3px 이상, 설정된 최대 폭 이하이고, 후보 중심은 회전된 seed에서 축별 `max(4, 0.35r)` 이내여야 합니다. r=18이면 각 축 6.3px입니다. 반경 18이 곧 18px 이동 허용이라는 뜻은 아닙니다.
6. **공간 지지**: 팔 두 개 × 경계 두 개의 네 조합을 검사합니다. 각 경계는 적어도 한 팔에서, 각 팔은 적어도 한 경계에서 보여야 합니다. 지지 판정에는 구간의 65% 이상에서 충분한 경계가 이어지는 조건을 씁니다. 네 조합 모두가 보일 필요는 없습니다.
7. **대비와 일관성**: 경계 강도 하한은 `max(3, 결합 profile의 95백분위 × 0.16)`입니다. 두 팔의 내부색 차이와 내부 질감은 경계 강도에 따라 6~12 Lab 거리 단위 이내여야 합니다. 가로·세로 street 내부 대표색 차이도 12 이하여야 합니다. 특정 RGB를 정답으로 쓰지는 않습니다.
8. **경쟁 후보**: 모델 점을 포함하면서 선택 후보보다 최소 2px 가까운 대안의 점수가 선택 후보의 50% 이상이면 해당 가설을 거부합니다. 최종 후보와 2px 이상 다른 교차점이 최상위 점수의 80% 이상이면 전체 보정을 거부합니다. 한 각도에서 X/Y 상위 8개씩의 조합을 검사합니다.

`refine_max_street_width`를 키워도 `cross_lines`의 이동 제한 `max(4, 0.35r)`은 늘어나지 않습니다. 반면 기존 `gradient`/색 모드의 경계 후보는 최대 폭에 따라 `max(4, 0.65 × 최대폭)`의 중심 이동 제한과 거리 가중치가 함께 바뀝니다. 같은 파라미터 변경의 효과가 모드마다 다릅니다.

이 모드는 경계가 완전히 사라진 곳을 복원하는 기법이 아닙니다. 흰색 die와 흰색 street 때문에 한 경계가 한 팔에서 사라지는 경우는 반대쪽 지지로 보완할 수 있지만, **모든 필요한 위치에 실제 명암·색 경계가 없다면 파라미터로 근거를 만들 수 없습니다.** 이때 모델 중심을 유지하는 것이 정상 동작입니다.

## 5. confidence와 실제 이동을 구분하기

보정 confidence는 YOLO 검출 confidence와 별개인 내부 점수입니다. `0.8`을 실제 정답 확률 80%로 해석하면 안 됩니다. 검출 결과를 거르는 `confidence_threshold`와 보정 결과를 거르는 `refine_min_confidence`는 서로 다른 단계입니다.

`cross_lines`의 confidence는 다음 식입니다.

```text
support_x/y = 각 축의 4개 팔-경계 조합 중 유효한 조합 비율
contrast_x/y = 각 축 양쪽 경계 강도의 기하평균
confidence = min(support_x, support_y)
             × min(1, min(contrast_x, contrast_y) / 12)
```

후보의 순위를 정할 때는 질감·팔 간 색상 차이도 반영하지만, 최종 confidence는 위의 지지율·경계 강도로 계산합니다. 모드 간 숫자가 같은 신뢰도를 뜻한다고 보장하지 않습니다.

`refine_min_confidence=0`이면 코드의 `>=` 비교상 confidence 0도 채택 분기를 통과합니다. 그러나 내부 안전 검사에서 거부된 경우 함수의 후보 좌표가 **이미 원래 모델 점**이므로 이동하지 않습니다. 그러므로 `confidence >= threshold`만으로 “보정 성공”이라고 표시하면 threshold 0에서 잘못된 성공 표시가 생길 수 있습니다. 실제 이동량과 `confidence > 0`을 함께 확인하십시오. confidence가 양수여도 후보가 이미 모델 점과 같으면 좌표가 움직이지 않을 수 있습니다.

| 진단 필드 | 정확한 의미 |
|---|---|
| `dm.grid_estimate.raw_points_clip` | 포맷 변환·검출 필터·중복 제거 이후, 영상 보정 이전의 모델 중심 |
| `dm.grid_estimate.points_clip` | confidence gate 이후 실제 격자 계산에 사용한 중심. 원래 점을 유지한 항목도 포함 |
| `dm.grid_estimate.refinement_confidences` | 각 후보의 영상 보정 점수. 하한 미달로 좌표가 원점으로 돌아가도 후보 점수 자체는 보존 |
| `dm.grid_estimate.refinement_mode` | 요청 모드. `refine=False`이면 `"none"` |
| `dm.grid_estimate.refined` | 보정 옵션을 켰는지 여부. 모든 점의 성공이나 이동 여부가 아님 |

별도의 `dm.refine.confidences`나 `grid.refine.confidences` 필드는 없습니다. 각 목록은 동일한 순서로 대응하므로 다음처럼 진단합니다. `min_confidence_used`에는 **실제 호출에 사용한 값**을 적습니다. 객체에 설정 하한이 자동 저장되지는 않습니다.

```python
import math

min_confidence_used = 0.15
grid = dm.grid_estimate
print("mode:", grid.refinement_mode, "enabled:", grid.refined)

for i, (raw, used, confidence) in enumerate(zip(
    grid.raw_points_clip,
    grid.points_clip,
    grid.refinement_confidences,
)):
    dx, dy = used[0] - raw[0], used[1] - raw[1]
    distance = math.hypot(dx, dy)
    if not grid.refined:
        state = "보정 꺼짐"
    elif confidence <= 0:
        state = "유효한 영상 후보 없음: 모델 점 유지"
    elif confidence < min_confidence_used:
        state = "후보 점수 미달: 모델 점 유지"
    elif distance <= 1e-6:
        state = "유효한 후보가 모델 점과 같음"
    else:
        state = "후보 채택: 좌표 이동"
    print(i, "raw=", raw, "used=", used,
          "delta=", (dx, dy), "distance_px=", distance,
          "confidence=", confidence, "state=", state)
```

원본/보정 점을 겹쳐 보려면 `make_clip_overlay(center_clip_bgr, grid)`를 사용합니다. 입력은 검출 모델에 넣었던 동일한 clip이어야 합니다. 점수 0의 세부 거부 이유는 현재 결과 객체에 별도 코드로 저장되지 않으므로, 점수만 보고 ROI 잘림·낮은 대비·경쟁 후보 중 어느 원인인지 확정할 수는 없습니다.

## 6. 값을 바꿀 때 확인할 것

| 관찰 | 확인할 항목 | 변경의 한계 |
|---|---|---|
| 모든 점이 그대로이고 confidence가 0 | ROI가 이미지 안에 있는지, 실제 대비가 있는지, cross_lines의 r≥12인지, 네 팔에 필요한 경계가 있는지 | 최소 confidence를 낮춰도 앞단 거부를 해결하지 못함 |
| 실제 교차점이 현재 ROI/이동 범위 밖 | clip 좌표·bbox 형식을 먼저 확인하고 반경을 한 단계씩 변경 | 큰 반경은 경쟁 패턴 및 clip 경계 접촉을 늘리고 계산량을 증가시킬 수 있음 |
| 폭이 넓은 street가 후보가 되지 않음 | 실제 경계 간 폭과 `refine_max_street_width` 비교 | 폭을 늘리면 die 내부 띠도 후보가 되므로 이동 좌표와 경쟁 여부를 함께 확인 |
| 점 노이즈가 심함 | 동일 이미지에서 kernel 1/3/5/7 결과를 비교 | 큰 kernel은 얇은 street 자체를 없앨 수 있음. 이는 비교 예시이지 검증된 장비 권장값이 아님 |
| 네 corner가 흰 street와 섞여 기존 색 모드가 거부됨 | 경계·팔 지지가 보이는지 확인한 뒤 `cross_lines`와 비교 | 모든 경계가 같은 흰색으로 사라지면 새 모드도 복원 불가 |
| 점수가 낮지만 후보 방향은 맞아 보임 | ground truth 또는 수동 확인 좌표와 오차를 비교하며 하한을 조정 | 낮은 하한이 정확도를 올린다고 보장하지 않음 |
| 보정 뒤 pitch/angle이 달라짐 | 모든 점의 delta, 선택된 center/side/below 점, angle pair를 비교 | 보정한 점이 이후 중심 선택·pitch·angle에 모두 영향을 줄 수 있음 |

한 번에 한 파라미터만 바꾸고 동일한 이미지·검출 좌표로 `refine=False` 결과와 비교하면 원인을 좁히기 쉽습니다. 원본 좌표가 모두 같은 방향으로 반 bbox 크기만큼 어긋났다면 보정보다 먼저 `xywh`가 중심형 `[cx,cy,w,h]`인지와 clip 원점 합산을 확인합니다.

실제 실패 이미지와 장비·모델 출력에 대한 성공률은 이 문서에서 주장하지 않습니다. 합성 테스트가 통과하더라도 실제 영상의 반사, 결함, 격자 회전, 패턴과 동일한 조건이라는 뜻은 아닙니다. 이전 보정 안전성 변경과 처리 흐름은 [REFINEMENT_SAFETY_KO.md](REFINEMENT_SAFETY_KO.md)를 함께 참고하십시오.
