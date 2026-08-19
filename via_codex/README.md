# 중앙 검정·짙은 갈색 VIA 검출 가이드

이 문서는 `via_checker.py`의 Codex 변경판만 설명합니다. 기존
`pad_via_inspector.py`, `via_code.py`, `GUIDE.md`, `ALGORITHM.md`는 원본 파이프라인의
쏠림 판정을 그대로 유지합니다.

## 바뀐 판정

| 상황 | `via_checker.py` 판정 | 코드 |
|---|---|---|
| PAD 중앙에서 검정/짙은 갈색 VIA 검출 | `OK` | `"1"` |
| 검출한 VIA가 중앙 검색영역 안에서 조금 치우침 | `OK` | `"1"` |
| 중앙 검색영역에 VIA 없음 | `VIA_MISSING` | `"42"` |
| PAD 외곽에만 검은 선 또는 테두리 존재 | `VIA_MISSING` | `"42"` |
| 입력 파일 오류 또는 크기 불일치 | `ERROR` | `"-1"` |

`CODE_VIA_OFFSET = "99"`는 기존 호출부가 import할 때 깨지지 않도록 남겨 두었습니다.
기본값 `VIA_OFFSET_DEFECT_ENABLED = False`에서는 반환되지 않습니다.

## VIA를 찾는 원리

검사는 VIA 설계도에 점이 있는 PAD만 대상으로 합니다.

1. PAD 설계 형상을 실측 이진 PAD의 무게중심 쪽으로 제한적으로 이동합니다.
   이 정합은 카메라와 설계도의 몇 픽셀 오차 때문에 중앙 검색원이 빗나가는 것을
   줄입니다.
2. 정합된 PAD 중심을 기준으로 원형 검색영역을 만듭니다. 기본 반지름은
   `PAD 등가반지름 × 0.50`입니다. 이 원 밖의 픽셀은 색을 보기 전에 버립니다.
3. ROI를 HSV로 바꿉니다. 검정은 V(밝기)가 낮은지, 갈색은 H(색상)가 갈색 범위인지,
   S(채도)가 충분한지, V가 충분히 낮은지를 함께 봅니다.
4. PAD 중앙을 제외한 영역에서 PAD 바탕의 V 중앙값을 구합니다. 후보는 이 값보다
   최소 `VIA_MIN_VALUE_DROP`만큼 어두워야 합니다. PAD마다 바탕 밝기가 달라도 같은
   상대 기준이 적용됩니다.
5. Black-hat으로 주변보다 국소적으로 어두운 곳만 남깁니다. 밝은 이물을 검출하던
   Top-hat은 제거했습니다.
6. 남은 화소를 연결성분으로 묶고 면적, PAD 경계 여유, 최소외접원 채움비,
   장단비, 반경편차를 검사합니다. 선과 비원형 후보는 완전히 탈락시키며 다시
   되살리지 않습니다.
7. 조건을 모두 통과한 가장 큰 연결성분을 VIA로 선택합니다. 중심 거리는
   `debug_via()`에 진단값으로 남지만 기본 판정에는 쓰지 않습니다.

이 순서 때문에 PAD 외곽의 선은 검정이어도 2단계에서 먼저 제거됩니다. 외곽 선의
일부가 중앙 검색영역까지 들어오더라도 길쭉한 형상은 6단계에서 다시 제거됩니다.

## 코드에서 수정할 위치

`via_checker.py`에서 다음 문자열을 검색하세요.

- `[SECTOR: VIA_DETECTION_CONFIG]`: 숫자 임계값을 조절하는 곳
- `[SECTOR: VIA_DETECTION_CORE]`: 실제 후보 마스크와 연결성분을 만드는 곳

주로 수정할 값은 다음과 같습니다.

| 상수 | 기본값 | 의미 | 조절 방향 |
|---|---:|---|---|
| `VIA_CENTER_SEARCH_RATIO` | `0.50` | PAD 반지름 대비 중앙 검색 반지름 | 외곽 선 과검출 시 감소, 정상 VIA가 잘리면 증가 |
| `VIA_BLACK_VALUE_MAX` | `85` | 검정으로 허용하는 HSV V 상한 | 흐린 검정 미검출 시 증가 |
| `VIA_BROWN_HUE_MIN` | `3` | 갈색 HSV H 하한 | 실제 갈색 샘플 분포에 맞춰 조절 |
| `VIA_BROWN_HUE_MAX` | `28` | 갈색 HSV H 상한 | 노랑 계열 과검출 시 감소 |
| `VIA_BROWN_SAT_MIN` | `55` | 갈색 최소 채도 | 회색 얼룩 과검출 시 증가 |
| `VIA_BROWN_VALUE_MAX` | `165` | 짙은 갈색 V 상한 | 밝은 갈색 과검출 시 감소 |
| `VIA_MIN_VALUE_DROP` | `15` | PAD 바탕보다 어두워야 하는 최소 V 차이 | 얼룩 과검출 시 증가 |
| `VIA_MAX_ASPECT` | `2.2` | 연결성분 장단비 상한 | 검은 선 과검출 시 감소 |
| `VIA_MAX_RADIAL_DEV` | `0.45` | 원형에서 벗어난 정도의 상한 | 부스러기 과검출 시 감소 |
| `VIA_MIN_FILL` | `0.45` | 최소외접원 채움비 하한 | 링/호 과검출 시 증가 |

한 번에 여러 값을 바꾸지 말고, 실제 양품/불량 샘플을 함께 두고 한 값씩 변경하는
것이 안전합니다.

## `dark_offset` 사용법

기존 함수 호출을 깨지 않기 위해 인자를 유지했습니다.

```python
code, result, via_bin = check_via(
    image, bin_mask, pad_design, via_design,
    dark_offset=-5,
)
```

- `dark_offset > 0`: 검정/갈색 V 상한을 올리고 최소 밝기 차를 줄여 더 민감하게 검출
- `dark_offset < 0`: V 상한을 내리고 최소 밝기 차를 늘려 더 엄격하게 검출

기본값 0에서 5~10 단위로 움직이는 것을 권장합니다.

## 디버깅

```python
from via_checker import debug_via

code, result, via_bin, rows = debug_via(
    image, bin_mask, pad_design, via_design,
    quiet=False,
)

for row in rows:
    print(
        row["pad_id"],
        row["status"],
        row["search_radius"],
        row["pad_value_median"],
        row["black_value_max"],
        row["brown_value_max"],
        row["color_candidate_pixels"],
        row["shape_rejected"],
        row["edge_rejected"],
    )
```

- `color_candidate_pixels == 0`: 중앙에 색 조건을 통과한 픽셀이 없음
- `shape_rejected > 0`: 검은 선이나 비원형 후보가 모양 조건에서 탈락
- `edge_rejected > 0`: 경계 여유 또는 채움비 조건에서 탈락
- `offset_norm`: 검출 중심 거리 진단값이며 기본 판정에는 영향 없음

## 회귀 테스트

```bash
python -m unittest -v test_via_checker.py
```

테스트는 중앙 검정, 중앙 짙은 갈색, 쏠림 비활성, 외곽 검은 선 무시, 밝은 색 이물
무시를 확인합니다.
