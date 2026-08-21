# PAD 평균 밝기 기반 중앙 VIA 검출 가이드

이 문서는 `via_checker.py`의 단순화된 Codex 버전을 설명합니다. 기존
`pad_via_inspector.py`와 `via_code.py`는 변경하지 않았습니다.

## 판정

| 상황 | 판정 | 코드 |
|---|---|---|
| PAD 평균보다 충분히 어두운 덩어리의 중심이 PAD 중앙에 있음 | `OK` | `"1"` |
| 어두운 덩어리가 없거나 중심에서 벗어남 | `VIA_MISSING` | `"42"` |
| 입력 오류 또는 이미지 크기 불일치 | `ERROR` | `"-1"` |

`CODE_VIA_OFFSET = "99"`는 기존 import 호환을 위해 상수만 남아 있습니다.
쏠림 판정 분기는 제거했으므로 code `"99"`는 반환되지 않습니다.

## PAD 존재 필터 제거

`PAD_PRESENT_MIN`과 `VIA_EXCLUDE_RATIO` 상수 및 `_coverage()` 판정은 제거했습니다.
따라서 VIA 설계도에 점이 있는 PAD는 `bin_mask`의 커버리지가 낮거나 VIA 구멍이
커도 `_find_via()` 검사를 반드시 수행합니다.

`bin_mask`는 설계 PAD 중심을 실측 PAD 쪽으로 조금 이동시키는 국소 정합에만
사용됩니다. `bin_mask`가 비어 있거나 정합에 필요한 픽셀이 부족하면 이동을
생략하고 설계 PAD의 원래 중심으로 계속 검사합니다.

`debug_via()`의 `row["pad_coverage"]` 키는 기존 호출부 호환을 위해 남지만 값은
항상 `None`입니다.

## `_find_via` 원리

결정 조건은 세 개뿐입니다.

1. 컬러 ROI를 회색조로 변환합니다.
2. 침식한 PAD 영역의 평균 밝기 `pad_mean`을 구합니다.
3. 아래 식으로 어두운 픽셀을 이진화합니다.

   ```text
   threshold = pad_mean - VIA_GRAY_DROP + dark_offset
   candidate = gray < threshold
   ```

4. 이진 후보를 연결성분으로 묶고 `VIA_MIN_AREA`보다 작은 카메라 잡음만 버립니다.
5. 각 연결성분의 중심과 PAD 중심 사이 거리를 구합니다. 거리가
   `PAD반지름 × VIA_CENTER_SEARCH_RATIO` 이내인 후보만 VIA로 인정합니다.
6. 여러 후보가 통과하면 PAD 중심에 가장 가까운 것을 선택합니다.

색상 범위, Black-hat, 원형도, 채움비, 최대 면적 조건은 사용하지 않습니다.
외곽 검은 선은 어두운 후보로는 나오지만 연결성분 중심이 PAD 중앙에서 멀기 때문에
VIA로 채택되지 않습니다.

## 수정할 위치

`via_checker.py`에서 다음 표식을 검색하세요.

- `[SECTOR: VIA_DETECTION_CONFIG]`: 세 개의 주요 숫자를 조절하는 곳
- `[SECTOR: VIA_DETECTION_CORE]`: 회색조 이진화와 중앙 위치 검사 본체

| 상수 | 기본값 | 의미 | 조절 방향 |
|---|---:|---|---|
| `VIA_GRAY_DROP` | `25.0` | PAD 평균보다 얼마나 어두워야 하는지 | 과검출 시 증가, 미검출 시 감소 |
| `VIA_CENTER_SEARCH_RATIO` | `0.30` | PAD 반지름 대비 후보 중심 허용거리 | 외곽 과검출 시 감소, 정상 VIA 미검출 시 증가 |
| `VIA_MIN_AREA` | `4` | 연결성분 최소 픽셀 수 | 점 잡음 과검출 시 증가 |

## `dark_offset`

기존 함수 호출 호환을 위해 유지한 실행 시 조절값입니다.

```python
code, result, via_bin = check_via(
    image, bin_mask, pad_design, via_design,
    dark_offset=5,
)
```

- 양수: 이진화 임계값이 올라가 연한 VIA까지 검출
- 음수: 이진화 임계값이 내려가 더 어두운 VIA만 검출

기본값 0에서 5~10 단위로 조절하세요.

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
        row["pad_mean"],
        row["dark_threshold"],
        row["dark_candidate_pixels"],
        row["search_radius"],
        row["offset_norm"],
    )
```

- `pad_mean`: 해당 PAD 내부 회색조 평균
- `dark_threshold`: 실제 이진화 임계값
- `dark_candidate_pixels`: 평균보다 충분히 어두운 픽셀 수
- `search_radius`: 후보 중심에 허용한 중앙 거리(px)
- `offset_norm`: 검출된 VIA 중심 거리 진단값

## 테스트

```bash
python -m unittest -v test_via_checker.py
```

중앙의 어두운 VIA 검출, 상대 밝기 임계, `dark_offset`, 중앙 위치 허용, 외곽 검은
선 제외, 빈 `bin_mask`에서도 VIA 검사가 진행되는지를 합성 영상으로 확인합니다.
