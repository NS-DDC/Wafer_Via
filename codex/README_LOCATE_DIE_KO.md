# locate_die 입력과 반환

`wafer_via_notch_standalone.py` 전체를 복사하면 됩니다. `locate_die()`만 복사하면
내부 helper가 빠집니다. adaptive 복붙 파일과 유지보수용 모듈에도 같은 기능이 적용됩니다.

```python
from wafer_via_notch_standalone import locate_die

# 기존 동작: point가 속한 die 또는 bbox 중심이 속한 die 하나(dict)
hit = locate_die(dm, point=(x, y))
hit = locate_die(dm, bbox=(x1, y1, x2, y2), return_all=False)

# 새 동작: 입력 영역에 해당하는 모든 die(list[dict])
hits = locate_die(dm, bbox=(x1, y1, x2, y2), return_all=True)

# 점도 동일 플래그 사용: 내부 1개, 공유 경계 2개, 공유 모서리 최대 4개
hits = locate_die(dm, point=(x, y), return_all=True)
```

| 조건 | 기본 `return_all=False` | `return_all=True` |
|---|---|---|
| die 내부의 점 | dict 1개 | dict 1개를 담은 list |
| 공유 경계 또는 모서리의 점 | 기존 floor 규칙으로 1개 | 경계를 공유하는 모든 저장된 die |
| 여러 die에 걸친 bbox | bbox 중심점의 die 1개 | bbox와 면적으로 겹치는 모든 저장된 die |
| 대응하는 DM entry가 없음 | 기존 격자 추정 dict 유지 | 빈 list `[]` |

- `point`와 `bbox` 중 하나만 전달합니다. 여러 점 목록 입력은 추가하지 않았습니다.
- 입력 좌표는 DM의 이미지 좌표계입니다. notch 정렬 DM은 `dm.aligned_image` 좌표계입니다.
- bbox는 `(x1, y1, x2, y2)`이며 새 모드에서는 `x1 < x2`, `y1 < y2`여야 합니다.
- bbox와 선/모서리만 닿는 die는 제외합니다. point 경계 판정에는 1e-7 px 수치 허용오차를 씁니다.
- 회전된 격자도 실제 die 사각형과 bbox의 겹침을 판정합니다. 축 정렬 bounding box만 비교하지 않습니다.
- 결과 순서는 `(iy, ix)` 오름차순입니다. 이미지상 위→아래 순서와 다릅니다.
- 각 항목의 키는 기존 단일 반환과 같습니다. `die_index`, `die_rect_px`, `crop_rect_px` 등을 그대로 사용합니다.
- `query_px`는 모든 항목에서 입력 point 또는 bbox 중심입니다. `in_wafer`, `real_coord`도 이 조회점을 기준으로 합니다.
- 매칭 기준은 저장된 die의 전체 격자 사각형입니다. wafer 밖으로 잘린 부분도 해당 셀에 속합니다.
  실제 crop 범위는 `crop_rect_px`이며, 영상 밖의 셀은 빈 crop이 될 수 있습니다.
- `include_edge=False` 등으로 DM에서 제외한 die는 새 모드가 만들어서 반환하지 않습니다.
- 기존 단일 모드의 index 공식, 경계 처리, crop 사각형, 반환 키는 유지했습니다.

## 보정 이미지에서 잘라 쓰기

```python
hits = locate_die(dm, bbox=(x1, y1, x2, y2), return_all=True)
for hit in hits:
    left, top, right, bottom = hit["crop_rect_px"]
    if right <= left or bottom <= top:
        continue
    die_image = dm.aligned_image[top:bottom, left:right].copy()
    die_index = hit["die_index"]
```

`crop_rect_px`는 `(left, top, right, bottom)`입니다. 오른쪽/아래쪽은 NumPy slice의
끝값(exclusive)입니다. `die_center_px`에서 정수 폭을 빼서 crop 위치를 다시 만들 필요가 없습니다.

원본 이미지의 점을 조회할 때는 보정 좌표로 한 번 변환합니다.

```python
from wafer_via_notch_standalone import transform_point_to_aligned

aligned_point = transform_point_to_aligned(dm, (original_x, original_y))
hits = locate_die(dm, point=aligned_point, return_all=True)
```

bbox를 다른 좌표계로 옮길 때는 네 모서리를 모두 변환해야 합니다. 회전 후 그 점들을 감싼
축 정렬 bbox는 원래 회전 사각형보다 넓을 수 있으므로 추가 die가 포함될 수 있습니다.

shift 현상은 [원인 분석](LOCATE_DIE_SHIFT_ANALYSIS_KO.md)에 정리했습니다.
