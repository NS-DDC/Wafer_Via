# locate_die crop shift 원인 분석

2026-09-16 현재 코드와 합성 좌표로 검증했습니다. 실제 문제가 난 point/bbox,
DM 인스턴스, 이미지 slicing 호출부가 제공되지 않아 장비에서 발생한 원인은 확정하지 않았습니다.
shift 관련 기존 동작은 수정하지 않았습니다. 기능 변경은 선택적인 `return_all` 모드입니다.

## 확인한 사실

`generate_die_map()`이 die polygon과 `crop_rect_px`를 저장합니다. `locate_die()`는
계산한 index의 DM entry가 있으면 저장된 `rect_px`, `crop_rect_px`를 그대로 반환합니다.
이미지를 직접 자르는 기능은 없습니다.

10000×10000 좌표계, 중심 `(5000, 5000)`, 반지름 4700, 기준 corner `(5013.17, 4992.31)`,
pitch `(90.2, 92.15)`로 다음을 확인했습니다.

| 격자 각도 | 조회한 die 중심 수 | index 불일치 | die rect 불일치 | crop rect 불일치 |
|---|---:|---:|---:|---:|
| 0° | 8,551 | 0 | 0 | 0 |
| 3.25° | 8,548 | 0 | 0 | 0 |

기존 버전의 `locate_die()`와 같은 DM/입력으로 비교한 17,099개 중심 조회 및
무작위 bbox 600개에서 기본 반환 dictionary의 모든 값이 일치했습니다.
저장된 crop 사각형으로 x좌표 값을 픽셀에 넣은 영상도 잘라 확인했으며, 최초 x=5193,
마지막 x=5283으로 `(5193, 5176, 5284, 5269)` 사각형과 일치했습니다.

## 재현한 원인 후보

### 1. 원본 이미지 좌표와 보정 이미지 좌표 혼용

notch DM은 `coordinate_space="aligned_image"`입니다. `grid_angle_deg=0`이어도
원본 이미지에서 회전이 없었다는 뜻은 아닙니다. 원본에 적용한 회전은 `image_rotation_deg`입니다.

보정각 0.2°에서 원본점 `(5185.6593, 8698.9806)`은 보정 후
`(5198.5700, 8698.3100)`으로 이동합니다. x 차이는 **12.91px**입니다.

- 보정한 점 조회: `(2, -41)`
- 원본점을 보정 DM에 그대로 조회: `(1, -41)`

올바른 die를 조회해도 그 사각형으로 원본 wafer를 잘라 쓰면 패턴 위치가 어긋날 수 있습니다.
입력 좌표, DM, crop 대상 이미지가 같은 좌표계여야 합니다.

### 2. 격자 경계의 부동소수점과 표시 좌표 반올림

기존 단일 모드는 `floor((x-x0)/pitch_x)`를 사용합니다.
`x0=5013.17`, `pitch_x=90.2`, 경계 x=5103.37이면 몫이
`0.999999999999998`이 되어 오른쪽 셀 index 1 대신 왼쪽 index 0을 선택하는 것을 재현했습니다.

또한 overlay에 표시하려고 경계 x=5103.37을 5103으로 반올림하고 그 정수점을 다시 조회하면
실제로 왼쪽 셀 안에 들어갑니다. 이것은 표시 좌표와 계산 좌표의 차이입니다.

경계에서 어느 쪽 셀을 선택할지는 규칙이 필요하며, 이번 작업에서는 기존 단일 선택 규칙을 유지했습니다.
새 다중 모드는 공유 경계의 양쪽을 반환합니다. 이 변경이 실제 crop shift의 원인을 해결했다는 의미는 아닙니다.

### 3. crop 호출부에서 정수 die 폭을 반복 간격으로 사용

`pitch_x=90.2`일 때 `die_w=round(pitch_x)=90`입니다. 호출부가 40칸 이동을
`40 * die_w`로 계산하면 실제 경계보다 **8px 왼쪽**으로 이동합니다.
현재 `locate_die()` 자체는 소수 pitch와 저장된 사각형을 사용하므로 이 재계산은 하지 않습니다.

같은 크기의 이미지/DM을 쓰고, 중심 조회가 정확한데 crop만 서서히 옆으로 밀린다면
호출부에서 `die_w`, `int(pitch_x)` 또는 중심 좌표로 crop을 재구성하는지 확인해야 합니다.

### 4. bbox의 기존 의미

기존 bbox 조회는 겹친 die를 모두 검색하지 않고 bbox의 중심점 한 곳으로 선택합니다.
bbox가 옆 die까지 걸치면 사용자가 예상한 첫 번째 die와 다른 셀을 반환할 수 있습니다.
이 경우 새 `return_all=True`로 해당 영역에 걸친 모든 셀을 조회할 수 있습니다.

## 실제 실패 사례에서 비교할 값

입력 `point` 또는 `bbox`, `dm.coordinate_space`, `dm.image_rotation_deg`,
`dm.x0/y0`, `pitch_x/y`, 반환 `die_index/crop_rect_px`, 그리고
실제 사용하는 `image[top:bottom, left:right]` 구문을 함께 비교하면 위 원인들을 구분할 수 있습니다.
저장된 DM이 오래됐거나 overlay와 crop에 서로 다른 DM을 사용했는지도 확인 대상입니다.

재현 도구: `tools/analyze_locate_die_shift.py`

```powershell
py -3.13 tools/analyze_locate_die_shift.py --output locate_die_shift_evidence.json
```

`--baseline 이전_wafer_via.py`를 추가하면 이전 단일 반환과 직접 비교합니다.
첨부 [JSON](locate_die_shift_evidence.json)은 변경 전 파일을 baseline으로 비교한 실제 실행 결과입니다.
