# 원본 이미지 한 장으로 VIA 이진화하기

[`via_binarize.py`](via_binarize.py)의 `binarize_via()`는 원본 이미지 한 장을 받아
VIA 후보가 흰색인 이진 마스크를 반환합니다. PAD/VIA 설계도나 매칭 좌표는
입력하지 않습니다. 함수 전체를 복사해서 사용할 수 있고, 의존성은 NumPy와
OpenCV입니다.

## 기본 사용

저장소 루트에서 실행하거나, 저장소 루트가 Python 모듈 검색 경로에 있는 경우:

```python
import cv2
from via_codex.via_binarize import binarize_via

image = cv2.imread("original.png", cv2.IMREAD_UNCHANGED)
if image is None:
    raise ValueError("원본 이미지를 읽을 수 없습니다.")

via_bin = binarize_via(
    image,
    block_size=31,
    gray_drop=25,
    min_area=4,
    max_area=200,           # 예시값: 큰 덩어리 제거. 실측 VIA 크기에 맞춰 조정
    shape_filter=True,
    max_aspect_ratio=1.8,
    min_circularity=0.65,
)
cv2.imwrite("via_binary.png", via_bin)
```

파일을 호출 스크립트와 같은 폴더에 복사했다면
`from via_binarize import binarize_via`로 가져옵니다.
기본값으로 실행하려면 `via_bin = binarize_via(image)`만 호출하면 됩니다.
이때 `max_area`는 `None`이라 최대 면적 제한이 없습니다.

## 입력과 반환

입력은 **uint8(0~255) NumPy 배열**입니다. 다음 채널 형식을 자동으로 처리합니다.

| 입력 형태 | 처리 |
|---|---|
| `(H, W)` | 회색조로 사용 |
| `(H, W, 1)` | 마지막 채널 축을 제거하고 회색조로 사용 |
| `(H, W, 3)` | OpenCV BGR → 회색조 |
| `(H, W, 4)` | OpenCV BGRA → 회색조, 알파 제외 |

PIL 등에서 가져온 RGB 배열이라면 먼저
`image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)`로 바꾸세요.
채널 수로 RGB와 BGR 순서를 구별할 수는 없습니다.
16비트나 실수 배열은 먼저 촬영 데이터 범위에 맞춰 uint8로 변환해야 합니다.

반환값은 원본과 같은 `(H, W)` 크기의 `uint8` 배열입니다.
VIA 후보는 `255`, 나머지는 `0`이며 원본 배열과 해상도를 변경하지 않습니다.

## 설정값

| 인자 | 기본값 | 의미와 조절 방향 |
|---|---:|---|
| `block_size` | `31` | 주변 평균을 구할 한 변. 3 이상의 홀수. 실제 VIA 지름의 2~3배부터 시작 |
| `gray_drop` | `25.0` | 주변 평균보다 이 값 넘게 어두운 픽셀 추출. 낮추면 연한 VIA도 포함 |
| `min_area` | `4` | 최소 연결성분 픽셀 수. 1~3픽셀 잡음 제거. `0`이면 최소 제한 해제 |
| `max_area` | `None` | 최대 연결성분 픽셀 수. `None`이면 최대 제한 해제 |
| `shape_filter` | `True` | 종횡비와 원형도 필터 적용 여부. 면적 필터와 독립 |
| `max_aspect_ratio` | `1.8` | 회전 사각형의 긴 변/짧은 변 상한. 낮추면 길쭉한 덩어리를 더 제거 |
| `min_circularity` | `0.65` | 외곽 원형도 하한. 높이면 불규칙 덩어리를 더 제거 |

### `min_area` / `max_area`

면적은 이진화 후 **8방향으로 연결된 흰 픽셀의 개수**입니다. 사각형 넓이나
반지름, 설계도 면적, `cv2.contourArea()` 값이 아닙니다.

```python
# 연결된 흰 픽셀이 4~200개인 덩어리만 면적 조건 통과 (양 끝 포함).
# 이후 종횡비/원형도 조건도 통과해야 최종 결과에 남습니다.
via_bin = binarize_via(image, min_area=4, max_area=200)

# 큰 덩어리 제한 해제. 최소 면적/형상 조건은 적용됩니다.
via_bin = binarize_via(image, max_area=None)

# 면적 조건만 적용하고 형상 필터는 끄기.
via_bin = binarize_via(image, min_area=4, max_area=200, shape_filter=False)

# 최대 면적 조건만 적용하기.
via_bin = binarize_via(image, min_area=0, max_area=200, shape_filter=False)

# 면적/형상 필터를 모두 끄고 이진화 결과 전체 확인하기.
via_bin = binarize_via(image, min_area=0, max_area=None, shape_filter=False)
```

`max_area=200`이면 200픽셀은 허용하고 201픽셀부터 덩어리 전체를 제거합니다.
`max_area`에는 `None` 또는 `min_area` 이상의 양의 정수를 넣습니다.
`max_area=0`은 끄기가 아니라 입력 오류입니다.
선과 VIA가 연결되면 하나의 덩어리로 계산되므로 너무 작은 상한은 실제 VIA도
제거할 수 있습니다. 먼저 `max_area=None`으로 결과를 확인한 뒤 상한을 정하세요.

## 설계도가 가로·세로 약 2배 큰 경우

모든 설정은 **실제 입력 영상의 픽셀 기준**입니다.

- 설계도 지름 20px → 실측 지름 약 10px → `block_size=31`부터 시도.
- 설계도 연결성분 면적 400px → 실측 면적 약 100px.
- 길이는 약 1/2, 면적은 약 1/4로 환산합니다.

환산값은 초기 추정치입니다. 이진화 경계, 촬영 배율 오차 등을 감안해
`min_area`/`max_area`에 여유를 두세요. 함수 안에서 2배 확대하거나 정합하지 않습니다.

## 검출 원리와 조절 순서

1. 입력 채널 수에 맞춰 회색조로 변환합니다.
2. 각 픽셀 주변 `block_size × block_size` 영역의 평균을 구합니다.
3. `gray < local_mean - gray_drop` 조건을 만족하는 픽셀을 흰색으로 만듭니다.
4. 연결성분별로 `min_area <= 픽셀 수 <= max_area`를 검사합니다.
   `max_area=None`이면 상한 검사는 생략합니다.
5. `shape_filter=True`이면 회전 사각형 종횡비로 대각선을 포함한 긴 선을 걸러내고,
   `4π × 윤곽면적 / 윤곽둘레²` 원형도로 불규칙한 경계의 덩어리를 걸러냅니다.
6. 통과한 덩어리의 원래 이진 픽셀을 그대로 반환합니다.

| 증상 | 먼저 조절할 값 |
|---|---|
| 어두운 VIA 자체가 이진화되지 않음 | `gray_drop`을 25 → 20 → 15로 낮추기 |
| 큰 VIA 내부가 비어 보임 | `block_size`를 키우기 |
| 작은 점 잡음이 많음 | `min_area` 올리기 |
| 큰 어두운 덩어리가 남음 | `max_area` 설정 또는 낮추기 |
| 길쭉한 선이 남음 | `max_aspect_ratio` 낮추기 |
| 들쭉날쭉한 잡음이 남음 | `min_circularity`를 0.65 → 0.75로 높이기 |
| 정상 VIA가 형상 필터에서 빠짐 | `min_circularity`를 0.55로 낮추거나 `max_aspect_ratio` 높이기 |

작은 원은 픽셀 계단 때문에 원형도가 낮아질 수 있습니다. 둥근 잡티나
VIA와 밝기·형상이 비슷한 패턴은 남을 수 있고, 경계에서 잘리거나 선과 붙은 VIA는
제외될 수 있습니다. 이 함수는 패턴 매칭에 사용할 VIA 후보 이진화를 제공합니다.

코드를 직접 수정하려면 `VIA_BINARIZATION_CORE`(밝기 이진화)와
`VIA_NOISE_FILTER`(면적/형상 필터) 주석을 검색하세요.
