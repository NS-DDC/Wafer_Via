# via_claude

PAD 안의 VIA 를 **순수 OpenCV 만으로** 찾아내고 양·불량 코드를 돌려주는 모듈입니다.
딥러닝 · 학습 데이터 · 외부 모델 없이 동작합니다.

- Python 3.9 이상
- 필요한 것은 `opencv-python`, `numpy` 두 개뿐입니다

## 파일

| 파일 | 하는 일 |
|---|---|
| [`via_checker.py`](via_checker.py) | 원본 사진에서 **VIA 를 직접 찾아** 판정합니다 (입력 4장) |
| [`via_code.py`](via_code.py) | **이미 이진화된 VIA 이미지**를 받아 판정만 합니다 (입력 3장) |
| [`DETECTION.md`](DETECTION.md) | 어떤 원리로 VIA 를 찾는지, 어느 값을 만지면 되는지 |

두 모듈 모두 **파일 하나만 복사해 가면 바로 쓸 수 있습니다.**
서로를 import 하지 않고, 설정 객체도 없고, 디스크에 아무것도 쓰지 않습니다.

## 반환 코드

| 코드 | 뜻 |
|---|---|
| `"1"` | 양품 |
| `"42"` | VIA 없음 |
| `"-1"` | 입력 오류 (파일 없음 · 크기 불일치 등) |

이미지 안의 PAD 가 하나라도 `42` 면 이미지 전체 코드가 `42` 입니다.
쏠림(편심) 코드는 과검이 너무 많아 **판정에서 뺐습니다.** 이유는 [`DETECTION.md`](DETECTION.md) 5장에 있습니다.

## 쓰는 법

```python
from via_checker import check_via

code, result, via_bin = check_via("원본.jpg", "이진화.png",
                                  "PAD설계도.png", "VIA설계도.png")
# code    : "1" / "42" / "-1"
# result  : 검출 결과를 그려 넣은 BGR 이미지
# via_bin : 찾아낸 VIA 만 남긴 이진화 이미지
```

```python
from via_code import check_via_code

code, result = check_via_code("내가_이진화한_VIA.png",
                              "PAD설계도.png", "VIA설계도.png")
```

너무 많이 잡거나 너무 못 잡으면 코드를 고치지 않고 호출할 때 한 번만 밀 수 있습니다.

```python
check_via(..., dark_offset=-15)   # 음수 = 덜 잡는다 / 양수 = 더 잡는다
```

두 모듈 다 `verbose=True` 를 주면 PAD 마다 왜 그렇게 판정했는지
(면적 · 중심거리 · 채움비 · 어느 필터에서 몇 개가 탈락했는지) 표로 찍어 줍니다.

## 검출 원리 요약

한 PAD 안에서 아래 네 조건을 **모두** 만족하는 화소만 VIA 후보로 봅니다.

| | 조건 | 왜 |
|---|---|---|
| 1 | PAD 대표색과 색이 다르다 | VIA 는 PAD 표면과 다른 층이다 |
| 2 | PAD 대표 밝기보다 어둡다 | VIA 는 항상 검정 또는 진한 갈색이다 |
| 3 | 검정 또는 갈색 계열이다 (`B ≤ R`, `G ≤ R`) | 파랗거나 밝은 이물을 뺀다 |
| 4 | 주변에서 홀로 우묵하다 (Black-hat) | 넓게 깔린 그늘·조명 얼룩을 뺀다 |

그 다음 덩어리 단위로 **면적 · PAD 중심에서의 거리 · 경계 여유 · 채움비 · 모양**을 봅니다.
특히 중심 거리 조건이 "PAD 외곽의 검은 선을 VIA 로 오인" 하는 문제를 없앴습니다.

자세한 근거와 실측 수치는 [`DETECTION.md`](DETECTION.md) 를 보세요.

## 검출 성능 (실측)

| | 테스트셋 PAD 판정 | VIA 가 없는 실물에서 VIA 를 찾아낸 수 |
|---|---|---|
| 개선 전 | 772 / 783 | 191 / 777 |
| 개선 후 | 772 / 783 | **2 / 777** |

측정 방법과 임계값 근거는 [`DETECTION.md`](DETECTION.md) 3장 · 6장에 있습니다.
