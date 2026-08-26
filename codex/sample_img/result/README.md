# sample_img 3장 평가 — `measure_die_render_angle` (V5 bare 버전)

대상: `wafer_via_die_render.py` 의 `measure_die_render_angle`
(V5 `wafer_die_map_v5.py:1562` 를 그대로 옮긴 5줄짜리 함수).

웨이퍼 중심은 실제 파이프라인처럼 YOLO 검출 좌표에서 뽑지 않고
`detect_wafer_boundary` 로 잡은 **wafer center** 를 그대로 썼다.

## 입력

세 장 모두 1254×1254, pitch ≈ 38 px, **street 가 die 보다 어둡다**,
그리고 원본은 이미 거의 축정렬돼 있다 (측정각 +0.001° ~ +0.002°).

| 파일 | wafer center | r | 원본 측정각 |
|---|---|---|---|
| natural_amber_olive_bronze | (626, 623) | 613 | +0.0017° |
| natural_rose_violet_iceblue | (626, 623) | 613 | +0.0010° |
| natural_teal_bluegray | (627, 623) | 613 | +0.0013° |

## 평가 방법 — 왕복(round-trip)

정답 각도가 없으므로 **알려진 각도 θ 로 회전시켜 넣고 그걸 되찾는지** 본다.
기대값은 `원본측정각 − θ`, 오차는 `측정값 − 기대값`.

### 결과 (θ = −4° … +4°, 13 지점)

| 파일 | \|오차\| max | RMS |
|---|---|---|
| natural_amber_olive_bronze | **0.0283°** | 0.0169° |
| natural_rose_violet_iceblue | **0.0228°** | 0.0121° |
| natural_teal_bluegray | **0.0266°** | 0.0139° |

세 장 모두 **오차 0.03° 미만**. fine step 이 0.02° 이므로 사실상 분해능 한계다.
(`eval_report.txt` / `eval_report.json` 에 13 지점 전부)

### 한계 — `search_deg` 경계 (`eval_limit.txt`)

기본 `search_deg = 6.0`. 이 범위 안에서만 정상이다.

| 넣은 각 | 결과 |
|---|---|
| ±5.0, ±6.0 | OK (오차 < 0.03°) |
| ±7.0, ±8.0 | **CLAMP** — 경계값 ±6.15° 를 반환 |
| ±10, ±12 | **완전히 틀림** — 부호가 반대인 값까지 나온다 |

가장 위험한 건 `natural_amber_olive_bronze` 에 −10° 를 준 경우로,
경계값이 아니라 **−3.82°** 라는 그럴듯한 값을 냈다. 경계에 붙은 값은
의심이라도 할 수 있지만 이건 알아챌 방법이 없다.

**어떤 경우에도 `None` 이나 예외가 나오지 않는다.** 범위 밖 기울기는
조용히 틀린 각도로 돌아온다. 기울기가 6°를 넘을 수 있는 입력이라면
`search_deg` 를 호출부에서 키워야 한다.

## 산출물

| 파일 | 내용 |
|---|---|
| `flow_*.png` | **데이터 흐름** — 이미지가 6단계로 어떻게 변하는지 |
| `codeflow.png` | **코드 흐름** — 어느 함수가 어떤 순서로 불리는지 (줄번호 포함) |
| `*_aligned.png` | 측정각으로 정렬한 결과 |
| `eval_report.txt` / `.json` | 왕복 평가 전체 |
| `eval_limit.txt` / `.json` | 경계 한계 측정 |

## 알고리즘 흐름 (`flow_*.png` 의 6패널)

1. `detect_wafer_boundary` → `cx, cy, r` → 중앙 ROI `half = r × 0.55`
   **측정에 쓰이는 건 이 초록 박스 안뿐이다.**
2. crop → gray → `max_dim=1400` 으로 축소 (1254 입력은 ROI 674px 이라 축소 없음).
   **여기서 컬러가 버려진다.**
3. blur(3×3) → **Otsu 이진화** → 원 밖은 0.
   흰색 = die, 검정 = street. 임계값은 자동(이 이미지들은 78).
4. 틀린 각도로 회전 → street 가 열 방향으로 번짐 → 열합 프로파일이 밋밋 → **분산 낮음**
5. 맞은 각도로 회전 → street 가 축과 나란해짐 → 열합이 크게 진동 → **분산 최대**
   (측정 예: 0.84e+08 → 1.24e+09, 약 15배)
6. `_search_peak`: coarse 0.15° 간격으로 ±6° 훑기 → 최대점 주변을
   fine 0.02° 로 재훑기 → 3점 **포물선 보간**으로 소수점 아래까지.

점수 함수는 `var(열합) + var(행합)` 하나뿐이다. 즉
**"격자가 축에 가장 잘 정렬되는 각도"** 를 직접 찾는 방식이고,
격자 검출·템플릿 렌더링 같은 중간 단계가 없다.
