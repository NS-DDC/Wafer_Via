# Wafer_Via

## codex — wafer die-map

YOLO centre-clip cross-points를 이용하는 wafer die-map 모듈입니다.

- 구현/사용법: [`codex/README.md`](codex/README.md)
- 코드: [`codex/wafer_via.py`](codex/wafer_via.py)
- 샘플: [`codex/sample_img/Clip_sample.png`](codex/sample_img/Clip_sample.png)
- 테스트: [`tests/test_wafer_via.py`](tests/test_wafer_via.py)

## via_codex — 중앙 검정·짙은 갈색 VIA 검사

`pad-via-inspector`의 `via_checker.py`를 별도 변경한 Codex 버전입니다.
PAD 중앙 원형 구역에서만 검정 또는 짙은 갈색 VIA를 찾고, 쏠림은 불량으로
판정하지 않습니다. PAD 외곽의 검은 선은 검색영역과 형상 필터로 제외합니다.

- 코드: [`via_codex/via_checker.py`](via_codex/via_checker.py)
- 원리·임계값 수정 가이드: [`via_codex/README.md`](via_codex/README.md)
- 회귀 테스트: [`via_codex/test_via_checker.py`](via_codex/test_via_checker.py)

## via_claude — PAD 안의 VIA 검출·판정

순수 OpenCV 만으로 PAD 안의 VIA 를 찾아 양·불량 코드(`1` / `42` / `-1`)를 돌려줍니다.
파일 하나만 복사해 가면 바로 쓸 수 있습니다.

- 사용법: [`via_claude/README.md`](via_claude/README.md)
- 검출 원리: [`via_claude/DETECTION.md`](via_claude/DETECTION.md)
- 코드: [`via_claude/via_checker.py`](via_claude/via_checker.py) (검출+판정) · [`via_claude/via_code.py`](via_claude/via_code.py) (판정만)
