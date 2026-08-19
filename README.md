# Wafer_Via

## codex — wafer die-map

YOLO centre-clip cross-points를 이용하는 wafer die-map 모듈입니다.

- 구현/사용법: [`codex/README.md`](codex/README.md)
- 코드: [`codex/wafer_via.py`](codex/wafer_via.py)
- 샘플: [`codex/sample_img/Clip_sample.png`](codex/sample_img/Clip_sample.png)
- 테스트: [`tests/test_wafer_via.py`](tests/test_wafer_via.py)

## via_claude — PAD 안의 VIA 검출·판정

순수 OpenCV 만으로 PAD 안의 VIA 를 찾아 양·불량 코드(`1` / `42` / `-1`)를 돌려줍니다.
파일 하나만 복사해 가면 바로 쓸 수 있습니다.

- 사용법: [`via_claude/README.md`](via_claude/README.md)
- 검출 원리: [`via_claude/DETECTION.md`](via_claude/DETECTION.md)
- 코드: [`via_claude/via_checker.py`](via_claude/via_checker.py) (검출+판정) · [`via_claude/via_code.py`](via_claude/via_code.py) (판정만)
