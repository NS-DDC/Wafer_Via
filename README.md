# Wafer_Via

## codex — wafer die-map

YOLO centre-clip cross-points를 이용하는 wafer die-map 모듈입니다.

- 구현/사용법: [`codex/README.md`](codex/README.md)
- 코드: [`codex/wafer_via.py`](codex/wafer_via.py)
- 샘플: [`codex/sample_img/Clip_sample.png`](codex/sample_img/Clip_sample.png)
- 테스트: [`tests/test_wafer_via.py`](tests/test_wafer_via.py)

## via_codex — PAD 평균 밝기 기반 중앙 VIA 검사

`pad-via-inspector`의 `via_checker.py`를 별도 변경한 Codex 버전입니다.
회색조에서 PAD 평균보다 충분히 어두운 연결성분을 뽑고, 실제 후보 픽셀이 PAD
중앙 허용영역에 있을 때 VIA로 인정합니다. 쏠림 불량 및 PAD 커버리지 사전
필터는 사용하지 않습니다.

- 코드: [`via_codex/via_checker.py`](via_codex/via_checker.py)
- 원리·임계값 수정 가이드: [`via_codex/README.md`](via_codex/README.md)
- 회귀 테스트: [`via_codex/test_via_checker.py`](via_codex/test_via_checker.py)

## via_claude — PAD 안의 VIA 검출·판정

순수 OpenCV 만으로 PAD 안의 VIA 를 찾아 양·불량 코드(`1` / `42` / `-1`)를 돌려줍니다.
파일 하나만 복사해 가면 바로 쓸 수 있습니다.

- 사용법: [`via_claude/README.md`](via_claude/README.md)
- 검출 원리: [`via_claude/DETECTION.md`](via_claude/DETECTION.md)
- 코드: [`via_claude/via_checker.py`](via_claude/via_checker.py) (검출+판정) · [`via_claude/via_code.py`](via_claude/via_code.py) (판정만)

## claude — YOLO 십자점 -> 10000x10000 wafer die map

512x512 센터 클립의 YOLO 십자 좌표를 서브픽셀 보정해 pitch_x/pitch_y/회전각을
구하고, 웨이퍼 외곽선을 검출해 die map 을 만듭니다. 좌표를 넣으면 die index 가
나옵니다. 색 조합이 바뀌어도 같은 코드가 돕니다.

- 사용법/실측 성능: [`claude/README.md`](claude/README.md)
- 보정: [`claude/via_refine_claude.py`](claude/via_refine_claude.py)
- 격자/회전각: [`claude/via_grid_claude.py`](claude/via_grid_claude.py)
- 외곽선/die map/locate/오버레이: [`claude/via_diemap_claude.py`](claude/via_diemap_claude.py)
- 합성 테스트 데이터: `claude/synth_clip_claude.py` · `claude/synth_wafer_claude.py`
- 테스트: `claude/test_refine_claude.py` · `claude/test_grid_claude.py` · `claude/test_diemap_claude.py`

![overlay](claude/overlay_example.png)
