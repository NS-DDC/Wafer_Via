# -*- coding: utf-8 -*-
"""
build_single_claude.py
====================================================================
via_refine_claude.py + via_grid_claude.py + via_diemap_claude.py
    -> wafer_via_claude.py  (통짜 한 파일)

왜 스크립트로 합치나
--------------------------------------------------------------------
손으로 2200줄을 옮기면 옮기다 틀린다. 그리고 원본 세 파일을 고친 뒤
합친 파일을 다시 만들지 않으면 **두 벌이 조용히 갈라진다**. 이 스크립트가
있으면 언제든 다시 만들 수 있고, test_yolo_api_claude.py 를
VIA_MODULE=wafer_via_claude 로 한 번 더 돌려서 두 벌이 같은지 확인한다.

하는 일은 세 가지뿐이다.
  1) 각 모듈에서 헤더(docstring + import + __all__) 를 떼고 본문만 남긴다
  2) 같은 패키지 안끼리 부르던 지연 import 를 지운다 (이제 한 파일이라 필요없다)
  3) 하나의 헤더 + 하나의 __all__ 을 앞에 붙인다

실행
--------------------------------------------------------------------
    python build_single_claude.py
    python build_single_claude.py --out ../wafer_via_claude.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SRC = ["via_refine_claude.py", "via_grid_claude.py", "via_diemap_claude.py"]
OUT = "wafer_via_claude.py"

# 본문 시작점. 세 파일 모두 이 줄로 import 블록이 끝난다.
IMPORT_END = "import numpy as np\n"

# 지울 지연 import (한 파일이 됐으니 자기 자신을 부르는 꼴이 된다)
DEFERRED = re.compile(
    r"^[ \t]*from via_(?:refine|grid|diemap)_claude import [^\n]*\n", re.M)

# 남은 문자열/주석 속 모듈 이름
RENAME = re.compile(r"via_(?:refine|grid|diemap)_claude")

HEADER = '''# -*- coding: utf-8 -*-
"""
wafer_via_claude.py
====================================================================
YOLO 십자점 -> 격자 -> 웨이퍼 die map.  **이 파일 하나면 된다.**

    pip install numpy opencv-python        # 이게 전부

쓰는 법
--------------------------------------------------------------------
    import cv2
    from ultralytics import YOLO
    from wafer_via_claude import build_die_map_from_yolo

    wafer_bgr      = cv2.imread("wafer.png")            # 10000x10000
    center_clip_bgr = wafer_bgr[h//2-256:h//2+256,      # 정중앙 512x512
                                w//2-256:w//2+256]
    results = YOLO("best.pt")(center_clip_bgr)

    dm = build_die_map_from_yolo(
        wafer_image=wafer_bgr,
        clip_image=center_clip_bgr,
        detections=results[0].boxes.xywh.cpu().numpy(),
        detection_format="xywh",
        refine=True,
        refine_mode="auto",
        refine_radius=24,
        refine_noise_kernel=5,
        refine_min_confidence=0.15,
    )

    print(dm.x0, dm.y0)          # 전체 wafer 좌표의 center corner
    print(dm.pitch_x, dm.pitch_y)
    print(dm.grid_angle_deg)
    print(dm.angle_confidence)   # P(|각도오차| < 0.05deg), 0..1
    print(dm.num_dies)
    print(dm.dies)               # [{index, center_px, quad_px, ...}, ...]
    print(dm.dies_by_index)      # {(i,j): 위와 **같은 객체**}
    print(dm.wafer_boundary)     # WaferBoundary(cx, cy, r, contour)
    print(dm.aligned_image)      # 회전 보정본 (처음 볼 때 만든다, 286 MB)

    # 좌표 -> die
    info = locate_die_via(dm, point=(7321.0, 4180.5))
    print(info["die_index"], info["die_center_px"], info["is_edge"])

파이프라인
--------------------------------------------------------------------
    512x512 클립 + YOLO 검출 (N,4)
        |  detections_to_points     bbox 중심만 뽑는다
        |  refine_points            서브픽셀 보정 (색-무관)
        |  fit_grid                 센터 코너 / pitch_x,y / 회전각 / 신뢰도
        |  detect_wafer_adaptive    웨이퍼 외곽선 (색-무관)
        v  build_die_map_via        격자를 웨이퍼 전체로 외삽
    ViaDieMap

색에 안 흔들리는 이유
--------------------------------------------------------------------
회색조로 안 바꾼다. 회색조는 밝기가 같고 색만 다른 die/street 쌍을
통째로 지워버린다. 대신 **배경색으로부터의 Lab 거리**를 본다.
"검정보다 밝은가"가 아니라 "배경과 다른가"라서 극성 가정이 없다.

이 파일은 build_single_claude.py 가 만든 것이다 (직접 고치지 말고
via_refine/via_grid/via_diemap 을 고친 뒤 다시 만들 것).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    # --- 진입점 (보통 이것만 쓴다) ---------------------------------
    "build_die_map_from_yolo", "locate_die_via",
    # --- 결과 형식 -------------------------------------------------
    "ViaDieMap", "WaferBoundary", "GridFit", "RefineResult",
    "WaferProfile", "DieMapDiag",
    # --- 단계별로 직접 부르고 싶을 때 ------------------------------
    "detections_to_points", "refine_points", "streetness_map",
    "rough_pitch_from_points", "fit_grid", "analyze_clip",
    "detect_wafer_adaptive", "clean_wafer_outside", "build_die_map_via",
    # --- 잘라내기 / 확인용 -----------------------------------------
    "clip_die", "crop_die", "save_debug_overlay",
]
'''

BANNER = """

# ####################################################################
# %s  (원본: %s)
# ####################################################################
"""

TITLE = {
    "via_refine_claude.py": "1) 서브픽셀 보정 - YOLO 좌표를 십자 중심으로 당긴다",
    "via_grid_claude.py":   "2) 격자 - 센터 코너 / pitch_x,y / 회전각 / 신뢰도",
    "via_diemap_claude.py": "3) die map - 웨이퍼 외곽선, 격자 외삽, 좌표->die",
}


def strip_header(text: str, name: str) -> str:
    """모듈 헤더(docstring + import + __all__) 를 떼고 본문만 돌려준다."""
    if IMPORT_END not in text:
        raise SystemExit("%s 에서 import 끝(%r)을 못 찾았다" % (name, IMPORT_END))
    body = text.split(IMPORT_END, 1)[1]
    # 바로 뒤에 오는 __all__ = [ ... ] 블록을 지운다
    body = re.sub(r"\A\s*__all__\s*=\s*\[[^\]]*\]\s*", "", body)
    return body.strip("\n")


def build(src_dir: Path, out: Path) -> None:
    parts = [HEADER.rstrip("\n")]
    n_deferred = 0
    for name in SRC:
        p = src_dir / name
        if not p.exists():
            raise SystemExit("원본이 없다: %s" % p)
        body = strip_header(p.read_text(encoding="utf-8"), name)
        body, k = DEFERRED.subn("", body)
        n_deferred += k
        parts.append(BANNER % (TITLE[name], name))
        parts.append(body)

    merged = "\n".join(parts) + "\n"
    # 문자열/주석에 남은 옛 모듈 이름을 새 파일 이름으로 바꾼다
    merged = RENAME.sub(out.stem, merged)
    out.write_text(merged, encoding="utf-8")

    n_lines = merged.count("\n")
    print("%s  <-  %s" % (out, " + ".join(SRC)))
    print("  %d 줄, %.0f KB, 지연 import %d개 제거"
          % (n_lines, len(merged.encode("utf-8")) / 1024.0, n_deferred))
    if RENAME.search(merged):
        print("  경고: 옛 모듈 이름이 아직 남아 있다")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    build(here, (here / args.out).resolve())


if __name__ == "__main__":
    _main()
