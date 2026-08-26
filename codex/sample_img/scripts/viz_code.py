# -*- coding: utf-8 -*-
"""measure_die_render_angle 의 '코드 흐름' 을 한 장으로 그린다.

앞서 만든 flow_*.png 는 '이미지가 어떻게 변하는가'(data flow) 였고,
이건 '어느 함수가 어떤 순서로 불리고 어디서 갈라지는가'(control flow) 다.
줄번호는 전부 wafer_via_die_render.py 기준.

cv2.putText 는 한글을 못 그리므로 라벨은 전부 ASCII.
"""
import os

import cv2
import numpy as np

OUT = r"E:\app_dir\V10_Wafer\sample_img\result\codeflow.png"

W, H = 1560, 2150
F = cv2.FONT_HERSHEY_SIMPLEX

BG = (18, 18, 18)
img = np.full((H, W, 3), BG, np.uint8)

# 색: 호출=파랑, 분기=주황, 반환=초록, 죽은인자=회색, 경고=빨강
C_CALL = (255, 196, 120)
C_BRANCH = (110, 190, 255)
C_RET = (140, 240, 160)
C_DEAD = (110, 110, 110)
C_WARN = (110, 120, 255)
C_TXT = (225, 225, 225)
C_DIM = (150, 150, 150)


def text(s, x, y, scale=0.46, color=C_TXT, thick=1):
    cv2.putText(img, s, (x, y), F, scale, color, thick, cv2.LINE_AA)


def box(x, y, w, h, color, title, lines, fill=(30, 30, 30), tscale=0.52):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    text(title, x + 12, y + 26, tscale, color, 1)
    for i, ln in enumerate(lines):
        text(ln, x + 12, y + 50 + i * 20, 0.42, C_TXT)
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, color=C_DIM, label=None, lx=6, ly=-6):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA, tipLength=0.13)
    if label:
        text(label, min(x1, x2) + lx, (y1 + y2) // 2 + ly, 0.40, color)


# ---------------------------------------------------------------- header
text("measure_die_render_angle  --  code flow", 24, 40, 0.80, (120, 235, 255), 2)
text("wafer_via_die_render.py   |   line numbers are exact", 24, 68, 0.46, C_DIM)

cv2.line(img, (24, 84), (W - 24, 84), (60, 60, 60), 1)

# 범례
lx = 24
for lab, col in (("call", C_CALL), ("branch", C_BRANCH), ("return", C_RET),
                 ("unused arg", C_DEAD), ("danger", C_WARN)):
    cv2.rectangle(img, (lx, 96), (lx + 14, 110), col, -1)
    text(lab, lx + 20, 108, 0.40, C_DIM)
    lx += 20 + 9 * len(lab) + 26

# ================================================================ LEFT column
LX = 24
LW = 700

y = 132
box(LX, y, LW, 96, C_CALL,
    "build_die_map_from_yolo(...)   L2245",
    ["public entry. angle_align_method = 'die_render' | 'yolo'",
     "angle_align_enabled = True | False"])
a_entry = y + 96

y = 250
box(LX, y, LW, 92, C_CALL,
    "_build_die_map_from_yolo_yolo(...)   L2276",
    ["YOLO detections -> base_dm  (pitch, wafer_cx/cy/r, grid_angle)",
     "runs FIRST. reaching the angle code means dies were found."])
arrow(LX + LW // 2, a_entry, LX + LW // 2, y)
a1 = y + 92

y = 362
box(LX, y, LW, 70, C_BRANCH,
    "branch on angle_align_enabled / method   L2294",
    ["three mutually exclusive paths ->"])
arrow(LX + LW // 2, a1, LX + LW // 2, y)
a2 = y + 70

# 3 branches
bw = 222
bx0 = LX
by = 470
box(bx0, by, bw, 132, C_BRANCH, "enabled == False", [
    "L2296  final_angle = 0.0",
    "source = 'off'",
    "",
    "NOT rotated, but the",
    "aligned_image field is",
    "still produced (== input)."])
box(bx0 + (bw + 17), by, bw, 132, C_BRANCH, "method == 'yolo'", [
    "L2284  final_angle =",
    "  base_dm.grid_angle_deg",
    "source = 'yolo'",
    "",
    "V5 angle never runs."])
box(bx0 + 2 * (bw + 17), by, bw, 132, C_CALL, "method == 'die_render'", [
    "L2304  measure_die_",
    "  render_angle(wafer, cx,",
    "  cy, r, search_deg, ...)",
    "",
    "-> RIGHT column"])
for i in range(3):
    arrow(LX + LW // 2, a2, bx0 + i * (bw + 17) + bw // 2, by)

a3 = by + 132

y = 640
box(LX, y, LW, 96, C_BRANCH,
    "measured_angle is None ?   L2315",
    ["yes -> final_angle = 0.0, source = 'die_render_no_signal'",
     "no  -> final_angle = measured, confidence = 1.0",
     "confidence is a yes/no flag, NOT a quality score  (L2326)"])
arrow(bx0 + 2 * (bw + 17) + bw // 2, a3, LX + LW // 2, y)
a4 = y + 96

y = 756
box(LX, y, LW, 92, C_CALL,
    "generate_die_map(...)   L2357",
    ["base_dm geometry + final_angle -> the returned die map",
     "grid_estimate rewritten; YOLO pair fields cleared (L2348)"])
arrow(LX + LW // 2, a4, LX + LW // 2, y)
a5 = y + 92

y = 868
box(LX, y, LW, 110, C_RET,
    "align + return   L2379-2399",
    ["_alignment_matrices -> original_to_aligned / aligned_to_original",
     "align_wafer_image  -> result.aligned_image      (L2385)",
     "result.die_render_info = {'angle','confidence','agree','source'}",
     "return result"])
arrow(LX + LW // 2, a5, LX + LW // 2, y)

# ================================================================ RIGHT column
RX = 780
RW = 756

y = 132
box(RX, y, RW, 190, C_CALL,
    "measure_die_render_angle(...)   L2178",
    ["used  : image_bgr, wafer_cx, wafer_cy, wafer_r,",
     "        search_deg, coarse_step, fine_step, center,",
     "        roi_ratio, max_dim",
     "",
     "the whole body is 5 lines (L2201-2206)."])
# 죽은 인자 박스
box(RX + 430, y + 96, 314, 84, C_DEAD, "accepted but never read", [
    "die_rects   dies",
    "grid_method  thickness",
    "-> back-compat only (L2079)"], fill=(24, 24, 24), tscale=0.44)
r1 = y + 190

y = 350
box(RX, y, RW, 250, C_CALL,
    "_projection_score(image, cx, cy, r, roi_ratio, max_dim)   L2083",
    ["L2094  BGR -> gray",
     "L2096  half = max(16, round(r * roi_ratio))          # 613*0.55 = 337",
     "L2097  crop [cx +- half, cy +- half]",
     "L2099  if the box is < 8 px  ->  return None   (only None path #1)",
     "L2103  scale = min(1, max_dim / max(roi.shape))      # 674 -> no resize",
     "L2115  circle_mask   radius = half * scale",
     "L2119  GaussianBlur(3,3) -> L2120 Otsu threshold",
     "L2124  grid[~circle_mask] = 0",
     "L2125  if grid.sum() < 1  ->  return None      (only None path #2)",
     "L2128  inner_mask at 0.92*r   # blocks corner bleed while rotating"])
arrow(RX + RW // 2, r1, RX + RW // 2, y)
r2 = y + 250

y = 630
box(RX, y, RW, 132, C_RET,
    "returns a CLOSURE, not a number   L2135-2143",
    ["def score(angle_deg):",
     "    M = getRotationMatrix2D(center=(w/2, h/2), angle_deg, 1.0)",
     "    rot = warpAffine(grid, M) * inner_mask",
     "    return rot.sum(axis=0).var() + rot.sum(axis=1).var()",
     "grid / inner_mask are captured once -> Otsu runs ONE time total."])
arrow(RX + RW // 2, r2, RX + RW // 2, y)
r3 = y + 132

y = 792
box(RX, y, RW, 200, C_CALL,
    "_search_peak(score, center, search_deg, coarse, fine)   L2146",
    ["L2153  coarse = arange(center-6.0, center+6.0, 0.15)   # 81 calls",
     "L2157  coarse_best = coarse[argmax]",
     "L2159  fine = arange(coarse_best-0.15, +0.15, 0.02)    # 16 calls",
     "L2165  best_index = argmax(fine_scores)",
     "L2168  if the max is interior:",
     "L2172      denom = s[i-1] - 2*s[i] + s[i+1]",
     "L2174      best += 0.5*(s[i-1]-s[i+1])/denom * fine_step   # parabola",
     "~97 score() calls total, i.e. 97 warpAffine on a 674x674 f32."])
arrow(RX + RW // 2, r3, RX + RW // 2, y)
r4 = y + 200

y = 1022
box(RX, y, RW, 70, C_RET,
    "return float(ang)   L2206",
    ["a single number. no confidence, no diagnostics, no second opinion."])
arrow(RX + RW // 2, r4, RX + RW // 2, y)
r5 = y + 70

# 되돌아가는 화살표
cv2.arrowedLine(img, (RX, r5 - 34), (LX + LW + 12, 700), (90, 150, 110), 1,
                cv2.LINE_AA, tipLength=0.03)
text("back to L2315", LX + LW + 20, 692, 0.40, C_RET)

# ================================================================ 하단: 위험 구역
y = 1140
box(RX, y, RW, 236, C_WARN,
    "the only two ways out",
    ["return None   <- ROI could not be built, or the mask is empty.",
     "                 that is ALL. it never means 'no grid found'.",
     "float         <- every other case, including a completely wrong one.",
     "",
     "search_deg = 6.0 is a hard wall (L2153). outside it:",
     "   +-7, +-8    -> clamps to +-6.15",
     "   +-10, +-12  -> arbitrary, sign can flip",
     "   amber -10   -> returned -3.8155 : a plausible-looking wrong answer",
     "",
     "no exception, no None, no flag. the caller cannot tell."])

# 왼쪽 하단: 전체 요약
y = 1010
box(LX, y, LW, 366, C_DIM,
    "what this replaced",
    ["the previous stack in this file did:",
     "   FFT peak -> lattice gate -> grid cue -> iterate -> agree/disagree",
     "and produced a confidence number from all of it.",
     "",
     "V5's version is the 5 lines at L2201-2206:",
     "   score = _projection_score(...)",
     "   if score is None: return None",
     "   ang, _ = _search_peak(score, center, search_deg, coarse, fine)",
     "   return float(ang)",
     "",
     "there is no grid detection anywhere in the measurement path.",
     "the score is var(colsum) + var(rowsum) and nothing else --",
     "it asks 'at which angle is the lattice most axis-aligned'",
     "and answers directly, with no intermediate model to be wrong about."],
    fill=(24, 24, 24))

# ================================================================ 하단 타임라인
y = 1420
cv2.line(img, (24, y), (W - 24, y), (60, 60, 60), 1)
text("call sequence, flattened", 24, y + 34, 0.62, (120, 235, 255), 1)

seq = [
    ("build_die_map_from_yolo", "L2245", C_CALL, 0),
    ("_build_die_map_from_yolo_yolo", "L2276", C_CALL, 1),
    ("_load_bgr", "L2283", C_CALL, 1),
    ("branch: enabled / method", "L2294", C_BRANCH, 1),
    ("measure_die_render_angle", "L2304", C_CALL, 2),
    ("_projection_score", "L2201", C_CALL, 3),
    ("cvtColor / crop / resize", "L2094-2109", C_DIM, 4),
    ("GaussianBlur + Otsu", "L2119-2122", C_DIM, 4),
    ("build circle + inner mask", "L2115-2132", C_DIM, 4),
    ("-> closure score(angle)", "L2135", C_RET, 4),
    ("_search_peak", "L2205", C_CALL, 3),
    ("score(a) x 81   coarse 0.15 deg", "L2156", C_DIM, 4),
    ("score(a) x 16   fine 0.02 deg", "L2164", C_DIM, 4),
    ("parabola on 3 points", "L2168-2174", C_DIM, 4),
    ("-> float angle", "L2206", C_RET, 3),
    ("None check", "L2315", C_BRANCH, 2),
    ("generate_die_map", "L2357", C_CALL, 1),
    ("_alignment_matrices", "L2379", C_CALL, 1),
    ("align_wafer_image", "L2385", C_CALL, 1),
    ("return result", "L2399", C_RET, 1),
]

yy = y + 62
for name, line, col, depth in seq:
    x = 40 + depth * 34
    cv2.circle(img, (x, yy - 5), 3, col, -1, cv2.LINE_AA)
    if depth > 0:
        cv2.line(img, (x - 34 + 3, yy - 5), (x - 4, yy - 5), (55, 55, 55), 1)
    text(name, x + 12, yy, 0.44, col if col != C_DIM else C_TXT)
    text(line, 640, yy, 0.42, C_DIM)
    yy += 26

# 오른쪽 하단 비용
box(RX, 1482, RW, 120, C_DIM, "cost per call", [
    "Otsu / mask / resize : 1 time",
    "warpAffine 674x674 float32 : ~97 times   (81 coarse + 16 fine)",
    "everything else is numpy .var() on a 674-long vector.",
    "-> the whole measurement is one Otsu plus ~97 rotations."],
    fill=(24, 24, 24))

box(RX, 1622, RW, 120, C_DEAD, "what is NOT here", [
    "no grid / corner detection      no template rendering",
    "no FFT                          no confidence estimate",
    "no multi-scale, no iteration    no fallback to a second method",
    "die_rects / dies / grid_method / thickness are never touched."],
    fill=(24, 24, 24))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, img)
print("saved", OUT, img.shape)
