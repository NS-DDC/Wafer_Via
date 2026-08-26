# -*- coding: utf-8 -*-
"""measure_die_render_angle 의 내부 흐름을 단계별 그림으로 만든다.

_projection_score 안에서 실제로 일어나는 연산을 그대로 재현해서 그린다
(코드를 복사한 것이 아니라 같은 순서/같은 파라미터로 다시 계산한 것).
"""
import os
import glob

import cv2
import numpy as np

import wafer_via_die_render as M

SRC = r"E:\app_dir\V10_Wafer\sample_img"
OUT = r"E:\app_dir\V10_Wafer\sample_img\result"
os.makedirs(OUT, exist_ok=True)

F = cv2.FONT_HERSHEY_SIMPLEX
PANEL = 460          # 각 패널 이미지 한 변
PAD = 14
HEAD = 36            # 패널 제목 높이
FOOT = 108            # 패널 아래 설명/그래프 높이


def to_bgr(g):
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def fit(img, size=PANEL):
    h, w = img.shape[:2]
    s = size / float(max(h, w))
    out = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 3), np.uint8)
    y = (size - out.shape[0]) // 2
    x = (size - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def plot_profile(values, width, height, color=(90, 220, 255), title=None):
    """1D 배열을 작은 꺾은선 그래프로."""
    img = np.full((height, width, 3), 22, np.uint8)
    v = np.asarray(values, np.float64)
    if v.size < 2:
        return img
    lo, hi = float(v.min()), float(v.max())
    rng = (hi - lo) or 1.0
    xs = np.linspace(0, width - 1, v.size)
    ys = height - 6 - (v - lo) / rng * (height - 16)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (5, 13), F, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return img


GRAPH = 84           # 패널 아래 그래프 자리(없어도 높이는 예약해 크기를 통일)


def panel(img, title, caption_lines, foot=None):
    """제목 + 이미지 + (선택)그래프 + 캡션 을 하나로. 높이는 항상 같다."""
    w = PANEL
    h = HEAD + PANEL + GRAPH + FOOT
    out = np.full((h, w, 3), 18, np.uint8)
    cv2.putText(out, title, (6, 25), F, 0.60, (120, 235, 255), 1, cv2.LINE_AA)
    out[HEAD:HEAD + PANEL] = img
    y = HEAD + PANEL
    if foot is not None:
        out[y:y + foot.shape[0]] = foot
    y += GRAPH
    for i, line in enumerate(caption_lines[:5]):
        cv2.putText(out, line, (6, y + 20 + i * 21), F, 0.45,
                    (205, 205, 205), 1, cv2.LINE_AA)
    return out


def build(path):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    wb = M.detect_wafer_boundary(img)
    cx, cy = (int(round(v)) for v in wb.center_px)
    r = int(round(float(wb.radius_px)))

    roi_ratio = M.DEFAULT_DIE_RENDER_ROI_RATIO
    max_dim = M.DEFAULT_DIE_RENDER_MAX_DIM
    search_deg = M.DEFAULT_DIE_RENDER_SEARCH_DEG
    coarse_step = M.DEFAULT_DIE_RENDER_COARSE_STEP
    fine_step = M.DEFAULT_DIE_RENDER_FINE_STEP

    # ---- 1. 원본 + wafer circle + ROI box -------------------------------
    half = max(16, int(round(r * roi_ratio)))
    p1 = img.copy()
    cv2.circle(p1, (cx, cy), r, (60, 90, 255), 4)
    cv2.rectangle(p1, (cx - half, cy - half), (cx + half, cy + half),
                  (90, 255, 120), 4)
    cv2.drawMarker(p1, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)

    # ---- 2. gray ROI ----------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    roi = gray[y0:y1, x0:x1]
    scale = min(1.0, float(max_dim) / float(max(roi.shape[:2])))
    sw = max(8, int(round(roi.shape[1] * scale)))
    sh = max(8, int(round(roi.shape[0] * scale)))
    if scale < 1.0:
        roi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    p2 = to_bgr(roi)

    # ---- 3. Otsu 이진화 + 원 마스크 = grid --------------------------------
    lcx = (cx - x0) * scale
    lcy = (cy - y0) * scale
    yy, xx = np.ogrid[:sh, :sw]
    rs = half * scale
    circle_mask = ((xx - lcx) ** 2 + (yy - lcy) ** 2 <= rs ** 2)
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    thr, binary = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grid = binary.astype(np.float32)
    grid[~circle_mask] = 0.0
    p3 = to_bgr(grid)

    inner = ((xx - lcx) ** 2 + (yy - lcy) ** 2 <= (rs * 0.92) ** 2).astype(np.float32)
    rot_c = (sw / 2.0, sh / 2.0)

    def rot(a):
        Mr = cv2.getRotationMatrix2D(rot_c, float(a), 1.0)
        out = cv2.warpAffine(grid, Mr, (sw, sh), flags=cv2.INTER_LINEAR)
        return out * inner

    def sc(a):
        g = rot(a)
        return float(g.sum(axis=0).var() + g.sum(axis=1).var())

    # ---- 4/5. 틀린 각 vs 맞은 각 ------------------------------------------
    ang = M.measure_die_render_angle(img, cx, cy, r)
    best = float(ang) if ang is not None else 0.0
    bad = best - 3.0

    g_bad, g_best = rot(bad), rot(best)

    def zoom(g):
        """가운데 일부만 잘라 확대 — 3 deg 차이를 눈으로 보이게."""
        k = 150
        yc, xc = sh // 2, sw // 2
        c = np.clip(g[yc - k:yc + k, xc - k:xc + k], 0, 255)
        return to_bgr(cv2.resize(c, (PANEL, PANEL),
                                 interpolation=cv2.INTER_NEAREST))

    p4 = zoom(g_bad)
    p5 = zoom(g_best)
    cv2.putText(p4, "zoom x%.1f" % (PANEL / 300.0), (8, PANEL - 10), F,
                0.46, (110, 130, 255), 1, cv2.LINE_AA)
    cv2.putText(p5, "zoom x%.1f" % (PANEL / 300.0), (8, PANEL - 10), F,
                0.46, (110, 255, 150), 1, cv2.LINE_AA)
    f4 = plot_profile(g_bad.sum(axis=0), PANEL, 84,
                      (110, 130, 255),
                      "column sum  ->  var = %.3g" % g_bad.sum(axis=0).var())
    f5 = plot_profile(g_best.sum(axis=0), PANEL, 84,
                      (110, 255, 150),
                      "column sum  ->  var = %.3g" % g_best.sum(axis=0).var())

    # ---- 6. score curve --------------------------------------------------
    coarse = np.arange(-search_deg, search_deg + 1e-9, coarse_step)
    cs = np.asarray([sc(a) for a in coarse])
    cbest = float(coarse[int(np.argmax(cs))])
    fine = np.arange(cbest - coarse_step, cbest + coarse_step + 1e-9, fine_step)
    fs = np.asarray([sc(a) for a in fine])

    p6 = np.full((PANEL, PANEL, 3), 22, np.uint8)
    lo, hi = float(cs.min()), float(cs.max())
    rng = (hi - lo) or 1.0
    xs = (coarse + search_deg) / (2 * search_deg) * (PANEL - 20) + 10
    ys = PANEL - 40 - (cs - lo) / rng * (PANEL - 70)
    cv2.polylines(p6, [np.stack([xs, ys], 1).astype(np.int32)], False,
                  (120, 220, 255), 2, cv2.LINE_AA)
    # 0 deg 기준선, coarse 최대, 최종 피크
    for val, col, lab in ((0.0, (70, 70, 70), "0"),
                          (cbest, (90, 160, 255), "coarse"),
                          (best, (110, 255, 150), "peak")):
        px = int((val + search_deg) / (2 * search_deg) * (PANEL - 20) + 10)
        cv2.line(p6, (px, 12), (px, PANEL - 34), col, 1, cv2.LINE_AA)
        cv2.putText(p6, lab, (max(2, px - 16), PANEL - 22), F, 0.34, col, 1, cv2.LINE_AA)
    cv2.putText(p6, "-%.0f deg" % search_deg, (6, PANEL - 6), F, 0.36,
                (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(p6, "+%.0f deg" % search_deg, (PANEL - 74, PANEL - 6), F, 0.36,
                (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(p6, "score(angle)", (PANEL - 118, 22), F, 0.40,
                (120, 220, 255), 1, cv2.LINE_AA)
    f6 = plot_profile(fs, PANEL, 84, (110, 255, 150),
                      "fine scan %.2f deg step -> parabola fit" % fine_step)

    panels = [
        panel(fit(p1), "1) input + wafer circle + ROI",
              ["detect_wafer_boundary -> cx,cy,r",
               "half = r * roi_ratio(%.2f) = %d px" % (roi_ratio, half),
               "green box = the only region measured"]),
        panel(fit(p2), "2) crop -> gray -> downscale",
              ["roi %dx%d,  max_dim=%d" % (roi.shape[1], roi.shape[0], max_dim),
               "scale = %.3f %s" % (scale, "(no downscale)" if scale >= 1 else ""),
               "colour is discarded here"]),
        panel(fit(p3), "3) blur -> Otsu -> circle mask",
              ["Otsu threshold = %.0f" % thr,
               "white = die, black = street",
               "outside circle forced to 0"]),
        panel(fit(p4), "4) WRONG angle (%+.2f deg)" % bad,
              ["rotate the binary grid,",
               "then sum every column.",
               "streets smear -> flat profile -> low var"], foot=f4),
        panel(fit(p5), "5) BEST angle (%+.2f deg)" % best,
              ["streets line up with the axis,",
               "column sums swing hard.",
               "variance is maximal -> this is the answer"], foot=f5),
        panel(p6, "6) search: coarse -> fine -> subpixel",
              ["coarse %.2f deg step over +-%.0f deg" % (coarse_step, search_deg),
               "fine %.3f deg step around the winner" % fine_step,
               "parabola fit on 3 points -> %+.4f deg" % best]),
    ]

    ph, pw = panels[0].shape[:2]
    cols, rows = 3, 2
    TITLE = 58
    sheet = np.full((TITLE + rows * ph + (rows + 1) * PAD,
                     cols * pw + (cols + 1) * PAD, 3), 10, np.uint8)
    cv2.putText(sheet, "measure_die_render_angle  --  data flow", (PAD, 30),
                F, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, "%s   |   wafer r=%d   pitch~%dpx   ->  angle %+.4f deg"
                % (name, r, int(round(38)), best), (PAD, 50), F, 0.46,
                (150, 200, 255), 1, cv2.LINE_AA)
    for i, p in enumerate(panels):
        rr, cc = divmod(i, cols)
        y = TITLE + PAD + rr * (ph + PAD)
        x = PAD + cc * (pw + PAD)
        sheet[y:y + ph, x:x + pw] = p
    dst = os.path.join(OUT, "flow_%s.png" % name)
    cv2.imwrite(dst, sheet)
    print("wrote", dst, sheet.shape)
    return dst


for p in sorted(glob.glob(os.path.join(SRC, "*.png"))):
    build(p)
