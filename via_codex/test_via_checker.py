# -*- coding: utf-8 -*-
"""via_checker.py의 중앙/색상 제한 회귀 테스트.

실행:
    python -m unittest -v test_via_checker.py
"""

import unittest

import cv2
import numpy as np

from via_checker import CODE_OK, CODE_VIA_MISSING, debug_via


class ViaCheckerCenterColorTest(unittest.TestCase):
    SIZE = 128
    CENTER = (64, 64)
    PAD_RADIUS = 25
    PAD_COLOR = (185, 190, 195)  # BGR

    def _case(self):
        image = np.full((self.SIZE, self.SIZE, 3), 35, np.uint8)
        actual = np.zeros((self.SIZE, self.SIZE), np.uint8)
        pad_design = np.zeros_like(actual)
        via_design = np.zeros_like(actual)

        cv2.circle(image, self.CENTER, self.PAD_RADIUS, self.PAD_COLOR, -1)
        cv2.circle(actual, self.CENTER, self.PAD_RADIUS, 255, -1)
        # 검사기의 DESIGN_PAD_SHRINK=2 복원을 고려해 설계 PAD는 2px 작게 만듭니다.
        cv2.circle(pad_design, self.CENTER, self.PAD_RADIUS - 2, 255, -1)
        cv2.circle(via_design, self.CENTER, 2, 255, -1)
        return image, actual, pad_design, via_design

    def _run(self, draw):
        image, actual, pad_design, via_design = self._case()
        draw(image)
        return debug_via(image, actual, pad_design, via_design, quiet=True)

    def test_center_black_via_is_found(self):
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (8, 8, 8), -1))
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)

    def test_center_dark_brown_via_is_found(self):
        # HSV로 약 H=13, S=181, V=120인 짙은 갈색입니다.
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (35, 70, 120), -1))
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertGreater(rows[0]["color_candidate_pixels"], 0)
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)

    def test_detected_offset_is_not_a_defect(self):
        # 기존 기준(offset_norm > 0.30, 거리 > 2.2px)을 넘지만 중앙 검색원 안입니다.
        shifted = (self.CENTER[0] + 8, self.CENTER[1])
        code, _, _, rows = self._run(
            lambda image: cv2.circle(image, shifted, 3, (5, 5, 5), -1))
        self.assertGreater(rows[0]["offset_norm"], 0.30)
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])

    def test_outer_black_line_is_ignored(self):
        # 검은 선이 PAD 외곽에 있어도 중앙 검색원 밖이므로 VIA로 채택되지 않습니다.
        def draw(image):
            x = self.CENTER[0] + 18
            cv2.line(image, (x, self.CENTER[1] - 13),
                     (x, self.CENTER[1] + 13), (0, 0, 0), 3)

        code, _, via_bin, rows = self._run(draw)
        self.assertEqual(CODE_VIA_MISSING, code)
        self.assertEqual("VIA_MISSING", rows[0]["status"])
        self.assertEqual(0, int(np.count_nonzero(via_bin)))

    def test_bright_colored_spot_is_ignored(self):
        # 중앙에 있어도 밝은 빨강은 검정/짙은 갈색이 아니므로 VIA가 아닙니다.
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (30, 30, 230), -1))
        self.assertEqual(CODE_VIA_MISSING, code)
        self.assertEqual("VIA_MISSING", rows[0]["status"])
        self.assertEqual(0, int(np.count_nonzero(via_bin)))


if __name__ == "__main__":
    unittest.main()
