# -*- coding: utf-8 -*-
"""via_checker.py의 평균 밝기/중앙 위치 회귀 테스트.

실행:
    python -m unittest -v test_via_checker.py
"""

import unittest

import cv2
import numpy as np

from via_checker import CODE_OK, CODE_VIA_MISSING, debug_via


class ViaCheckerMeanCenterTest(unittest.TestCase):
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

    def _run(self, draw, dark_offset=0.0):
        image, actual, pad_design, via_design = self._case()
        draw(image)
        return debug_via(image, actual, pad_design, via_design,
                         quiet=True, dark_offset=dark_offset)

    def test_center_black_via_is_found(self):
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (8, 8, 8), -1))
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)

    def test_center_dark_brown_via_is_found(self):
        # 색상 분류 없이 회색조에서 PAD 평균보다 어두운지만 봅니다.
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (35, 70, 120), -1))
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertGreater(rows[0]["dark_candidate_pixels"], 0)
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)

    def test_small_center_offset_is_still_ok(self):
        shifted = (self.CENTER[0] + 5, self.CENTER[1])
        code, _, _, rows = self._run(
            lambda image: cv2.circle(image, shifted, 3, (5, 5, 5), -1))
        self.assertGreater(rows[0]["offset_px"], 2.2)
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])

    def test_dark_blob_away_from_center_is_ignored(self):
        shifted = (self.CENTER[0] + 12, self.CENTER[1])
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, shifted, 4, (5, 5, 5), -1))
        self.assertEqual(CODE_VIA_MISSING, code)
        self.assertEqual("VIA_MISSING", rows[0]["status"])
        self.assertEqual(0, int(np.count_nonzero(via_bin)))

    def test_outer_black_line_is_ignored(self):
        # 검은 선의 연결성분 중심이 PAD 중심에서 멀어 VIA로 채택되지 않습니다.
        def draw(image):
            x = self.CENTER[0] + 18
            cv2.line(image, (x, self.CENTER[1] - 13),
                     (x, self.CENTER[1] + 13), (0, 0, 0), 3)

        code, _, via_bin, rows = self._run(draw)
        self.assertEqual(CODE_VIA_MISSING, code)
        self.assertEqual("VIA_MISSING", rows[0]["status"])
        self.assertEqual(0, int(np.count_nonzero(via_bin)))

    def test_small_gray_drop_is_ignored(self):
        # PAD 평균보다 약간만 어두우면 기본 VIA_GRAY_DROP을 넘지 못합니다.
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (170, 175, 180), -1))
        self.assertEqual(CODE_VIA_MISSING, code)
        self.assertEqual("VIA_MISSING", rows[0]["status"])
        self.assertEqual(0, int(np.count_nonzero(via_bin)))

    def test_positive_dark_offset_accepts_lighter_via(self):
        code, _, via_bin, rows = self._run(
            lambda image: cv2.circle(image, self.CENTER, 4, (170, 175, 180), -1),
            dark_offset=15)
        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)

    def test_empty_bin_mask_does_not_skip_via_check(self):
        image, actual, pad_design, via_design = self._case()
        actual.fill(0)  # 예전 PAD_PRESENT_MIN 필터라면 PAD_ABSENT로 건너뛰던 입력
        cv2.circle(image, self.CENTER, 4, (8, 8, 8), -1)

        code, _, via_bin, rows = debug_via(
            image, actual, pad_design, via_design, quiet=True)

        self.assertEqual(CODE_OK, code)
        self.assertEqual("OK", rows[0]["status"])
        self.assertIsNone(rows[0]["pad_coverage"])
        self.assertGreater(int(np.count_nonzero(via_bin)), 0)


if __name__ == "__main__":
    unittest.main()
