# -*- coding: utf-8 -*-
"""obstacle_detect.py 核心识别逻辑的单元测试。

不需要摄像头：直接喂两张「人工构造」的灰度图，检查分区占比是否符合预期。

运行（在项目根目录）：
    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

import obstacle_detect as od


def blank(w=300, h=300):
    """造一张全黑的灰度图（相当于「画面完全没变」）。"""
    return np.zeros((h, w), dtype=np.uint8)


class TestDetectZones(unittest.TestCase):

    def test_identical_frames_report_no_change(self):
        """两帧完全一样 → 每一区占比都应该是 0。"""
        a = blank()
        ratios, mask = od.detect_zones(a, a.copy())
        self.assertEqual(len(ratios), od.ZONES)
        self.assertTrue(all(r == 0.0 for r in ratios))
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_change_in_left_zone_only(self):
        """只有左区有变化 → 只有左区占比大于 0。

        变化块刻意远离分区边界（x 从 10 到 90，左区是 0~99），
        这样高斯模糊的扩散不会沾到中区，测试才是确定的。
        """
        prev = blank()
        cur = prev.copy()
        cur[100:200, 10:90] = 255
        ratios, _ = od.detect_zones(prev, cur)
        self.assertGreater(ratios[0], 0.0)
        self.assertEqual(ratios[1], 0.0)
        self.assertEqual(ratios[2], 0.0)

    def test_change_in_middle_zone_only(self):
        """只有中区有变化。"""
        prev = blank()
        cur = prev.copy()
        cur[100:200, 110:190] = 255
        ratios, _ = od.detect_zones(prev, cur)
        self.assertEqual(ratios[0], 0.0)
        self.assertGreater(ratios[1], 0.0)
        self.assertEqual(ratios[2], 0.0)

    def test_change_in_right_zone_only(self):
        """只有右区有变化。"""
        prev = blank()
        cur = prev.copy()
        cur[100:200, 210:290] = 255
        ratios, _ = od.detect_zones(prev, cur)
        self.assertEqual(ratios[0], 0.0)
        self.assertEqual(ratios[1], 0.0)
        self.assertGreater(ratios[2], 0.0)

    def test_tiny_noise_stays_below_threshold(self):
        """孤立的单个噪点不应该触发报警（占比要小于判定阈值）。"""
        prev = blank()
        cur = prev.copy()
        cur[150, 50] = 255
        ratios, _ = od.detect_zones(prev, cur)
        self.assertLess(ratios[0], od.MIN_PIXELS_RATIO)

    def test_whole_frame_change_hits_every_zone(self):
        """整幅画面都变了 → 每一区占比都很高（这是光照突变的情形）。"""
        prev = blank()
        cur = np.full((300, 300), 255, dtype=np.uint8)
        ratios, _ = od.detect_zones(prev, cur)
        self.assertTrue(all(r > 0.5 for r in ratios))
