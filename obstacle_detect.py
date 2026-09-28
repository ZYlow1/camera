# -*- coding: utf-8 -*-
"""
避障识别小工具（路线 A：区域划分 + 帧间差分）

作用：用摄像头实时判断前方「左 / 中 / 右」哪一侧有东西在动。
      相当于避障系统里的「眼睛」——只负责识别，不控制任何东西。

适用范围（重要）：
    摄像头必须固定不动。如果摄像头自己在移动（比如装在行驶的小车上），
    整个画面都在变，这个方法会全部误报，等于失效。

运行：
    python obstacle_detect.py

按键：
    q    退出
    +    提高变化阈值（更迟钝，更不容易误报）
    -    降低变化阈值（更灵敏）
    d    切换调试图（看清算法到底「看到」了什么）
    m    开关镜像
"""

import cv2
import numpy as np

# ---- 可调参数（想改就改这里）----
WIDTH, HEIGHT = 1280, 720   # 摄像头分辨率（本机摄像头支持到 720p）

ZONES = 3                   # 画面横向分成几块。3 = 左 / 中 / 右

DIFF_THRESHOLD = 25         # 两帧差值多大才算「变化」（0~255）。
                            # 调大 → 更迟钝、更抗噪；调小 → 更灵敏、更容易误报。

BLUR_KERNEL = 5             # 高斯模糊的核大小，必须是奇数。作用：抹掉传感器噪点

MIN_PIXELS_RATIO = 0.02     # 某一区内「变化像素占比」超过多少，才算这一区有障碍

CONFIRM_FRAMES = 3          # 连续命中多少帧才确认（防抖，避免状态疯狂跳动）

GLOBAL_CHANGE_RATIO = 0.6   # 全画面变化占比超过多少 → 判定为光照突变/整体抖动，整帧作废

MORPH_KERNEL = np.ones((3, 3), np.uint8)   # 形态学清理用的小核


def detect_zones(prev_gray, gray, diff_threshold=DIFF_THRESHOLD):
    """核心识别逻辑（纯函数：只吃两张灰度图，不碰摄像头、不碰窗口）。

    返回 (ratios, mask)：
      ratios —— 长度 ZONES 的列表，第 i 项是第 i 区里「变化像素」的占比（0~1）
      mask   —— 二值图（0/255），白色表示检测到变化，用于显示调试图
    """
    # 1) 两帧相减：找出「哪里变了」
    diff = cv2.absdiff(prev_gray, gray)

    # 2) 高斯模糊：抹掉传感器噪点，免得一两个噪点就被当成障碍
    diff = cv2.GaussianBlur(diff, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # 3) 二值化：变化超过阈值的变成白色(255)，否则黑色(0)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # 4) 开运算（先腐蚀再膨胀）：去掉零散的小白点，保留成片的真变化
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

    # 5) 按列切块，统计每一块里的白像素占比
    h, w = mask.shape[:2]
    ratios = []
    for i in range(ZONES):
        x0 = w * i // ZONES
        x1 = w * (i + 1) // ZONES
        zone = mask[:, x0:x1]
        ratios.append(float(np.count_nonzero(zone)) / zone.size)
    return ratios, mask
