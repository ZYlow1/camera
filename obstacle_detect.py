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
    [    降低最小占比（更灵敏，更容易检出小动静）
    ]    提高最小占比（更迟钝，更能压住噪声误报）
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

MIN_PIXELS_RATIO = 0.03     # 某一区内「变化像素占比」超过多少，才算这一区有障碍。
                            # 实测：画面静止时噪声占比在 0.002~0.026 之间波动（随光照变化），
                            # 所以拿 0.03 作起点——牺牲一点灵敏度来压住噪声。
                            # 另：避障通常「宁可误报，不可漏报」，发现漏检就按 [ 调低。
                            # ⚠ 这个值必须你在自己的场景里标定——画面上的实时占比就是依据。

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


def open_camera(index=0):
    """打开摄像头。

    Windows 上优先用 MSMF 后端——实测本机摄像头在 1280x720 下 MSMF 有 30 帧，
    而 DSHOW 只有 10 帧。MSMF 打不开时再依次退回 DSHOW 和默认后端。
    """
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def side_name(i):
    """把分区编号翻译成中文，方便打印。三分区时就是 左 / 中 / 右。"""
    names = ["左", "中", "右"]
    return names[i] if i < len(names) else f"第{i}区"


def main():
    cap = open_camera(0)
    if not cap.isOpened():
        print("摄像头打不开。检查是否被其他软件占用，或试试 open_camera(1)")
        return

    # 请求分辨率，并把实际生效的值打印出来（摄像头可能不接受我们请求的尺寸）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际分辨率：{real_w}x{real_h}（请求的是 {WIDTH}x{HEIGHT}）")

    print("按键说明：q 退出 / + - 调变化阈值 / [ ] 调最小占比 / d 调试图 / m 镜像")
    print(f"分区数：{ZONES}    变化阈值：{DIFF_THRESHOLD}    连续 {CONFIRM_FRAMES} 帧确认")

    mirror = True                  # 自拍镜像，看着更自然
    show_debug = False             # 是否显示二值调试图
    min_ratio = MIN_PIXELS_RATIO   # 判定阈值，运行时可用 [ ] 现场调整
    diff_threshold = DIFF_THRESHOLD
    prev_gray = None               # 上一帧灰度图
    hit_counts = [0] * ZONES       # 每个区连续命中了几帧
    confirmed = [False] * ZONES    # 每个区是否已确认「有障碍」

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读不到画面")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # 识别只需要灰度信息，彩色转灰度能省一半计算量
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 第一帧没有「上一帧」可比，先存下来，这一帧不做判断
        if prev_gray is None:
            prev_gray = gray
            continue

        ratios, mask = detect_zones(prev_gray, gray, diff_threshold)
        prev_gray = gray

        # 全画面变化过大 → 多半是开/关灯、窗帘被吹动之类的整体变化，这一帧整帧作废
        global_ratio = float(np.count_nonzero(mask)) / mask.size
        if global_ratio > GLOBAL_CHANGE_RATIO:
            hit_counts = [0] * ZONES
            confirmed = [False] * ZONES
        else:
            for i in range(ZONES):
                # 命中就累加计数，没命中就归零——连续命中够多才确认，避免状态乱跳
                if ratios[i] > min_ratio:
                    hit_counts[i] += 1
                else:
                    hit_counts[i] = 0
                was = confirmed[i]
                confirmed[i] = hit_counts[i] >= CONFIRM_FRAMES
                # 只在状态发生变化时打印，否则会刷屏
                if confirmed[i] != was:
                    print(f"{side_name(i)}区：{'有障碍' if confirmed[i] else '无障碍'}")

        # 调试图直接看二值掩码，正常图看原始画面
        if show_debug:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # 给每个分区画一条状态色条：绿=无障碍，黄=本帧命中，红=已确认有障碍
        h, w = display.shape[:2]
        for i in range(ZONES):
            x0 = w * i // ZONES
            x1 = w * (i + 1) // ZONES
            if confirmed[i]:
                color = (0, 0, 255)
            elif hit_counts[i] > 0:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(display, (x0, 0), (x1 - 1, 8), color, -1)
            cv2.line(display, (x0, 0), (x0, h), color, 1)
        cv2.line(display, (w - 1, 0), (w - 1, h), color, 1)

        # 左上角显示当前设置，方便边调边看
        cv2.putText(
            display,
            f"thr:{diff_threshold}  min:{min_ratio:.3f}  zones:{ZONES}  debug:{int(show_debug)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # 把每一区的实时占比写在画面上——这是现场标定阈值最重要的依据：
        # 画面静止时这些数字就是「噪声水平」，判定阈值（min）必须明显高于它。
        zone_text = "  ".join(f"{side_name(i)}:{ratios[i]:.3f}" for i in range(ZONES))
        cv2.putText(display, zone_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("obstacle", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):       # 「+」要按 shift，实际常收到「=」
            diff_threshold = min(255, diff_threshold + 5)
            print("变化阈值 =", diff_threshold)
        elif key in (ord('-'), ord('_')):
            diff_threshold = max(1, diff_threshold - 5)
            print("变化阈值 =", diff_threshold)
        elif key in (ord('['), ord('{')):
            min_ratio = max(0.001, round(min_ratio - 0.005, 4))
            print("最小占比 =", min_ratio)
        elif key in (ord(']'), ord('}')):
            min_ratio = min(0.5, round(min_ratio + 0.005, 4))
            print("最小占比 =", min_ratio)
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
