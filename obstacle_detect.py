# -*- coding: utf-8 -*-
"""
避障识别程序（深度估计方案）

判断前方「左 / 中 / 右」哪一侧有东西挡路。**只做识别，不做任何控制。**

原理（和最初那版帧差法完全不同）：
    深度模型对每个像素估计远近，我们把「近处的东西」找出来，分左中右统计占比。
    因为是单帧推理、不比较帧间变化，所以**静止不动的障碍物也能一直被看见**
    —— 这正是帧差法做不到的那件事。

关于「距离」：
    用的是**相对深度**（越亮 = 越近），没有物理单位。
    所以判据是「这块是不是比别处近得多」，而不是「这块有几米」。
    阈值必须对着你的真实场景现场标定（见 NEAR_THRESHOLD 的说明）。

两个窗口：
    camera —— 原图 + 检测框 + 分区状态色条
    depth  —— 深度伪彩色（颜色越暖/越亮 = 越近）

运行：
    python -X utf8 obstacle_detect.py

按键：
    q    退出
    [ ]  降低 / 提高「近处」阈值（现场标定用，画面上有实时数值）
    n    开关检测框
    m    开关镜像
"""

import time

import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForObjectDetection,
)

# ---- 模型路径：放在项目目录之外，不会被 git 跟踪 ----
DEPTH_MODEL_DIR = r"D:\models\Depth-Anything-V2-Small"
DETECT_MODEL_DIR = r"D:\models\rtdetr_r18vd"

CAM_W, CAM_H = 1280, 720

# ---- 判定参数（都需要对着真实场景现场调）----
ZONES = 3                   # 画面横向分成几块：3 = 左 / 中 / 右

ROI_TOP = 0.5               # 只统计画面下半部分（从 50% 高度往下）。
                            # 摄像头固定朝前时，画面底部才是「前方低处的空间」；
                            # 上半部分（天花板、远处的墙）掺进来只会稀释信号。

NEAR_THRESHOLD = 180        # 深度值 > 这个数 = 「近处的东西」。
                            # 深度图是 0~255 灰度，255 = 最近。
                            # ⚠ 这是相对值、没有物理单位，必须现场标定：
                            #   把东西放在「你觉得该报警的距离」，按 [ ] 调到
                            #   它所在区域的占比明显高于其他区为止。
                            #   空旷时各区的占比应该都很低。

MIN_NEAR_RATIO = 0.10       # 某区里「近像素」占比超过多少，才算这一区有障碍

CONFIRM_FRAMES = 3          # 连续命中多少帧才确认（防抖，避免状态乱跳）

DETECT_THRESHOLD = 0.5      # 检测框置信度门槛（检测只是辅助信息，不参与判定）
DETECT_EVERY_N = 3          # 每几帧跑一次检测；结果缓存、每帧照画，所以框不会闪


def load_models():
    """把两个模型都加载到 GPU 上。本地文件加载，不需要联网。"""
    print("加载深度模型 ...", end="", flush=True)
    depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_DIR)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_DIR).to("cuda").half().eval()
    print(" OK")

    print("加载检测模型 ...", end="", flush=True)
    det_proc = AutoImageProcessor.from_pretrained(DETECT_MODEL_DIR)
    det_model = AutoModelForObjectDetection.from_pretrained(DETECT_MODEL_DIR).to("cuda").half().eval()
    print(" OK")

    return depth_proc, depth_model, det_proc, det_model


def estimate_depth(proc, model, bgr):
    """整幅画面的「远近图」，返回与原图同尺寸的 uint8 灰度图（越亮 = 越近）。

    注意：Depth Anything V2 普通版输出的是**相对逆深度**，不是真实米数。
    想做「1 米内有障碍」的判断，需要换 Metric 版模型。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # fp16：模型和输入都压成半精度。实测比 fp32 快一倍多（52ms → 23ms），
    # 而归一化后的逐像素差异不到 0.3%，对这个判断没有影响。
    inputs = proc(images=rgb, return_tensors="pt")["pixel_values"].to("cuda").half()
    with torch.no_grad():
        out = model(pixel_values=inputs)
    d = out.predicted_depth.unsqueeze(1)
    # 模型输出比原图小（如 518x686），插值拉回原尺寸才好和画面叠着看
    d = torch.nn.functional.interpolate(d, size=bgr.shape[:2], mode="bicubic", align_corners=False).squeeze()
    # 归一化直接在 GPU 上做完，只把 1/4 大小的 uint8 拷回 CPU，省一次大拷贝
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).to(torch.uint8).cpu().numpy()


def detect(proc, model, bgr, threshold=DETECT_THRESHOLD):
    """目标检测。返回 {scores, labels, boxes}，坐标已换算回原图尺度。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt")["pixel_values"].to("cuda").half()
    with torch.no_grad():
        out = model(pixel_values=inputs)
    results = proc.post_process_object_detection(
        out,
        target_sizes=torch.tensor([bgr.shape[:2]]),   # [[高, 宽]]
        threshold=threshold,
    )[0]
    return results


def side_name(i):
    """把分区编号翻译成中文，方便打印。三分区时就是 左 / 中 / 右。"""
    names = ["左", "中", "右"]
    return names[i] if i < len(names) else f"第{i}区"


def zone_near_ratios(depth_vis, near_threshold=NEAR_THRESHOLD, roi_top=ROI_TOP):
    """判定的核心：算出每一区里「近处像素」的占比。

    纯函数——只吃一张深度图，不碰摄像头、不碰窗口，所以能脱离硬件测试。

    depth_vis      : uint8 灰度图，越亮 = 越近
    near_threshold : 多亮才算「近」
    roi_top        : 从画面高度的百分之多少往下开始统计（0.5 = 只看下半部分）
    """
    h, w = depth_vis.shape[:2]
    y0 = int(h * roi_top)
    roi = depth_vis[y0:, :]
    near = roi > near_threshold      # 布尔图：True = 这块「近」

    ratios = []
    for i in range(ZONES):
        x0 = int(w * i / ZONES)
        x1 = int(w * (i + 1) / ZONES)
        # 布尔数组的 mean() 就是 True 的占比
        ratios.append(float(near[:, x0:x1].mean()))
    return ratios


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用——本程序需要 GPU，先检查显卡驱动。")
        return

    depth_proc, depth_model, det_proc, det_model = load_models()

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("摄像头打不开。检查是否被其他软件占用。")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际分辨率：{real_w}x{real_h}")
    print("按键：q 退出 / [ ] 调近处阈值 / n 检测框 / m 镜像")
    print(f"分区数 {ZONES}，只看画面下方 {int(ROI_TOP * 100)}%，"
          f"近处阈值 {NEAR_THRESHOLD}，占比门槛 {MIN_NEAR_RATIO}")

    mirror = True
    show_boxes = True
    near_threshold = NEAR_THRESHOLD     # 可现场用 [ ] 调整
    det_res = None                      # 缓存的检测结果
    frame_idx = 0
    hit_counts = [0] * ZONES            # 每个区连续命中了几帧
    confirmed = [False] * ZONES         # 每个区是否已确认「有障碍」
    ratios = [0.0] * ZONES              # 每个区当前「近处占比」
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读不到画面")
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        # 深度：每帧都算（这是本方案真正的「眼睛」）
        depth_vis = estimate_depth(depth_proc, depth_model, frame)
        heat = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # 判定：哪一区「有东西很近」
        ratios = zone_near_ratios(depth_vis, near_threshold)
        for i in range(ZONES):
            # 命中就累加计数，没命中就归零——连续命中够多才确认，避免状态乱跳
            if ratios[i] > MIN_NEAR_RATIO:
                hit_counts[i] += 1
            else:
                hit_counts[i] = 0
            was = confirmed[i]
            confirmed[i] = hit_counts[i] >= CONFIRM_FRAMES
            # 只在状态变化时打印，否则会刷屏
            if confirmed[i] != was:
                print(f"{side_name(i)}区：{'有障碍' if confirmed[i] else '无障碍'}")

        # 检测：只补充「那是什么」，不参与判定，可以按键关掉省算力
        if show_boxes:
            # 隔 N 帧才算一次；结果缓存着，每一帧都照画，所以框不会闪
            if det_res is None or frame_idx % DETECT_EVERY_N == 0:
                det_res = detect(det_proc, det_model, frame)
            for score, label, box in zip(det_res["scores"], det_res["labels"], det_res["boxes"]):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                name = det_model.config.id2label[int(label)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {float(score):.2f}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            det_res = None      # 关掉时清空缓存，下次打开会立刻重算
        frame_idx += 1

        # ---- 可视化 ----
        h, w = frame.shape[:2]
        y0 = int(h * ROI_TOP)

        # 标出统计区域（那条黄线以下才是程序在判定的范围），两个窗口都画
        cv2.line(frame, (0, y0), (w, y0), (255, 255, 0), 1)
        cv2.line(heat, (0, y0), (w, y0), (255, 255, 0), 1)

        # 每个区一条状态色条 + 分界线：绿=无障碍，黄=本帧命中，红=已确认有障碍
        color = (0, 255, 0)
        for i in range(ZONES):
            x0 = int(w * i / ZONES)
            x1 = int(w * (i + 1) / ZONES)
            if confirmed[i]:
                color = (0, 0, 255)
            elif hit_counts[i] > 0:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x0, 0), (x1 - 1, 8), color, -1)
            cv2.line(frame, (x0, y0), (x0, h), color, 1)
            cv2.line(heat, (x0, y0), (x0, h), color, 1)
        cv2.line(frame, (w - 1, 0), (w - 1, h), color, 1)

        now = time.time()
        fps = 1 / (now - prev) if now > prev else 0
        prev = now

        # 第一行：帧率 + 当前阈值（调参时最需要看的数字）
        cv2.putText(frame, f"{fps:.1f} FPS   near:{near_threshold}   min:{MIN_NEAR_RATIO}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 第二行：每一区的实时「近处占比」——标定阈值的直接依据
        zone_text = "  ".join(f"{side_name(i)}:{ratios[i]:.2f}" for i in range(ZONES))
        cv2.putText(frame, zone_text, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("camera", frame)
        cv2.imshow("depth", heat)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('['), ord('{')):       # 「[」不用按 shift
            near_threshold = max(1, near_threshold - 5)
            print("近处阈值 =", near_threshold)
        elif key in (ord(']'), ord('}')):
            near_threshold = min(255, near_threshold + 5)
            print("近处阈值 =", near_threshold)
        elif key == ord('n'):
            show_boxes = not show_boxes
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
