# -*- coding: utf-8 -*-
"""
深度估计 + 目标检测 双窗口实时 demo（探路脚本，不是最终程序）

它只回答两个问题：
  1) 深度估计的真实效果如何——**静止不动的物体也能一直被「看见」**，这正是帧差法做不到的
  2) 两层叠在一起跑，在你这台机器上还剩多少帧率

两个窗口：
  camera —— 摄像头原图 + 检测框（RT-DETR）
  depth  —— 深度伪彩色（INFERNO 色表，颜色越暖/越亮 = 越近）

运行：
    python depth_demo.py

按键：
    q  退出
    n  开关检测框
    m  开关镜像
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

# ---- 模型路径：放在项目目录之外，不会被 git 跟踪（.gitignore 也已兜底）----
DEPTH_MODEL_DIR = r"D:\models\Depth-Anything-V2-Small"
DETECT_MODEL_DIR = r"D:\models\rtdetr_r18vd"

CAM_W, CAM_H = 1280, 720
DETECT_THRESHOLD = 0.5      # 检测框的置信度门槛，调低会看到更多框（远处、半身的目标置信度天然就低）

DETECT_EVERY_N = 3          # 每几帧才真正跑一次检测。深度仍然每帧都算；
                            # 检测结果会缓存下来每帧照画——所以框不会闪，
                            # 只是位置最多滞后 N 帧（对「那是什么」这种慢信息无所谓）。


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
    （若将来要拿原始数值做判断，把最后一行改成拷 float 回去即可。）
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # fp16：模型和输入都压成半精度。实测比 fp32 快一倍多（52ms → 23ms），
    # 而归一化后的逐像素差异不到 0.3%，对看画面/做判断都没影响。
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


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用——这个 demo 依赖 GPU，先检查显卡驱动。")
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

    print("按键：q 退出 / n 开关检测框 / m 镜像")

    mirror = True
    show_boxes = True
    prev = time.time()
    frame_idx = 0
    det_res = None              # 缓存的检测结果，隔帧才算一次，但每帧都拿来画

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读不到画面")
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        # 深度：每帧都算（这是这个方案真正的「眼睛」）
        # estimate_depth 直接返回 uint8 灰度（越亮 = 越近），归一化已在 GPU 上做完
        depth_vis = estimate_depth(depth_proc, depth_model, frame)
        heat = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # 检测：只是给深度加上「那是什么」，可以按键关掉
        if show_boxes:
            # 隔 N 帧才算一次检测（省算力）；结果缓存着，每一帧都照画一遍，
            # 所以框始终在画面上，不会一闪一闪
            if det_res is None or frame_idx % DETECT_EVERY_N == 0:
                det_res = detect(det_proc, det_model, frame)
            for score, label, box in zip(det_res["scores"], det_res["labels"], det_res["boxes"]):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                name = det_model.config.id2label[int(label)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {float(score):.2f}", (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            det_res = None      # 关掉检测框时清空缓存，下次打开会立刻重算

        frame_idx += 1

        now = time.time()
        fps = 1 / (now - prev) if now > prev else 0
        prev = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("camera", frame)
        cv2.imshow("depth", heat)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            show_boxes = not show_boxes
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
