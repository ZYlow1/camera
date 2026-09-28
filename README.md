# camera — 摄像头避障识别实验

用摄像头做图像识别，服务于机器人/小车的**避障**。只做「识别」，不做控制。

一句话讲清它的思路：**深度估计负责「有没有东西、有多近」，目标检测负责「那是什么」。**

---

## 当前状态

| 阶段 | 状态 |
|---|---|
| 摄像头实时预览工具 | ✅ 完成 |
| 帧差法避障（旧路线） | ❌ 已放弃并删除代码——它检测不到**静止**障碍，见下 |
| 深度估计 + 目标检测 demo | ✅ 可实时运行（约 **29 FPS**） |
| 避障**判定逻辑**（有/无障碍） | ⬜ 尚未实现 |

> **为什么放弃帧差法？** 它比较相邻两帧，检测的是「画面此刻是否在变化」。所以障碍物一旦停住就不再报警；反过来，它还会「忘记」已经确认过的障碍。而这正是避障最需要的能力。详见 [设计文档](docs/superpowers/specs/2026-09-28-obstacle-detection-design.md)。

---

## 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10+（开发环境为 3.12.8） |
| GPU | **需要**（开发环境为 RTX 4070 Laptop） |
| 主要依赖 | `opencv-python`、`torch`（CUDA 版）、`transformers`、`numpy` |

```bash
pip install opencv-python torch transformers numpy
```

> `torch` 要装 **CUDA 版**，纯 CPU 版跑不动深度模型（会慢到个位数 FPS）。

---

## 第一步：下载模型

⚠️ **模型不包含在本仓库里**（几百 MB，且可重新下载）。需要手动下载后放到**项目目录之外**的路径。

默认路径写在 `depth_demo.py` 顶部：

```python
DEPTH_MODEL_DIR  = r"D:\models\Depth-Anything-V2-Small"
DETECT_MODEL_DIR = r"D:\models\rtdetr_r18vd"
```

改成你自己的路径即可。

### 下载命令

国内直连 `huggingface.co` 通常不通，用镜像站 `hf-mirror.com`：

```bash
# Git Bash / Linux
hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir D:/models/Depth-Anything-V2-Small
hf download PekingU/rtdetr_r18vd --local-dir D:/models/rtdetr_r18vd
```

如果 `hf` 命令仍然连不上 huggingface.co，直接开浏览器从镜像站下载这几个文件，**文件名不要改**：

| 模型 | 必需文件 | 镜像站页面 |
|---|---|---|
| 深度 | `config.json`、`model.safetensors`、`preprocessor_config.json` | https://hf-mirror.com/depth-anything/Depth-Anything-V2-Small-hf |
| 检测 | 同上三个 | https://hf-mirror.com/PekingU/rtdetr_r18vd |

---

## 第二步：运行

### 深度估计 + 目标检测 demo（推荐先跑这个）

```bash
python -X utf8 depth_demo.py     # Windows 上加 -X utf8，中文输出不乱码
```

会弹出两个窗口：

- **`camera`** —— 原图 + 绿色检测框，左上角显示实时 FPS
- **`depth`** —— 深度伪彩色，**颜色越暖/越亮 = 越近**

按键：`q` 退出 · `n` 开关检测框 · `m` 开关镜像

**值得亲手试一下的对比实验**：拿个杯子放到镜头前**保持不动**，看 `depth` 窗口——杯子的轮廓会**一直**清晰可见。这是帧差法做不到的，也是换方案的原因。

### 摄像头预览工具（不需要模型，验证摄像头是否正常）

```bash
python -X utf8 camera-test.py
```

按键：`q` 退出 · `s` 截图（存到 `captures/`） · `g` 灰度 · `e` 边缘 · `r` 原图 · `m` 镜像

---

## 文件说明

```
camera-test.py            摄像头预览/调试工具（不依赖 GPU 和模型）
depth_demo.py             深度估计 + 目标检测的双窗口实时 demo
requirements.txt          依赖清单
docs/.../specs/           设计文档（方案选择、实测数据、已知限制、待决问题）
captures/                 截图输出目录（已 gitignore）
```

---

## 实测性能

本机（RTX 4070 Laptop，1280×720，fp16 半精度）：

| 项目 | 耗时 | 占比 |
|---|---|---|
| 深度估计 | 23.0 ms | 66%（**主要瓶颈**） |
| 目标检测 | ~9 ms（每 3 帧跑一次，摊薄后约 3 ms） | — |
| **综合** | **34.6 ms → 28.9 FPS** | |

优化过程：先做检测隔帧（13.0 → 15.8 FPS），再做 fp16 半精度（15.8 → **28.9 FPS**）。
fp16 的精度代价极小：归一化后逐像素平均差 0.00019。

> 测这类性能时**必须**在计时点调用 `torch.cuda.synchronize()`——CUDA 是异步的，不加会测出假数据。

---

## 已知限制

1. **输出的是「相对深度」，不是米数。** 它只说 A 比 B 近，不能直接判断「1 米内有障碍」。需要真实距离得换 Metric 版模型。
2. **检测层只认 COCO 80 类**，且远处/半身目标置信度天然偏低——这不是误检（实测确认），但也不能指望它发现没见过的障碍物。
3. **尚无判定逻辑**：目前只显示深度图和框，不做「有障碍 / 无障碍」判断。
4. **深度是瓶颈**（占 66%）。再提速需要降低模型输入分辨率，会真实损失精度。
5. 相机移动、强光、白墙、镜面/透明物体等场景**尚未实测**。

---

## 后续计划

1. 设计并实现避障判定逻辑（左/中/右分区 → 「有障碍」输出）
2. 评估是否换用 Metric 版深度模型以获得真实距离
3. 专项实测「静止障碍」和「相机移动」两项关键能力
4. （远期）装到小车上
