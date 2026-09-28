# 避障识别（路线 A：区域 + 帧间差分）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `obstacle_detect.py`，用摄像头实时判断前方「左 / 中 / 右」哪一侧有东西在动——只做识别，不做控制。

**Architecture:** 单文件。核心是纯函数 `detect_zones()`（两张灰度图 → 各分区变化像素占比），不碰摄像头和窗口，因此可以脱离硬件做单元测试；`main()` 负责摄像头读取、防抖判定、画面可视化和按键。

**Tech Stack:** Python 3.12、OpenCV（`opencv-python` 4.12，已装）、NumPy（已装）。测试用标准库 `unittest`。

**设计文档:** `docs/superpowers/specs/2026-09-28-obstacle-detection-design.md`

## Global Constraints

- **零新增依赖**：只用标准库 + OpenCV + NumPy。测试走 `unittest`，**不装 pytest**（设计文档第 3 节：选路线 A 的理由之一就是零新增依赖）。
- **不修改 `camera-test.py`**（设计文档第 2 节），一个字都不动。
- **逐行中文注释**，面向入门读者（与 `camera-test.py` 风格一致）。
- **适用边界**：仅用于摄像头固定不动的场景（设计文档第 4 节）。
- **与设计文档的一处已知偏差**：设计文档第 5 节把 `min_ratio` 写在 `detect_zones()` 签名里，但同一节又规定它返回「占比列表」。两者矛盾。本计划按「返回占比」为准，`min_ratio` 的判定移到 `main()`——理由是判定与防抖都属于决策层，`detect_zones()` 保持纯计算、便于测试。设计文档第 5 节需同步改这一行。

---

### Task 1: 核心识别逻辑 `detect_zones()` 与单元测试

**Files:**
- Create: `obstacle_detect.py`（本任务只建出「常量区 + detect_zones」，`main()` 留到 Task 2）
- Create: `tests/test_detect.py`

**Interfaces:**
- Consumes: 无（全新文件）
- Produces:
  - 常量：`ZONES: int`、`DIFF_THRESHOLD: int`、`BLUR_KERNEL: int`、`MIN_PIXELS_RATIO: float`、`CONFIRM_FRAMES: int`、`GLOBAL_CHANGE_RATIO: float`
  - `detect_zones(prev_gray, gray, diff_threshold=DIFF_THRESHOLD) -> (list[float], np.ndarray)`
    - `prev_gray` / `gray`：`uint8` 单通道灰度图，形状 `(h, w)`
    - 返回 `ratios`：长度 `ZONES` 的浮点列表，每项是「第 i 区白像素占比」，取值 0~1
    - 返回 `mask`：`uint8` 单通道二值图，255 = 检测到变化

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_detect.py`：

```python
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
```

- [ ] **Step 2: 跑测试，确认它失败**

Run:
```bash
cd "D:\python_project\camera" && python -m unittest discover -s tests -v
```
Expected: FAIL，报 `ModuleNotFoundError: No module named 'obstacle_detect'`（因为实现文件还没建）。

- [ ] **Step 3: 写出让测试通过的最小实现**

创建 `obstacle_detect.py`：

```python
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
```

- [ ] **Step 4: 跑测试，确认全部通过**

Run:
```bash
cd "D:\python_project\camera" && python -m unittest discover -s tests -v
```
Expected: `Ran 6 tests` … `OK`

如果 `test_tiny_noise_stays_below_threshold` 意外失败（占比 ≥ 0.02），说明模糊后的残留面积偏大，
把 `BLUR_KERNEL` 由 `5` 调成 `7` 再跑一次——不要改测试的断言。

- [ ] **Step 5: 提交**

```bash
cd "D:\python_project\camera"
git add obstacle_detect.py tests/test_detect.py
git commit -m "feat: 新增 detect_zones 核心识别逻辑与单元测试"
```

---

### Task 2: 摄像头主循环、可视化与按键

**Files:**
- Modify: `obstacle_detect.py`（在 `detect_zones()` 之后追加 `open_camera()` 与 `main()`，并补上 `if __name__ == "__main__":`）

**Interfaces:**
- Consumes: Task 1 的 `detect_zones()` 及全部常量
- Produces: `open_camera(index=0) -> cv2.VideoCapture`；`main()`；脚本可直接 `python obstacle_detect.py` 运行

- [ ] **Step 1: 追加实现代码**

在 `obstacle_detect.py` 末尾追加：

```python


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

    print("按键说明：q 退出 / + - 调变化阈值 / d 调试图 / m 镜像")
    print(f"分区数：{ZONES}    变化阈值：{DIFF_THRESHOLD}    连续 {CONFIRM_FRAMES} 帧确认")

    mirror = True                  # 自拍镜像，看着更自然
    show_debug = False             # 是否显示二值调试图
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
                if ratios[i] > MIN_PIXELS_RATIO:
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
            f"threshold:{diff_threshold}  zones:{ZONES}  debug:{int(show_debug)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

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
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法与单元测试都要过**

Run:
```bash
cd "D:\python_project\camera" && python -m unittest discover -s tests -v
```
Expected: `Ran 6 tests` … `OK`（追加 `main()` 不应影响任何测试）

再跑一次语法检查：
```bash
cd "D:\python_project\camera" && python -c "import ast; ast.parse(open('obstacle_detect.py', encoding='utf-8').read()); print('语法 OK')"
```
Expected: `语法 OK`

- [ ] **Step 3: 实机运行（需要人看着画面）**

Run:
```bash
cd "D:\python_project\camera" && python obstacle_detect.py
```
Expected: 打印实际分辨率；窗口标题为 `obstacle`；左上角显示 `threshold:25 zones:3 debug:0`。

- [ ] **Step 4: 提交**

```bash
cd "D:\python_project\camera"
git add obstacle_detect.py
git commit -m "feat: 加入摄像头主循环、分区状态可视化与按键调参"
```

---

### Task 3: 实机验收与交付

**Files:**
- 无代码改动（除非验收发现问题）

**Interfaces:**
- Consumes: Task 2 的可运行脚本
- Produces: 验收结论；推送到 GitHub 的最终提交

- [ ] **Step 1: 跑设计文档第 9 节的四条验收**

运行 `python obstacle_detect.py`，逐条核对：

1. 手在镜头前**从左向右**移动 → 终端依次打印「左区：有障碍 → 中区：有障碍 → 右区：有障碍」；
2. 画面完全静止（人不动、别乱晃）→ 一直是「无障碍」，终端不刷屏；
3. 反复**开关房间灯** → 不应打印「有障碍」（最多黄条闪一下，不该变红、不该打印）；
4. 按 `d` 切到调试图 → 能看到手的位置是白色，其余是黑色。

任一条不通过时：按 `+` / `-` 调 `diff_threshold`，或用 `d` 看调试图判断是阈值问题还是算法问题，
**不要直接改代码**；确认是哪一类问题后再回到 Task 2 改。

- [ ] **Step 2: 记录验收结果**

把四条的实际结果（通过 / 不通过 + 现象）写下来，不通过的要写清具体现象。
「阈值需要现场标定」是设计文档第 11 节已预期的，不算缺陷。

- [ ] **Step 3: 推送**

```bash
cd "D:\python_project\camera"
git push
```
Expected: 推送到 `https://github.com/ZYlow1/camera` 的 `main` 分支。

- [ ] **Step 4: 同步设计文档的那处偏差**

修改 `docs/superpowers/specs/2026-09-28-obstacle-detection-design.md` 第 5 节，
把 `detect_zones` 的签名行改为：

```
detect_zones(prev_gray, gray, diff_threshold=DIFF_THRESHOLD) -> (ratios, mask)
    纯函数，不碰摄像头、不碰窗口。
    输入：上一帧灰度图、当前帧灰度图
    输出：每个分区的变化像素占比列表、以及用于显示的二值图
    —— 识别逻辑集中在这里，将来替换算法（MOG2 / 深度估计）只需改这一个函数。
    （判定用阈值 MIN_PIXELS_RATIO 与防抖 CONFIRM_FRAMES 放在 main()，
      让本函数保持纯粹、便于测试。）
```

然后：
```bash
cd "D:\python_project\camera"
git add docs/superpowers/specs/2026-09-28-obstacle-detection-design.md
git commit -m "docs: 同步 detect_zones 签名（判定阈值移出纯函数）"
git push
```

---

## 自检记录

**1. 设计文档覆盖检查**

| 设计文档章节 | 对应任务 |
|---|---|
| 第 5 节 架构（常量区 / `open_camera` / `detect_zones` / `main`） | Task 1、Task 2 |
| 第 6 节 数据流（含全局变化判定、防抖、仅状态变化时打印） | Task 2 Step 1 |
| 第 7 节 可视化与按键（`q` / `+` `-` / `d` / `m`） | Task 2 Step 1 |
| 第 8 节 错误处理（打不开、读不到画面、首帧跳过） | Task 2 Step 1 |
| 第 9 节 验证方法（四条验收 + 纯函数可测） | Task 1 Step 1、Task 3 Step 1 |
| 第 11 节 未决问题（阈值现场标定） | Task 3 Step 1、Step 2 |
| 第 2 节 不改 `camera-test.py` | 全程未涉及 |

**2. 占位符扫描**：无 TBD / TODO；每个代码步骤都给了完整代码。

**3. 类型一致性检查**：`detect_zones` 返回 `(ratios, mask)`，Task 2 的 `main()` 中按
`ratios, mask = detect_zones(...)` 解包，一致；常量名 `ZONES` / `DIFF_THRESHOLD` /
`MIN_PIXELS_RATIO` / `CONFIRM_FRAMES` / `GLOBAL_CHANGE_RATIO` / `BLUR_KERNEL` / `MORPH_KERNEL`
在 Task 1 定义、Task 2 与测试中引用，拼写一致。
