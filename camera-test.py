# -*- coding: utf-8 -*-
"""
摄像头小工具（入门练习版）

作用：打开电脑摄像头，实时显示画面，并且能切换几种显示效果、随手截图。

运行前先装好 OpenCV：
    pip install opencv-python

然后在本文件所在目录执行：
    python camera-test.py

窗口里按这些键操作：
    q  退出
    s  截图（存到 captures/ 文件夹）
    g  灰度模式（黑白）
    e  边缘模式（只显示轮廓）
    r  原图模式
    m  开关镜像（左右翻转）
"""

# ---- 第一步：导入要用的库 ----
import cv2                      # OpenCV，用来操作摄像头和图像
import os                       # 操作系统相关工具，这里用来看系统类型、建文件夹
import time                     # 用来算帧率（FPS）

# ---- 可调参数（想改就改这里）----
# 分辨率：1280x720（俗称 720p）比 640x480 清晰很多，也是这个摄像头能拿到的最高一档。
# 如果摄像头不支持，程序会自动退回它自己支持的尺寸（下面会打印实际生效的分辨率）。
WIDTH, HEIGHT = 1280, 720

# ---- 第二步：打开摄像头 ----
# VideoCapture(0) 里的 0 表示"第 0 个摄像头"，一般就是笔记本自带的那个。
# 如果你的电脑有多个摄像头，可以试试改成 1、2。
if os.name == "nt":
    # os.name == "nt" 表示当前是 Windows 系统。
    # Windows 上有两种常见的摄像头驱动方式（OpenCV 叫它 backend）：
    #   CAP_MSMF  —— Media Foundation。实测这个摄像头能在 1280x720 下跑到 30 帧，所以优先用它。
    #   CAP_DSHOW —— DirectShow。兼容性最好，但同样的 720p 只能跑 10 帧。
    # 所以这里先用 MSMF，万一你的摄像头不支持，再退回 DSHOW。（MSMF 首次打开可能要等 1~2 秒）
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    # Mac / Linux 等系统，用默认方式打开即可。
    cap = cv2.VideoCapture(0)

# isOpened() 返回 True 表示摄像头成功打开了。
# 万一第一次失败（比如驱动方式不合适），这里再用默认方式重试一次。
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# 如果重试之后还是打不开，就没法继续了，直接退出并给出提示。
if not cap.isOpened():
    print("摄像头打不开。检查是否被其他软件占用，或试试 VideoCapture(1)")
    raise SystemExit

# ---- 第三步：设置画面参数 ----
# CAP_PROP_FRAME_WIDTH / HEIGHT 用来设定画面宽高（单位：像素）。
# 注意：这只是"请求"，不是所有摄像头都支持，实际尺寸要以取到的画面为准。
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# 摄像头不一定会接受我们请求的尺寸（比如上面写的 1920x1080 就常常被降级），
# 所以把实际生效的值读回来打印一下，心里有数。
real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"实际分辨率：{real_w}x{real_h}（请求的是 {WIDTH}x{HEIGHT}）")

# 创建存放截图的文件夹 captures（已经存在就跳过，exist_ok=True 保证不报错）。
os.makedirs("captures", exist_ok=True)

# ---- 第四步：准备几个状态变量 ----
mode = "raw"       # 当前显示模式：raw(原图) / gray(灰度) / edge(边缘)
mirror = True      # 是否镜像显示。True 就像照镜子一样左右反过来，自拍看着更自然
count = 0          # 已经截了多少张图，用来给截图文件编号
prev = time.time() # 记录上一帧的时间，用来计算帧率（FPS）

# 先把操作说明打印到终端，方便照着按。
print("按键说明：")
print("q 退出")
print("s 截图")
print("g 灰度")
print("e 边缘")
print("r 原图")
print("m 开关镜像")

# ---- 第五步：主循环，不停地读画面、显示画面 ----
while True:
    # read() 从摄像头取一帧画面：
    #   ret  —— 是否读取成功（True / False）
    #   frame —— 这一帧的图像数据（OpenCV 里是 BGR 三通道彩色图）
    ret, frame = cap.read()
    if not ret:
        # 读不到画面（比如摄像头被拔了），就退出循环。
        print("读不到画面")
        break

    # 自带摄像头通常需要镜像，像照镜子一样。
    # flip 的第二参数 1 表示"左右翻转"（0 是上下翻转）。
    if mirror:
        frame = cv2.flip(frame, 1)

    # 计算帧率：1 秒 ÷ 两帧之间的时间差 = 每秒显示多少帧。
    # 加 if now > prev 是为了防止时间差为 0（那样会除零报错）。
    now = time.time()
    fps = 1 / (now - prev) if now > prev else 0
    prev = now

    # 根据当前模式，生成一张"要显示给别人看"的图，统一叫 display。
    if mode == "gray":
        # 灰度：先用 cvtColor 把 BGR 彩色图转成单通道灰度图，
        # 因为 imshow 显示彩色更省事，再转回 BGR（此时三个通道值相同，看起来就是灰的）。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "edge":
        # 边缘：先转灰度，再用 Canny 找轮廓。
        # Canny 的两个数字是阈值，越大保留的边缘越少、越"干净"。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    else:
        # 原图：直接复制一份。用 copy() 是为了不破坏原来的 frame
        # （截图时我们还想保存干净的原始画面）。
        display = frame.copy()

    # shape 按 (高, 宽, 通道数) 排列，这里只取前两位得到 h 和 w。
    h, w = display.shape[:2]

    # 画两条竖线，把画面大致分成左/中/右三等份。
    # 纯粹是给眼睛看的参考线，不是在做识别。
    # line(图, 起点坐标, 终点坐标, 颜色(B,G,R), 线宽)
    # 颜色 (0, 255, 0) 表示绿色。
    cv2.line(display, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
    cv2.line(display, (w * 2 // 3, 0), (w * 2 // 3, h), (0, 255, 0), 1)

    # 在左上角写上一行状态文字：帧率、当前模式、是否镜像。
    # putText(图, 文字, 左下角起点坐标, 字体, 字号, 颜色, 线宽)
    # 颜色 (0, 0, 255) 是红色（注意 OpenCV 的顺序是 蓝、绿、红）。
    cv2.putText(
        display,
        f"FPS {fps:.1f}  mode:{mode}  mirror:{mirror}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # 把 display 显示在名叫 "camera" 的窗口里。
    cv2.imshow("camera", display)

    # waitKey(1) 表示"等键盘输入最多 1 毫秒"。
    # 它有两个作用：① 让画面能够刷新出来，② 读取你按下的键。
    # & 0xFF 是为了只保留按键的低 8 位（不同系统返回值格式有差异，这样更保险）。
    key = cv2.waitKey(1) & 0xFF

    # 根据按下的键做不同的事。ord('q') 就是把字母 q 转成对应的按键编码。
    if key == ord('q'):
        break                       # 退出循环，结束程序
    elif key == ord('s'):
        # 截图：用 f-string 拼出文件名，:04d 表示编号固定 4 位数（0000、0001……）。
        # 注意存的是 frame（可能已镜像的原图），不带头顶文字和竖线。
        path = f"captures/frame_{count:04d}.jpg"
        cv2.imwrite(path, frame)
        print("已保存:", path)
        count += 1                  # 编号加一
    elif key == ord('g'):
        mode = "gray"               # 切到灰度
    elif key == ord('e'):
        mode = "edge"               # 切到边缘
    elif key == ord('r'):
        mode = "raw"                # 切回原图
    elif key == ord('m'):
        mirror = not mirror         # 在 True / False 之间来回切换

# ---- 第六步：收尾清理 ----
cap.release()               # 释放摄像头，让别的软件可以再用它
cv2.destroyAllWindows()     # 关闭所有 OpenCV 打开的窗口
