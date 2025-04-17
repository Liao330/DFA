import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt # 引入 matplotlib 用于更方便地显示 RGB 图像

# --- 1. 加载原始图片 ---
# image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\frames\id0_0002\022.png'
image_path = r'/dataset/processed/Celeb-DF-v2/Celeb-synthesis/frames/id1_id3_0001/113.png'

# 使用 OpenCV 加载图像，默认格式为 BGR
image_bgr = cv2.imread(image_path)

# 检查图像是否成功加载
if image_bgr is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# (可选) 将 BGR 图像转换为 RGB 图像，方便后续使用 matplotlib 显示
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- 2. 加载关键点数据 ---
# landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\landmarks\id0_0002\022.npy'
landmarks_path = r'/dataset/processed/Celeb-DF-v2/Celeb-synthesis/landmarks/id1_id3_0001/113.npy'

try:
    landmarks = np.load(landmarks_path) # 加载为 NumPy 数组
except FileNotFoundError:
    print(f"Error: Could not find landmarks file at {landmarks_path}")
    exit()

# --- 3. 在图像上绘制关键点 ---
# 我们将在 BGR 图像上绘制，因为 OpenCV 的绘图函数默认使用 BGR 颜色
# 如果你想在 RGB 图上绘制，请将 image_bgr 替换为 image_rgb 的副本
# image_to_draw_on = image_rgb.copy()
image_to_draw_on = image_bgr.copy() # 创建副本以避免修改原始BGR图像

# 定义绘制参数
point_radius = 2         # 圆点半径
point_color_bgr = (0, 255, 0) # 绿色 (BGR 格式)
# point_color_rgb = (0, 255, 0) # 绿色 (RGB 格式) - 如果在 RGB 图上绘制则使用这个
thickness = -1          # -1 表示填充圆点

# 遍历所有关键点
print(f"Loaded {landmarks.shape[0]} landmarks.")
for point in landmarks:
    # 确保坐标是整数类型，因为像素坐标是离散的
    x = int(point[0])
    y = int(point[1])

    # 在图像上绘制圆点
    cv2.circle(image_to_draw_on, (x, y), point_radius, point_color_bgr, thickness)
    # 如果在 RGB 图上绘制:
    # cv2.circle(image_to_draw_on, (x, y), point_radius, point_color_rgb, thickness)

# --- 4. 显示带有关键点的图像 ---

# # 方法一：使用 OpenCV 显示 (窗口会一直显示直到按键)
# # 注意：cv2.imshow 期望 BGR 格式的图像
# cv2.imshow('Image with Landmarks (OpenCV)', image_to_draw_on)
# print("Press any key in the OpenCV window to close it.")
# cv2.waitKey(0) # 等待用户按键
# cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口

# 方法二：使用 Matplotlib 显示 (更适合在 Jupyter Notebook 或脚本中显示 RGB 图像)
# 将绘制好关键点的 BGR 图像转换为 RGB 以便 Matplotlib 正确显示
image_to_show_mpl = cv2.cvtColor(image_to_draw_on, cv2.COLOR_BGR2RGB)
# 或者，如果你一开始就在 image_rgb 上绘制，则直接使用 image_to_draw_on

plt.figure(figsize=(8, 8)) # 设置显示尺寸
plt.imshow(image_to_show_mpl)
# 如果你没有绘制，而是想单独绘制点：
# plt.imshow(image_rgb) # 显示原始 RGB 图
# plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='lime') # s是点的大小, c是颜色
plt.title('Image with Landmarks (Matplotlib)')
plt.axis('off') # 关闭坐标轴
plt.show()
