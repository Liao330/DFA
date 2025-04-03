# import numpy as np
# import cv2
# import torch
# import seaborn as sns
#
# # from matplotlib import pyplot as plt
# from torchvision.transforms import transforms
# # from src.config import MODEL_DOWNLOAD_ROOT, DEVICE
# # from src.models.DFACLIP.DFACLIP import DFACLIP
# # from src.models.DFACLIP.clip import load
#
#
# def create_heatmap(landmarks, heatmap_size=56, sigma=2):
#     """
#     创建热图。
#
#     Args:
#         landmarks (np.ndarray): 关键点坐标，形状为 (num_landmarks, 2)。
#         heatmap_size (int): 热图的大小。
#         sigma (int): 高斯分布的标准差。
#
#     Returns:
#         np.ndarray: 热图，形状为 (heatmap_size, heatmap_size)。
#     """
#     num_landmarks = landmarks.shape[0]
#     heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
#
#     for landmark in landmarks:
#         x, y = landmark
#         # 将关键点坐标归一化到热图大小
#         x = int(x / image.shape[1] * heatmap_size)
#         y = int(y / image.shape[0] * heatmap_size)
#
#         # 生成高斯分布
#         for i in range(heatmap_size):
#             for j in range(heatmap_size):
#                 heatmap[i, j] += np.exp(-((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2))
#
#     # 归一化热图
#     heatmap = heatmap / heatmap.max()
#     # # 显示热图
#     # cv2.imshow('Heatmap', heatmap)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return heatmap
#
# def overlay_heatmap(image, heatmap, alpha=0.5):
#     """
#     将热图叠加到原始图像上。
#
#     Args:
#         image (np.ndarray): 原始图像。
#         heatmap (np.ndarray): 热图。
#         alpha (float): 透明度。
#
#     Returns:
#         np.ndarray: 叠加后的图像。
#     """
#     # 将热图归一化到 [0, 255]
#     heatmap = (heatmap * 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#     # 将热图叠加到原始图像上
#     heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
#     overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
#     return overlayed_image
#
# def show_landmark(image, landmarks):
#     # 将关键点绘制到图像上
#     for landmark in landmarks:
#         x, y = landmark
#         cv2.circle(image, (int(x * image.shape[1] / 256), int(y * image.shape[0] / 256)), radius=2, color=(0, 255, 0),
#                    thickness=-1)  # 绿色圆点
#
#     # 显示图像
#     cv2.imshow('Image with Landmarks', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # 加载原始图片
# image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\frames\id0_0002\022.png'
# image = cv2.imread(image_path)
# # 将 BGR 图像转换为 RGB 图像
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 转换为张量
# to_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224))
# ])
# image_tensor = to_tensor(image).cuda()
# # print(image_tensor.shape) # [3, 224, 224]
#
# # 加载关键点数据
# landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\landmarks\id0_0002\022.npy'
# landmarks = np.load(landmarks_path)
# landmarks_tensor = torch.from_numpy(landmarks).type(torch.float)
# # print(landmarks)
# # print(landmarks_tensor.shape)  # [81, 2]
#
# show_landmark(image, landmarks)

# 创建热图
# heatmap = create_heatmap(landmarks, heatmap_size=56, sigma=2)
# print(heatmap.shape)  # (56, 56)

# 将热图叠加到原始图像上
# overlayed_image = overlay_heatmap(image, heatmap, alpha=0.5)
#
# # clip_name = "ViT-L/14"
# # model = DFACLIP().cuda()
# # # clip_images = image_tensor.unsqueeze(0)  # 添加批次维度，形状变为 [1, 3, 224, 224]
# # clip_images = to_tensor(overlayed_image).unsqueeze(0).cuda()  # 添加批次维度，形状变为 [1, 3, 224, 224]
# # clip_model, processor = load(clip_name, DEVICE, download_root=MODEL_DOWNLOAD_ROOT)
# # # clip_features = clip_model.encode_image(clip_images)
# # # 提取图像特征
# # clip_features = clip_model.encode_image(clip_images).detach().cpu().numpy()
#
# # 可视化热图
# # plt.figure(figsize=(12, 8))
# # sns.heatmap(clip_features, cmap='viridis', xticklabels=False, yticklabels=False)
# # plt.title('CLIP Features Heatmap')
# # plt.show()
# # clip_features = clip_model.extract_features(clip_images)
# # print(clip_features.shape) # [1, 768]
#
#
# # 显示叠加后的图像
# cv2.imshow('Image with Heatmap', overlayed_image)
# cv2.waitKey(10000) # 10s后自动关闭
# cv2.destroyAllWindows()


import cv2
import numpy as np
import torch
from torchvision import transforms


def create_heatmap(landmarks, heatmap_size=56, sigma=2):
    num_landmarks = landmarks.shape[0]
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

    for landmark in landmarks:
        x, y = landmark
        x = int(x / image.shape[1] * heatmap_size)
        y = int(y / image.shape[0] * heatmap_size)

        # 生成高斯分布
        for i in range(heatmap_size):
            for j in range(heatmap_size):
                heatmap[i, j] += np.exp(-((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2))

    # 归一化热图
    heatmap = heatmap / np.nanmax(heatmap)  # 使用 np.nanmax 替换 np.max 避免除以零的情况
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap


def draw_landmark_masks(image, landmarks, heatmap_size=56):
    # 创建一个和原始图像同样大小的黑色图像
    mask_image = np.zeros_like(image)
    # 定义每个面部特征区域的关键点索引
    parts = {
        'Brow': slice(17, 21),  # 假设这些索引是眉毛的关键点索引
        'Eyes': slice(22, 36),
        'Nose': slice(36, 41),
        'Mouth': slice(42, 68)
    }

    for part_name, idx_slice in parts.items():
        heatmap = create_heatmap(landmarks[idx_slice], heatmap_size, sigma=2)
        y, x = np.where(heatmap > 0)
        for yy, xx in zip(y, x):
            mask_image[yy, xx] = (0, 255, 0)  # 绿色

    return mask_image


# 加载原始图片
image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\frames\id0_0002\022.png'
image = cv2.imread(image_path)
# 将 BGR 图像转换为 RGB 图像
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为张量
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
image_tensor = to_tensor(image).cuda()
# print(image_tensor.shape) # [3, 224, 224]

# 加载关键点数据
landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\landmarks\id0_0002\022.npy'
landmarks = np.load(landmarks_path)
landmarks_tensor = torch.from_numpy(landmarks).type(torch.float)
# print(landmarks)
# print(landmarks_tensor.shape)  # [81, 2]


# 生成掩模
landmark_masks = draw_landmark_masks(image, landmarks)

# 显示结果
cv2.imshow('Landmark Masks', landmark_masks)
cv2.waitKey(0)
cv2.destroyAllWindows()