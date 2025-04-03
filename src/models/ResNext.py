# import os
#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from matplotlib import pyplot as plt
# from torch import nn
# from torchvision import models
# from torchvision.transforms import transforms
#
# from src.config import CRITERION, NUM_CLASS
#
#
# # 类名与文件名保持一致
# class ResNext(nn.Module):
#     def __init__(self, hidden_dim=2048):
#         super(ResNext, self).__init__()
#         # model = models.convnext_large(pretrained=True)
#         num_classes = NUM_CLASS
#         model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
#         model = model.cuda()
#         self.model = nn.Sequential(*list(model.children())[:-2])
#         # self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
#         self.relu = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.4)
#         self.linear1 = nn.Linear(hidden_dim, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         batch_size, c, h, w = x.shape
#         #print("111",x.shape) #[64,  3, 224, 224]
#         fmap = self.model(x)
#         # print("333", fmap.shape) # [64, 2048, 7, 7]
#         x = self.avgpool(fmap)
#         # print("444", x.shape) # [64, 2048, 1, 1]
#         x = x.view(batch_size,  -1)
#         # print("555", x.shape) # [64, 2048]
#         # x_lstm, _ = self.lstm(x, None)
#         # print("666", x_lstm.shape) #[4, 60, 1536]
#         outs = self.dp(self.linear1(x)) # [64, 2]
#         return outs
#
# # 设置环境变量
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
# # 加载模型
# model = ResNext().cuda()
# model.load_state_dict(torch.load('weights/best_ResNext_model_acc98.7386385848319.pth'))
# model.eval()  # 设置为评估模式
#
# # 加载图像
# image_path = r'E:\github_code\Unnamed1\dataset\processed\FaceForensics++\manipulated_sequences\DeepFakeDetection\c23\frames\13_25__walking_down_street_outside_angry__IW9OFVMG\310.png'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # 检查图像
# # plt.imshow(image)
# # plt.title('Input Image')
# # plt.show()
#
# # Convert NumPy array to PIL Image
# image_pil = Image.fromarray(image)
#
# # 转换为张量并归一化
# to_tensor = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# image_tensor = to_tensor(image_pil).unsqueeze(0).cuda()
#
# # 获取特征图
# with torch.no_grad():
#     fmap = model.model(image_tensor)
# print(fmap.shape)  # [1, 2048, 7, 7]
#
# # 检查原始值范围
# # fmap_raw = fmap[0, 0].cpu().detach().numpy()
# # 平均值聚合
# fmap_raw = fmap[0].mean(dim=0).cpu().detach().numpy()  # [7, 7]
# # 上采样特征图
# fmap_tensor = torch.from_numpy(fmap_raw).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
# fmap_up = torch.nn.functional.interpolate(fmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)  # [1, 1, 224, 224]
# fmap_up = fmap_up.squeeze().numpy()  # [224, 224]
#
# # 归一化
# if fmap_up.max() > fmap_up.min():
#     fmap_norm = (fmap_up - fmap_up.min()) / (fmap_up.max() - fmap_up.min() + 1e-8)
# else:
#     fmap_norm = fmap_up
#
# # 转换为热图
# heatmap = cv2.applyColorMap((fmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
#
# # 叠加到原始图像
# image_rgb = cv2.resize(image, (224, 224))  # 确保尺寸一致
# overlay = cv2.addWeighted(image_rgb, 0.5, heatmap, 0.5, 0)
#
# # 可视化
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(image_rgb)
# plt.title('Original Image')
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.imshow(fmap_norm, cmap='viridis')
# plt.title('Feature Map')
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.imshow(overlay)
# plt.title('Overlay')
# plt.axis('off')
# plt.show()

import os
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
import torch.nn.functional as F

# from src.utils.visualize import single_fmap_heatmap


# 类名与文件名保持一致
class ResNext(nn.Module):
    def __init__(self, hidden_dim=2048):
        super(ResNext, self).__init__()
        num_classes = 2  # Adjust based on your NUM_CLASS
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model = model.cuda()
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, -1)
        outs = self.dp(self.linear1(x))
        # return outs, fmap  # 返回 logits 和特征图
        return outs

# # 设置环境变量
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
# # 加载模型
# model = ResNext().cuda()
# model.load_state_dict(torch.load('weights/best_ResNext_model_acc98.7386385848319.pth'))
# model.eval()  # 设置为评估模式
#
# # 加载图像
# image_path = r'E:\github_code\Unnamed1\dataset\processed\FaceForensics++\manipulated_sequences\Face2Face\c23\frames\103_082\203.png'
# # image_path = r'E:\github_code\Unnamed1\dataset\processed\FaceForensics++\original_sequences\youtube\c23\frames\001\029.png'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Convert NumPy array to PIL Image
# image_pil = Image.fromarray(image)
#
# # 转换为张量并归一化
# to_tensor = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# image_tensor = to_tensor(image_pil).unsqueeze(0).cuda()
#
# # 获取模型输出和特征图
# model.eval()
# image_tensor.requires_grad_(True)  # 启用梯度计算
# logits, fmap = model(image_tensor)  # [1, 2], [1, 2048, 7, 7]
# single_fmap_heatmap(image, logits, fmap)