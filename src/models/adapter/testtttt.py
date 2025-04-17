import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from src.models.DFACLIP.DFACLIP import DFACLIP
from src.utils.visualize import  single_res_fmap_heatmap, single_clip_fmap_heatmap

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def visualize_mask(image, mask, alpha=0.5):
    """
    可视化图像和平均掩码的叠加结果。

    参数：
        image: torch.Tensor, 形状 [C, H, W]，输入图像
        mask: torch.Tensor, 形状 [num_channels, H, W]，掩码张量
        alpha: float, 叠加时的透明度
    """
    # 将图像转换为 numpy 格式
    image_np = image.permute(1, 2, 0).cpu().numpy() * 255
    image_np = image_np.astype(np.uint8)

    # 对掩码的通道维度取平均值
    mask_avg = mask.mean(dim=0)  # [H, W]，这里是 [14, 14]

    # 归一化掩码
    mask_avg = (mask_avg - mask_avg.min()) / (mask_avg.max() - mask_avg.min() + 1e-6)

    # 上采样到图像大小
    mask_resized = F.interpolate(mask_avg.unsqueeze(0).unsqueeze(0),
                                 size=image.shape[1:], mode='bilinear').squeeze()  # [H, W]
    mask_np = mask_resized.cpu().numpy()

    # 可视化
    plt.figure(figsize=(10, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 平均掩码
    plt.subplot(1, 3, 2)
    plt.title("Average Mask")
    plt.imshow(mask_np, cmap='jet')
    plt.axis('off')

    # 叠加图像
    heatmap = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()



# 测试
# image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v2\Celeb-synthesis\frames\id1_id3_0001\113.png'
image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v2\Celeb-real\frames\id0_0002\022.png'

image = cv2.imread(image_path)
image_mask = image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
image_tensor = to_tensor(image).cuda()
image_mask = to_tensor(image_mask).cuda()

# landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v2\Celeb-synthesis\landmarks\id1_id3_0001\113.npy'
landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v2\Celeb-real\landmarks\id0_0002\022.npy'

landmarks = np.load(landmarks_path)
landmarks_tensor = torch.from_numpy(landmarks).float()

# 缩放 landmarks
scale_factor = 224.0 / 255.0
landmarks_tensor = landmarks_tensor * scale_factor
landmarks_tensor = torch.clamp(landmarks_tensor, 0, 224) / 224.0

# 交换 x、y（如果需要）
# landmarks_tensor = landmarks_tensor[..., [1, 0]]  # 确保 [x, y] 格式

data_dict = {
    'image': image_tensor.unsqueeze(0),
    'landmark': landmarks_tensor.unsqueeze(0).cuda()
}
# 手动生成 mask
model = DFACLIP().cuda()
# model_weights_path = r"E:\github_code\Unnamed1\weights\best_DFACLIP_model.pth"
# model.load_state_dict(torch.load(model_weights_path, map_location='cuda'), strict=False)
# generator.load_state_dict(torch.load(r'E:\github_code\Unnamed1\weights\best_DFDACLIP_model_acc87.9703042715038.pth'))
# print(generator)
# mask = generator.LandmarkGuidedAdapter.mask_generator(landmarks_tensor.unsqueeze(0).cuda())  # [1, 4, 14, 14]
dict = model(data_dict)
mask, guide_preds, global_preds, guide_fmp, clip_fmp = dict['mask'], dict['guide_preds'], dict['global_preds'] ,dict['guide_fmp'], dict['clip_fmp']
# print("Mask sum:", mask.sum(dim=0))

# 只取平均值并可视化
visualize_mask(image_mask, mask[0])
single_res_fmap_heatmap(image, guide_preds, guide_fmp) # 练出来的model没效果

single_clip_fmap_heatmap(image, global_preds, clip_fmp)
print("Landmarks (first set):", landmarks_tensor[0])