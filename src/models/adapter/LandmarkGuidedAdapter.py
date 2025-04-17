import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LandmarkMaskGenerator(nn.Module):
    def __init__(self, mask_size=14, num_channels=10, radius=2):
        super().__init__()
        self.mask_size = mask_size
        self.num_channels = num_channels
        self.radius = radius
        self.groups = self._group_landmarks()

    def _group_landmarks(self):
        return [
            list(range(0, 17)),  # 脸部轮廓
            list(range(17, 22)),  # 左眉
            list(range(22, 27)),  # 右眉
            list(range(27, 36)),  # 鼻子
            list(range(36, 42)),  # 左眼
            list(range(42, 48)),  # 右眼
            list(range(48, 61)),  # 嘴外圈
            list(range(61, 68)),  # 嘴内圈
            list(range(68, 75)),  # 下巴
            list(range(75, 81))  # 其他
        ]

    def forward(self, landmarks):
        B, N, _ = landmarks.shape
        H, W = self.mask_size, self.mask_size
        mask = torch.zeros(B, self.num_channels, H, W, device=landmarks.device)

        y, x = torch.meshgrid(torch.arange(H, device=landmarks.device),
                              torch.arange(W, device=landmarks.device),
                              indexing='ij')
        grid = torch.stack([x, y], dim=-1).float()

        landmarks = landmarks * self.mask_size
        for c, group in enumerate(self.groups):
            group_landmarks = landmarks[:, group]
            group_landmarks = group_landmarks.unsqueeze(2).unsqueeze(2)
            grid_expanded = grid.unsqueeze(0).unsqueeze(1)
            dist = ((grid_expanded - group_landmarks) ** 2).sum(dim=-1)
            group_mask = torch.exp(-dist / (2 * (self.radius / 2) ** 2))
            mask[:, c] = group_mask.max(dim=1)[0]

        mask = mask / (mask.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
        return mask


class MaskWeightPredictor(nn.Module):
    def __init__(self, in_dim=768, num_channels=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_channels),
            nn.Softmax(dim=-1)
        ).float()

    def forward(self, features):
        return self.net(features)


class LandmarkGuidedAdapter(nn.Module):
    def __init__(self, dim=768, mask_size=14, num_channels=10, num_classes=2):
        super().__init__()
        self.mask_generator = LandmarkMaskGenerator(mask_size=mask_size, num_channels=num_channels)
        self.weight_predictor = MaskWeightPredictor(in_dim=dim, num_channels=num_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.visual_backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).cuda()
        self.extra_feature = nn.Sequential(*list(self.visual_backbone.children())[:-2])
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)

    def forward(self, clip_images, landmarks):
        batch_size, c, h, w = clip_images.shape
        clip_features = clip_images.float()
        fmap = self.extra_feature(clip_features)
        x = self.pool(fmap)
        x = x.view(batch_size, -1)
        outs = self.dp(self.linear1(x))
        landmarks = landmarks.float()
        B = clip_features.size(0)
        mask = self.mask_generator(landmarks)
        return fmap, outs, mask # [B, 2048, 7, 7] [B, 2], [B, 10, 14, 14]