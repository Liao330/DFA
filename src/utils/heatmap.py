import torch
import torch.nn as nn




class HeatmapGenerator(nn.Module):
    def __init__(self, heatmap_size=56, sigma=2):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def forward(self, landmarks):
        """
        Args:
            landmarks: [B, 81, 2], 批量关键点坐标（x, y），归一化到[0, 1]
        Returns:
            heatmaps: [B, 81, H, W], 多通道热图
        """
        B, N, _ = landmarks.shape  # B: batch_size, N: num_landmarks (81)
        H, W = self.heatmap_size, self.heatmap_size

        # 生成网格坐标
        y, x = torch.meshgrid(torch.arange(H, device=landmarks.device),
                              torch.arange(W, device=landmarks.device),
                              indexing='ij')
        grid = torch.stack([x, y], dim=-1).float()  # [H, W, 2]

        # 将landmarks缩放到heatmap_size
        landmarks = landmarks * torch.tensor([W, H], device=landmarks.device).float()  # [B, 81, 2]

        # 计算高斯热图
        heatmaps = []
        for b in range(B):  # 逐个样本处理（可优化为全批量）
            heatmap = torch.zeros(N, H, W, device=landmarks.device)
            for i, landmark in enumerate(landmarks[b]):
                x0, y0 = landmark[0], landmark[1]
                dist = ((grid - torch.tensor([x0, y0], device=landmarks.device)) ** 2).sum(dim=-1)  # [H, W]
                heatmap[i] = torch.exp(-dist / (2 * self.sigma ** 2))
            heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)  # [B, 81, H, W]

        # 归一化
        heatmaps = heatmaps / (heatmaps.max(dim=-1, keepdim=True)[0].max(dim=-1, keepdim=True)[0] + 1e-6)
        return heatmaps

# # 测试代码
# landmarks = torch.from_numpy(landmarks).float().unsqueeze(0)  # [1, 81, 2]
# generator = HeatmapGenerator(heatmap_size=56, sigma=2)
# heatmaps = generator(landmarks)
# print(heatmaps.shape)  # [1, 81, 56, 56]


