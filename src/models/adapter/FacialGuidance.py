import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CrossAttentionModule(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        """
        x: [B, L, D] - Image features
        context: [B, M, D] - Landmark-guided features
        """
        # Self-attention + FFN
        residual = x
        x = self.norm1(x)

        # Cross attention
        B, L, D = x.shape
        q = self.query(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, D/H]
        k = self.key(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D/H]
        v = self.value(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D/H]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, M]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, H, L, D/H]
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)  # [B, L, D]
        out = self.proj(out)

        x = residual + out

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=768, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Simplified encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4, batch_first=True)
            for _ in range(6)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]

        # Create patches
        x = self.patch_embed(x)  # [B, C, H/p, W/p]
        h, w = x.shape[2:4]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W/p^2, C]

        # Add position embeddings
        x = x + self.pos_embed[:, :(h * w), :]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        return x


class FacialGuidance(nn.Module):
    def __init__(self, dim=2048, mask_size=7, num_channels=10, num_classes=2):
        super().__init__()
        self.mask_generator = LandmarkMaskGenerator(mask_size=mask_size, num_channels=num_channels)
        self.weight_predictor = MaskWeightPredictor(in_dim=dim, num_channels=num_channels)

        # Feature encoder instead of ResNext
        self.feature_encoder = FeatureEncoder(in_channels=3, dim=dim)

        # Cross-attention modules between image features and landmark-guided features
        self.cross_attention1 = CrossAttentionModule(dim=dim)
        self.cross_attention2 = CrossAttentionModule(dim=dim)

        # Final classification
        self.norm = nn.LayerNorm(dim)
        self.dp = nn.Dropout(0.4)
        self.classifier = nn.Linear(dim, num_classes)

        # Feature projection for mask features
        self.mask_proj = nn.Sequential(
            nn.Conv2d(num_channels, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )

    def forward(self, images, landmarks):
        # Generate landmark masks
        landmarks = landmarks.float()
        masks = self.mask_generator(landmarks)  # [B, num_channels, mask_size, mask_size]

        # Extract image features
        batch_size = images.shape[0]
        image_features = self.feature_encoder(images.float())  # [B, L, dim]
        cls_token = image_features[:, 0:1]  # [B, 1, dim]
        patch_features = image_features[:, 1:]  # [B, L-1, dim]

        # Get global feature for weight prediction
        global_feature = cls_token.squeeze(1)  # [B, dim]
        mask_weights = self.weight_predictor(global_feature)  # [B, num_channels]

        # Apply weights to masks
        weighted_masks = masks * mask_weights.view(batch_size, -1, 1, 1)  # [B, num_channels, mask_size, mask_size]

        # Project weighted masks to same dimension as image features
        mask_features = self.mask_proj(weighted_masks)  # [B, dim, mask_size, mask_size]

        # Reshape mask features for cross-attention
        mask_size = masks.shape[-1]
        mask_features = mask_features.flatten(2).transpose(1, 2)  # [B, mask_size*mask_size, dim]

        # Apply cross-attention between image features and mask features
        guided_features = self.cross_attention1(patch_features, mask_features)
        guided_features = self.cross_attention2(guided_features, mask_features)

        # Global pooling and classification
        fused_features = torch.cat([cls_token, guided_features], dim=1)
        pooled_features = fused_features.mean(dim=1)  # [B, dim]
        pooled_features = self.norm(pooled_features)
        logits = self.classifier(self.dp(pooled_features))

        # Reshape guided features to create a feature map similar to the original output
        feature_map_size = int((guided_features.shape[1]) ** 0.5)
        fmap = guided_features.transpose(1, 2).reshape(batch_size, -1, feature_map_size, feature_map_size)
        # print(f"famp:{fmap.shape}") [B, 2048, 14, 14]
        return fmap, logits, masks  # [B, dim, size, size], [B, num_classes], [B, num_channels, mask_size, mask_size]