import torch
import torch.nn as nn


class Interactive_fusion_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_guide = nn.Linear(2048, 1024).cuda()  # 将 2048 维降到 1024 维
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8).cuda()
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2, enable_nested_tensor=False).cuda()
        self.classifier = nn.Linear(1024, 2).cuda()

    def forward(self, clip_fmp, guide_fmp):
        B = guide_fmp.shape[0]

        guide_fmp_flat = guide_fmp.view(B, 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]

        guide_fmp_aligned = self.linear_guide(guide_fmp_flat)  # [B, 49, 1024]

        combined_seq = torch.cat([clip_fmp, guide_fmp_aligned], dim=1)  # [B, 434, 1024]

        fused_seq = self.transformer(combined_seq)  # [B, 434, 1024]
        fused_seq_pooled = fused_seq.mean(dim=1)  # [B, 1024]
        logits = self.classifier(fused_seq_pooled)  # [B, 2]

        return logits