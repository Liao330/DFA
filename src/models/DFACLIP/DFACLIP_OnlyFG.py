# src/models/DFACLIP/DFACLIP_OnlyFG.py
import torch
import torch.nn.functional as F
from src.models.DFACLIP.DFACLIP import DFACLIP

class DFACLIP_OnlyFG(DFACLIP):
    def __init__(self, mode='video'):
        super().__init__(mode=mode)
        # 禁用 Global 和 IFC
        self.rec_attn_clip = None
        self.GlobalContextAdapter = None
        self.interactive_fusion_classifier = None
        # 冻结所有参数，仅用于推理
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, data_dict, inference=True):
        images = data_dict['image']
        landmarks = data_dict['landmark']
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # 只执行 FacialGuidance
        guide_fmp, guide_preds, mask = self.facialguidance(clip_images, landmarks)

        # 仅返回 FG 的预测
        pred_dict = {
            'cls': guide_preds.float(),
            'global_preds': None,
            'guide_preds': guide_preds.float(),
            'fusion_preds': None,
            'mask': mask,
            'clip_fmp': None,
            'guide_fmp': guide_fmp,
            'w_global': 0.0,
            'w_guide': 1.0,  # 权重固定为 1
            'w_fusion': 0.0
        }
        return pred_dict