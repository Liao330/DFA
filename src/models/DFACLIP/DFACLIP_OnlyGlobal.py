# src/models/DFACLIP/DFACLIP_OnlyGlobal.py
import torch
import torch.nn.functional as F
from src.models.DFACLIP.DFACLIP import DFACLIP

class DFACLIP_OnlyGlobal(DFACLIP):
    def __init__(self, mode='video'):
        super().__init__(mode=mode)
        # 禁用 FG 和 IFC
        self.facialguidance = None
        self.interactive_fusion_classifier = None
        # 冻结所有参数，仅用于推理
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, data_dict, inference=True):
        images = data_dict['image']
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # 提取 CLIP 特征并应用 GlobalContextAdapter
        clip_features = self.clip_model.extract_features(clip_images, self.GlobalContextAdapter.fusion_map.values())
        attn_biases = self.GlobalContextAdapter(data_dict, clip_features, inference)
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)
        clip_cls_output = self.clip_post_process(clip_output.float())

        # 仅返回 Global 的预测
        pred_dict = {
            'cls': clip_cls_output.float(),
            'global_preds': clip_cls_output.float(),
            'guide_preds': None,
            'fusion_preds': None,
            'mask': None,
            'clip_fmp': clip_fmp,
            'guide_fmp': None,
            'w_global': 1.0,  # 权重固定为 1
            'w_guide': 0.0,
            'w_fusion': 0.0
        }
        return pred_dict