import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DFACLIP.DFACLIP import DFACLIP


class DFACLIP_NoGlobal(DFACLIP):
    def __init__(self, mode='video'):
        super().__init__(mode=mode)
        # 冻结所有参数，仅用于推理
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, data_dict, inference=True):
        images = data_dict['image']
        landmarks = data_dict['landmark']
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # 提取 CLIP 特征，但跳过 GlobalContextAdapter
        clip_features = self.clip_model.extract_features(clip_images, self.GlobalContextAdapter.fusion_map.values())
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, None, inference, normalize=True) # attn_biases is None

        # 正常执行 FacialGuidance 和 IFC
        guide_fmp, guide_preds, mask = self.facialguidance(clip_images, landmarks)
        fusion_preds = self.interactive_fusion_classifier(clip_fmp, guide_fmp)
        clip_cls_output = self.clip_post_process(clip_output.float())

        # 计算权重
        weights = torch.softmax(torch.stack([self.w_global, self.w_guide, self.w_fusion]), dim=0)

        # 返回预测字典
        pred_dict = {
            'cls': fusion_preds.float(),
            'global_preds': clip_cls_output.float(),
            'guide_preds': guide_preds.float(),
            'fusion_preds': fusion_preds.float(),
            'mask': mask,
            'clip_fmp': clip_fmp,
            'guide_fmp': guide_fmp,
            'w_global': weights[0],
            'w_guide': weights[1],
            'w_fusion': weights[2]
        }
        return pred_dict

# 示例用法
# model = DFACLIP_NoGlobal(mode='video')
# model.load_state_dict(torch.load('pretrained_dfaclip.pth'))  # 加载预训练权重
# model.eval()