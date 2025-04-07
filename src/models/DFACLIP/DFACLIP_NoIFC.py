from src.models.DFACLIP.DFACLIP import DFACLIP
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFACLIP_NoIFC(DFACLIP):
    def __init__(self, mode='video'):
        super().__init__(mode=mode)
        # 冻结所有参数，仅用于推理
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, data_dict, inference=True):
        images = data_dict['image']
        landmarks = data_dict['landmark']
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # 正常执行 GlobalContextAdapter
        clip_features = self.clip_model.extract_features(clip_images, self.GlobalContextAdapter.fusion_map.values())
        attn_biases = self.GlobalContextAdapter(data_dict, clip_features, inference)
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)

        # 正常执行 FacialGuidance
        guide_fmp, guide_preds, mask = self.facialguidance(clip_images, landmarks)

        # 跳过 IFC，直接使用 guide_preds 作为 cls 输出
        clip_cls_output = self.clip_post_process(clip_output.float())

        # 计算权重
        weights = torch.softmax(torch.stack([self.w_global, self.w_guide, self.w_fusion]), dim=0)

        pred_dict = {
            'cls': guide_preds.float(),  # 使用 guide_preds 代替 fusion_preds
            'global_preds': clip_cls_output.float(),
            'guide_preds': guide_preds.float(),
            'fusion_preds': None,
            'mask': mask,
            'clip_fmp': clip_fmp,
            'guide_fmp': guide_fmp,
            'w_global': weights[0],
            'w_guide': weights[1],
            'w_fusion': weights[2]
        }
        return pred_dict

# 示例用法
# model = DFACLIP_NoIFC(mode='video')
# model.load_state_dict(torch.load('pretrained_dfaclip.pth'))
# model.eval()