from src.models.DFACLIP.interactive_fusion import Interactive_fusion_classifier
import torch
import torch.nn.functional as F

from src.models.DFACLIP.DFACLIP import DFACLIP

class DFACLIP_NoFG(DFACLIP):
    def __init__(self, mode='video'):
        super().__init__(mode=mode)
        # 冻结所有参数，仅用于推理
        for param in self.parameters():
            param.requires_grad = False

        # 修改 IFC 以适应无 guide_fmp 的情况
        self.interactive_fusion_classifier = Interactive_fusion_classifier()

    def forward(self, data_dict, inference=True):
        images = data_dict['image']
        landmarks = data_dict['landmark']
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # 正常执行 GlobalContextAdapter
        clip_features = self.clip_model.extract_features(clip_images, self.GlobalContextAdapter.fusion_map.values())
        attn_biases = self.GlobalContextAdapter(data_dict, clip_features, inference)
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)

        # 跳过 FacialGuidance，直接用 clip_fmp 进行分类
        fusion_preds = self.interactive_fusion_classifier(clip_fmp, None)
        clip_cls_output = self.clip_post_process(clip_output.float())

        # 计算权重（无 guide_preds，w_guide 可忽略）
        weights = torch.softmax(torch.stack([self.w_global, self.w_guide, self.w_fusion]), dim=0)

        pred_dict = {
            'cls': fusion_preds.float(),
            'global_preds': clip_cls_output.float(),
            'guide_preds': None,  # 无 FacialGuidance 输出
            'fusion_preds': fusion_preds.float(),
            'mask': None,
            'clip_fmp': clip_fmp,
            'guide_fmp': None,
            'w_global': weights[0],
            'w_guide': weights[1],
            'w_fusion': weights[2]
        }
        return pred_dict