import numpy as np
from sklearn import metrics
from torch import nn
from .clip.clip import load
import torch

from .interactive_fusion import Interactive_fusion_classifier
from ..adapter.LandmarkGuidedAdapter import LandmarkGuidedAdapter
from ..adapter.adapter import Adapter
from .attn import RecAttnClip
from .layer import PostClipProcess, MaskPostXrayProcess
import torch.nn.functional as F

from ...config import MODEL_DOWNLOAD_ROOT


# from trainer.metrics.base_metrics_class import calculate_metrics_for_train


class DFACLIP(nn.Module):
    def __init__(self, mode='video'):
        clip_name = "ViT-L/14"
        adapter_vit_name = "vit_tiny_patch16_224"
        num_quires = 128
        fusion_map = {1: 1, 2: 8, 3: 15}
        mlp_dim = 256
        mlp_out_dim = 128
        head_num = 16
        device = "cuda"
        super().__init__()
        self.device = device
        self.clip_model, self.processor = load(clip_name, device=device,download_root=MODEL_DOWNLOAD_ROOT)
        self.GlobalContextAdapter = Adapter(vit_name=adapter_vit_name, num_quires=num_quires+60, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device)
        self.LandmarkGuidedAdapter = LandmarkGuidedAdapter(dim=768, mask_size=14, num_channels=10)

        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires,device=self.device)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mode = mode
        self._freeze()

        self.interactive_fusion_classifier = Interactive_fusion_classifier()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(2)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.w_global = nn.Parameter(torch.tensor(0.3))
        self.w_guide = nn.Parameter(torch.tensor(0.3))
        self.w_fusion = nn.Parameter(torch.tensor(0.4))

    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_model' in name :
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        """
            data_dict come from dataset-processed
            pred_dict come from model-processed
        """

        label = data_dict['label'] #N
        cls_fusion = pred_dict['cls']  #N 2
        global_preds = pred_dict['global_preds']
        guide_preds = pred_dict['guide_preds']
        fusion_preds = pred_dict['fusion_preds']
        criterion = nn.CrossEntropyLoss()
        # loss_cls = torch.mean(cls_fusion ** 2) # L2loss
        loss_global = criterion(global_preds, label)
        loss_guide = criterion(guide_preds, label)
        loss_fusion = criterion(fusion_preds, label)

        w_global = pred_dict['w_global']
        w_guide = pred_dict['w_guide']
        w_fusion = pred_dict['w_fusion']


        # 加权组合
        loss_overall = w_global * loss_global + w_guide * loss_guide + w_fusion * loss_fusion

        loss_dict = {
            'global': loss_global,
            'guide': loss_guide,
            'fusion': loss_fusion,
            'overall': loss_overall
        }
        return loss_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']
        landmarks = data_dict['landmark']
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )


        clip_features = self.clip_model.extract_features(clip_images, self.GlobalContextAdapter.fusion_map.values())
        attn_biases = self.GlobalContextAdapter(data_dict, clip_features, inference)
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)

        # landmark引导部分
        cilp_pool = self.pool(clip_output.permute(0,2,1)).squeeze(-1) # [B, 768]
        guide_fmp, guide_preds, mask = self.LandmarkGuidedAdapter(clip_images, landmarks)  # [B, 2048, 7, 7] [B, 2] [B, 10, 14, 14]

        fusion_preds = self.interactive_fusion_classifier(clip_fmp, guide_fmp, return_features=True) # [B, 2]
        if isinstance(fusion_preds, tuple):
            fusion_preds, fused_features = fusion_preds
        else:
            fused_features = None

        clip_cls_output = self.clip_post_process(clip_output.float()) # [B, 2]
        # 归一化权重
        weights = torch.softmax(torch.stack([self.w_global, self.w_guide, self.w_fusion]), dim=0)
        w_global, w_guide, w_fusion = weights[0], weights[1], weights[2]

        pred_dict = {
            'cls': fusion_preds.float(),
            # 'prob': prob,
            'global_preds': clip_cls_output.float(), # [B, 2]
            'guide_preds': guide_preds.float(), # [B, 2]  这个负作用
            'fusion_preds': fusion_preds.float(),
            'mask': mask,  # 用于可视化
            'clip_fmp': clip_fmp,
            'guide_fmp': guide_fmp,
            'features': fused_features,
            'w_global': w_global,
            'w_guide': w_guide,
            'w_fusion': w_fusion
        }


        return pred_dict
