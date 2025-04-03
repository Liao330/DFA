import numpy as np
from sklearn import metrics
from torch import nn
from .clip.clip import load
import torch

from .interactive_fusion import Interactive_fusion_classifier
from ..adapter.FacialGuidance import FacialGuidance
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
        # 全局伪造适配器
        self.GlobalContextAdapter = Adapter(vit_name=adapter_vit_name, num_quires=num_quires+60, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device)
        # landmark引导适配器
        self.LandmarkGuidedAdapter = LandmarkGuidedAdapter(dim=768, mask_size=14, num_channels=10)
        # FG使用CrossAttention
        self.facialguidance = FacialGuidance()

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

    # def get_train_metrics(self, data_dict, pred_dict):
    #     label = data_dict['label']
    #     pred = pred_dict['cls']
    #     auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
    #     metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
    #     return metric_batch_dict

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
        # print(self.GlobalContextAdapter.fusion_map.values()) #· dict_values([1, 8, 15])
        # print(f"clip_features:{clip_features.shape}") #  'dict' object has no attribute 'shape'
        # print(f"clip_features:{clip_features[1].shape} {clip_features[8].shape} {clip_features[15].shape}")
        #torch.Size([32, 256, 1024]) torch.Size([32, 256, 1024]) torch.Size([32, 256, 1024])

        attn_biases = self.GlobalContextAdapter(data_dict, clip_features, inference)
        # print(attn_biases[0].shape, xray_preds[0].shape) # (N Head D h w)  (N Q h w)
        # torch.Size([32, 16, 128, 16, 16]) torch.Size([32, 128, 16, 16])
        clip_output, clip_fmp = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)
        # print(f"clip_output:{clip_output.shape}") # [B, 196-68, 768]
        # print(clip_fmp.shape) # [8, 385, 1024]
        # landmark融合部分
        # f_lfa = self.LandmarkFocusAdapter(clip_output, landmarks) # # [B, 768]
        # # print(f"f_lfa:{f_lfa.shape}") # [B, 768]
        #
        # cilp_pool = self.pool(clip_output.permute(0,2,1))
        # # print(f"cilp_pool:{cilp_pool.shape}") # [B, 768, 1]
        # landmark_final = torch.cat([cilp_pool.squeeze(-1), f_lfa], dim=-1)
        # # print(f"landmark_final:{landmark_final.shape}") # [B, 1536]
        # landmark_final = self.pool3(landmark_final)
        # data_dict['if_boundary'] = data_dict['if_boundary'].to(self.device)
        # xray_preds = [self.masked_xray_post_process(xray_pred, data_dict['if_boundary']) for xray_pred in xray_preds]
        # xray_preds = [self.masked_xray_post_process(xray_pred) for xray_pred in xray_preds]
        # # print(xray_preds[-1].shape) # [B, 1, 256, 256]
        #
        # clip_output = self.pool2(clip_output.permute(0, 2, 1))  # [B, 768, 196] -> [B, 768, 128]
        # clip_output = clip_output.permute(0, 2, 1)  # [B, 768, 128] -> [B, 128, 768]
        # cls_output = self.clip_post_process(clip_output.float()).squeeze()  # N 2
        # # print(cls_output.shape) # [B, 2]

        # landmark引导部分
        # cilp_pool = self.pool(clip_output.permute(0,2,1)).squeeze(-1) # [B, 768]
        # guide_fmp, guide_preds, mask = self.LandmarkGuidedAdapter(clip_images, landmarks)  # [B, 2048, 7, 7] [B, 2] [B, 10, 14, 14]
        # landmark_final = torch.cat([cilp_pool, f_guided], dim=-1)  # [B, 1536]
        # cls_output = self.classifier(landmark_final)  # [B, 2]

        # 使用FG
        guide_fmp, guide_preds, mask = self.facialguidance(clip_images, landmarks)  # [B, 2048, 7, 7] [B, 2] [B, 10, 14, 14]

        # print(f"clip_fmp: {clip_fmp.shape}") # [B, 385, 1024] 没问题
        # print(f"guide_fmp: {guide_fmp.shape}") # [B, 2048, 14, 14] 原来是--》[B, 2048, 7, 7]
        fusion_preds = self.interactive_fusion_classifier(clip_fmp, guide_fmp) # [B, 2]

        # print(f"clip_output:{clip_output.shape}") # [1, 128, 768]
        clip_cls_output = self.clip_post_process(clip_output.float()) # [B, 2]
        # print(f"clip_cls_output.shape: {clip_cls_output.shape}") # [1, 2]
        # cls_output = torch.cat([guide_preds])

        # print(xray_preds[-1].shape) # [B, 1, 256, 256])
        # prob = torch.softmax(cls_output, dim=1)[:, 1]

        # 归一化权重
        weights = torch.softmax(torch.stack([self.w_global, self.w_guide, self.w_fusion]), dim=0)
        w_global, w_guide, w_fusion = weights[0], weights[1], weights[2]

        pred_dict = {
            'cls': fusion_preds.float(), # 效果不好的话换成guide_preds 在loss中的权重也对应调高
            # 'prob': prob,
            'global_preds': clip_cls_output.float(), # [B, 2]
            'guide_preds': guide_preds.float(), # [B, 2]
            'fusion_preds': fusion_preds.float(),
            'mask': mask,  # 用于可视化
            'clip_fmp': clip_fmp,
            'guide_fmp': guide_fmp,
            'w_global': w_global,
            'w_guide': w_guide,
            'w_fusion': w_fusion
        }


        return pred_dict
