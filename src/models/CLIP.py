import torch
import torch.nn as nn
from src.utils.download_online_models import *
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig


# 类名与文件名保持一致
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        _, self.backbone = get_clip_visual()
        # _, self.backbone = get_vit_model() # vit model
        self.linear = nn.Linear(768, 1024)
        self.head = nn.Linear(1024, 2)  # for CLIP-large-14
        # self.head = nn.Linear(768, 2) # for CLIP-base-16

    def forward(self, img, inference=False):
        # get the features by backbone
        features = self.backbone(img)['pooler_output']
        # print(f"000  {features.shape}") # [64, 768]
        # change the out_channels by linear
        features = self.linear(features)
        # print(f"111  {features.shape}") # [64, 1024]
        # get the prediction by head
        pred = self.head(features)
        # print(f"222  {pred.shape}") # [64, 2]
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)
        # print(f"333  {prob.shape}") # [64, 2]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        labels = pred_dict['prob']
        # print(f"444  {labels.shape}") # [64]
        # print(f"555  {pred_dict['feat'].shape}") # [64， 1024]
        labels = labels.type(torch.float32) # expect float type
        return labels

# # 测试
# model = CLIP().cuda()
# inputs = torch.randn(64,3,224,224).cuda()
# out = model(inputs)
# print(out)