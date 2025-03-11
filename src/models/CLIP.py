import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

from src.utils.download_online_models import *


# 类名与文件名保持一致
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        _, self.backbone = get_clip_visual()
        # _, self.backbone = get_vit_model() # vit model
        self.head = nn.Linear(1024, 2)  # for CLIP-large-14
        # self.head = nn.Linear(768, 2) # for CLIP-base-16

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}

        return pred_dict

model = CLIP().cuda()
inputs = torch.randn(64,3,224,224).cuda()
out = model(inputs)
print(out.shape)