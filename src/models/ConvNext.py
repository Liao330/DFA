import torch
from torch import nn
from torchvision import models

from src.config import  NUM_CLASS


# 类名与文件名保持一致
class ConvNext(nn.Module):
    def __init__(self):
        super(ConvNext, self).__init__()
        num_classes = NUM_CLASS
        model = models.convnext_large(pretrained=True)
        # model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        model = model.cuda()
        self.model = nn.Sequential(*list(model.children())[:-2])
        # self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(1536, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        #print("111",x.shape) #[64,  3, 224, 224]
        fmap = self.model(x)
        # print("333", fmap.shape) # [64, 2048, 7, 7]
        x = self.avgpool(fmap)
        # print("444", x.shape) # [64, 2048, 1, 1]
        x = x.view(batch_size,  -1)
        # print("555", x.shape) # [64, 2048]
        # x_lstm, _ = self.lstm(x, None)
        # print("666", x_lstm.shape) #[4, 60, 1536]
        outs = self.dp(self.linear1(x)) # [64, 2]
        return outs

# model = ConvNext().cuda()
# inputs = torch.randn(64,3,224,224).cuda()
# out = model(inputs)
# print(out.shape)

