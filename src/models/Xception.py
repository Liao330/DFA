import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()

        # 修改初始卷积参数
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 修改padding
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 修改第二卷积层padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)  # 增加padding
        self.bn2 = nn.BatchNorm2d(64)

        # 修改block参数保持尺寸一致性
        # 保持后续block的strides=2的总次数不变
        self.block1 = Block(64, 128, reps=2, strides=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, strides=2, start_with_relu=True)
        self.block3 = Block(256, 728, reps=2, strides=2, start_with_relu=True)

        # Middle flow保持不变
        self.block4 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block5 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block6 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block7 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block8 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block9 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block10 = Block(728, 728, reps=3, strides=1, start_with_relu=True)
        self.block11 = Block(728, 728, reps=3, strides=1, start_with_relu=True)

        # 修改exit flow的block
        self.block12 = Block(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False)

        # 后续卷积保持padding=1
        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 输入尺寸变化跟踪（以256x256为例）
        # Input: 256x256
        x = self.conv1(x)  # (256-3+2*1)/2 +1 = 128
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # (128-3+2*1)/1 +1 = 128
        x = self.bn2(x)
        x = self.relu(x)

        # 每经过一个strides=2的block，尺寸减半
        x = self.block1(x)  # 128 -> 64
        x = self.block2(x)  # 64 -> 32
        x = self.block3(x)  # 32 -> 16

        # 中间flow保持尺寸不变
        x = self.block4(x)  # 16
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        x = self.block12(x)  # 16 -> 8
        x = self.conv3(x)  # 8
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)  # 8
        x = self.bn4(x)
        x = self.relu(x)

        # 最终特征图尺寸应为8x8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


# 验证尺寸正确性
if __name__ == "__main__":
    model = Xception(num_classes=2)
    input_tensor = torch.randn(8, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # 应输出 torch.Size([8, 2])

