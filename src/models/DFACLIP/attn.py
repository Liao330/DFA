from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from ..DFACLIP.layer import  MLP
from ...config import BATCH_SIZE


class ClipIntraBlock(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.num_features = num_features
        self.conv_first =nn.Conv1d(in_channels=self.num_features, out_channels=192, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_second =nn.Conv1d(in_channels=192, out_channels=self.num_features, kernel_size=1)

    def forward(self, x, data_dict, clip_L, inference):

        intra_x = x.permute(1, 2, 0)  # LND -> NDL
        intra_x = intra_x[:, :, -clip_L:].float()  # N  clipD 256/196

        intra_x = self.conv_first(intra_x)  # N D 256
        intra_x = self.relu(intra_x)
        intra_x = intra_x.permute(0, 2, 1)  # NDL-> NLD
        intra_x = intra_x.permute(0, 2, 1)  # NLD-> NDL
        intra_x = self.conv_second(intra_x)  # NDL

        intra_x = intra_x.permute(2, 0, 1)  # NDL- >LND
        return intra_x



class RecAttnClip(nn.Module):
    def __init__(self, vit, num_quires, device):
        super().__init__()
        self.vit = vit
        self.resblocks = self.vit.transformer.resblocks
        self.first_layer = 0
        self.clss_nums = num_quires
        self.ln_post = self.vit.ln_post
        self.proj = self.vit.proj
        self.num_features = self.vit.width
        self.device = device
        self.intra_scale = nn.Parameter(torch.zeros(1))
        self.intra_map = {6: 0}
        self.clip_intra_blocks = nn.ModuleList([ClipIntraBlock(self.num_features).to(self.device) for _ in range(1)])
        self._freeze()

    def build_attn_mask(self, attn_bias, clip_features, batch_size):
        v_len = clip_features[1].shape[1]  # 视觉特征长度，例如 256
        seq_len = self.clss_nums + 1 + v_len  # 128 + 1 + 256 = 385
        num_heads = self.resblocks[0].attn.num_heads  # 获取注意力头数

        if attn_bias is None or attn_bias.numel() == 0:
            # 生成默认掩码，形状为 [seq_len, seq_len]
            attn_mask = torch.zeros(seq_len, seq_len).to(self.device)
            # 设置默认掩码：查询部分（clss_nums）不关注其他位置
            attn_mask[:self.clss_nums, :] = -100  # 屏蔽所有位置
            attn_mask[torch.arange(self.clss_nums), torch.arange(self.clss_nums)] = 0  # 自注意力
            attn_mask[self.clss_nums:self.clss_nums + 1, :] = -100  # cls_token 不关注其他位置
            # 扩展为 [batch_size * num_heads, seq_len, seq_len]
            attn_mask = attn_mask[None, ...].expand(batch_size * num_heads, seq_len, seq_len).clone()
            return [attn_mask for _ in self.resblocks]  # 为每个块返回掩码
        else:
            n, Head, q, h, w = attn_bias.shape
            assert Head == num_heads, f"num_head={Head} is not supported. Modify to {num_heads}"
            attn_bias = attn_bias.reshape(n * Head, q, -1)
            l = attn_bias.shape[-1]
            attn_mask = attn_bias.new_zeros(q + 1 + l, q + 1 + l)
            attn_mask[:, :q] = -100
            attn_mask[torch.arange(q), torch.arange(q)] = 0
            attn_mask[:q, q] = -100
            attn_mask = attn_mask[None, ...].expand(n * Head, -1, -1).clone()
            attn_mask[:, :q, -l:] = attn_bias
            return [attn_mask for _ in self.resblocks]

    def forward(self, data_dict, clip_features, attn_bias, inference=False, normalize=False):
        batch_size = data_dict['image'].shape[0]  # 动态获取批次大小
        cls_token = clip_features[f'layer_{self.first_layer}_cls'].unsqueeze(1).permute(1, 0, 2).clone()
        vision_tokens = clip_features[self.first_layer].permute(1, 0, 2).clone()
        clss_token = cls_token.repeat(self.clss_nums, 1, 1)
        x = torch.cat([clss_token, cls_token, vision_tokens], dim=0)
        x.requires_grad = True
        clip_L = vision_tokens.shape[0]

        attn_biases = self.build_attn_mask(attn_bias, clip_features, batch_size)

        for i, blocks in enumerate(self.resblocks.children()):
            x = blocks(x, attn_biases[i])
            if i == 6:
                intra_x = self.clip_intra_blocks[self.intra_map[i]](x, data_dict, clip_L, inference)
                x[-clip_L:, ...] = intra_x * 0.05 + x[-clip_L:, ...]

        x = x.permute(1, 0, 2)
        clss_token = x[:, :self.clss_nums, :]
        clss_token = self.ln_post(clss_token)
        if self.proj is not None:
            clss_token = clss_token @ self.proj
        if normalize:
            clss_token = F.normalize(clss_token, dim=-1)

        feature_maps = self.clip_intra_blocks[self.intra_map[6]](x, data_dict, clip_L, inference)
        return clss_token, feature_maps

    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_intra_blocks' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
