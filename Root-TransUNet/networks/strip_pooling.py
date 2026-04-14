import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    来自 CVPR 2020 Strip Pooling 思路的简化版：
    - 沿 H 和 W 两个方向做条带平均池化
    - 生成 attention map，强调细长方向的连续结构
    用法：out = x * att + x （残差式）
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid = max(in_channels // reduction, 1)

        # 沿高度方向 (H,1)
        self.conv_h1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.conv_h2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

        # 沿宽度方向 (1,W)
        self.conv_w1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.conv_w2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # 高度条带池化: (B,C,H,W) -> (B,C,H,1)
        h_pool = F.adaptive_avg_pool2d(x, (H, 1))
        h = self.conv_h1(h_pool)
        h = self.act(h)
        h = self.conv_h2(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)

        # 宽度条带池化: (B,C,H,W) -> (B,C,1,W)
        w_pool = F.adaptive_avg_pool2d(x, (1, W))
        w = self.conv_w1(w_pool)
        w = self.act(w)
        w = self.conv_w2(w)
        w = F.interpolate(w, size=(H, W), mode="bilinear", align_corners=False)

        att = h + w
        att = self.bn(att)
        att = self.sigmoid(att)          # [0,1] attention

        out = x * att + x                # 残差
        return out
