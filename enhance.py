import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(x1)))
        max_out = self.fc2(self.relu(self.fc1(x2)))
        y = avg_out + max_out
        y = self.sigmoid(y)
        y1 = y * x
        out = y1 + x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        y = self.sigmoid(x2)
        y1 = y * x
        out = y1 + x
        return out


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_ca = self.ca(x)
        x_sa = self.sa(x)
        out = x_ca + x_sa
        return out
