import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as FN

from timm.layers.helpers import to_2tuple

from ..local import *
from ..spa_spe_transformer import *
from ..enhance import CBAM


class SEBlock(nn.Module):
    def __init__(self, in_channel, embed_dim, patch_size, mid_dim):
        super(SEBlock, self).__init__()
        self.conv_conv1 = nn.Conv2d(in_channel, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(embed_dim)

        self.GAPool = nn.AvgPool2d(patch_size, stride=1)
        self.fc_reduction = nn.Linear(in_features=embed_dim, out_features=embed_dim // mid_dim)
        self.fc_extention = nn.Linear(in_features=embed_dim // mid_dim, out_features=embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_conv1(x)
        x = F.relu(self.bn_bn1(x), inplace=True)

        se_out = self.GAPool(x)
        se_out = se_out.view(se_out.size(0), -1)
        se_out = F.relu(self.fc_reduction(se_out), inplace=True)
        se_out = self.fc_extention(se_out)
        se_out = self.sigmoid(se_out)
        se_out = se_out.view(se_out.size(0), se_out.size(1), 1, 1)  # b x c x 1 x 1
        return se_out * x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=11, stride=11, padding=0, in_chans=200, embed_dim=64, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, 1, stride=1, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Mynet(nn.Module):
    def __init__(self, band, embed_dims, num_classes, num_heads):
        super().__init__()

        # -------------------------------------------------spatial branch-----------------------------------------------
        self.spa_local = spa_LFEM(band=band, classes=num_classes)

        self.patch_embed_new1 = SEBlock(in_channel=band, embed_dim=embed_dims[0], patch_size=11, mid_dim=16)
        self.spa_global1 = SpatialBlock(dim=embed_dims[0], num_heads=num_heads)

        self.patch_embed_new2 = SEBlock(in_channel=embed_dims[0], embed_dim=embed_dims[1], patch_size=11, mid_dim=16)
        self.spa_global2 = SpatialBlock(dim=embed_dims[1], num_heads=num_heads)

        self.patch_embed_new3 = SEBlock(in_channel=embed_dims[1], embed_dim=embed_dims[2], patch_size=11, mid_dim=16)
        self.spa_global3 = SpatialBlock(dim=embed_dims[2], num_heads=num_heads)

        self.patch_embed_new4 = SEBlock(in_channel=embed_dims[2], embed_dim=embed_dims[3], patch_size=11, mid_dim=16)
        self.spa_global4 = SpatialBlock(dim=embed_dims[3], num_heads=num_heads)

        # -------------------------------------------------spectral branch----------------------------------------------
        self.spe_local = spe_LFEM(band=band, classes=num_classes)

        self.patch_embed_new1 = SEBlock(in_channel=band, embed_dim=embed_dims[0], patch_size=11, mid_dim=16)
        self.spe_global1 = SpectralBlock(dim=embed_dims[0], num_heads=num_heads)

        self.patch_embed_new2 = SEBlock(in_channel=embed_dims[0], embed_dim=embed_dims[1], patch_size=11, mid_dim=16)
        self.spe_global2 = SpectralBlock(dim=embed_dims[1], num_heads=num_heads)

        self.patch_embed_new3 = SEBlock(in_channel=embed_dims[1], embed_dim=embed_dims[2], patch_size=11, mid_dim=16)
        self.spe_global3 = SpectralBlock(dim=embed_dims[2], num_heads=num_heads)

        self.patch_embed_new4 = SEBlock(in_channel=embed_dims[2], embed_dim=embed_dims[3], patch_size=11, mid_dim=16)
        self.spe_global4 = SpectralBlock(dim=embed_dims[3], num_heads=num_heads)

        # -----------------------------------------spectral-spatial feature enhance-------------------------------------
        self.CBA = CBAM(channel=band, ratio=16, kernel_size=7)

        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)

        # ------------------------------------------------Classifier head-----------------------------------------------
        self.pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(0.)
        self.fc = nn.Linear(band, num_classes, bias=False)

    def forward(self, x):
        # spectral branch
        B, _, H, W = x.shape
        x1 = self.spe_local(x)
        x2 = self.patch_embed_new1(x1)
        x3 = self.spe_global1(x2)
        x3 = x3.permute(0, 2, 1).view(B, -1, H, W)
        x4 = self.patch_embed_new2(x3)
        x5 = self.spe_global2(x4)
        x5 = x5.permute(0, 2, 1).view(B, -1, H, W)
        x6 = self.patch_embed_new3(x5)
        x7 = self.spe_global3(x6)
        x7 = x7.permute(0, 2, 1).view(B, -1, H, W)
        x8 = self.patch_embed_new4(x7)
        x9 = self.spe_global4(x8)
        x9 = x9.permute(0, 2, 1).view(B, -1, H, W)

        # spatial branch
        y1 = self.spa_local(x)
        y2 = self.patch_embed_new1(y1)
        y3 = self.spa_global1(y2)
        y3 = y3.permute(0, 2, 1).view(B, -1, H, W)
        y4 = self.patch_embed_new2(y3)
        y5 = self.spa_global2(y4)
        y5 = y5.permute(0, 2, 1).view(B, -1, H, W)
        y6 = self.patch_embed_new3(y5)
        y7 = self.spa_global3(y6)
        y7 = y7.permute(0, 2, 1).view(B, -1, H, W)
        y8 = self.patch_embed_new4(y7)
        y9 = self.spa_global4(y8)
        y9 = y9.permute(0, 2, 1).view(B, -1, H, W)

        # fusion(add)
        lmd = torch.sigmoid(self.lamuda)
        out = lmd * x9 + (1 - lmd) * y9
        # out = x9 + y1

        # enhance
        out = self.CBA(out)

        # for image classification
        out = self.pool(self.dropout(out)).view(-1, out.shape[1])
        f = self.fc(out)

        return f


def DASSTN(dataset):
    model = None
    if dataset == 'SV':
        model = Mynet(band=204, embed_dims=[64, 204, 64, 204], num_classes=16, num_heads=4)
    elif dataset == 'IP':
        model = Mynet(band=200, embed_dims=[64, 200], num_classes=16, num_heads=8)
    elif dataset == 'HUST2013':
        model = Mynet(band=144, embed_dims=[64, 144], num_classes=15, num_heads=4)
    return model


if __name__ == "__main__":
    t = torch.randn(size=(64, 204, 11, 11))
    print("input shape:", t.shape)
    net = DASSTN(dataset='SV')
    net.eval()
    print(net)
    print("output shape:", net(t).shape)

