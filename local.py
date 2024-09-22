import torch
import torch.nn as nn
import torch.nn.functional as F


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 光谱局部特征提取模块
class spe_LFEM(nn.Module):
    def __init__(self, band, classes):
        super(spe_LFEM, self).__init__()

        self.band = band
        self.classes = classes

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(in_channels=self.band, out_channels=64, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),  # , eps=0.001, momentum=0.1, affine=True),
            mish(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(in_channels=self.band, out_channels=64, padding=(1, 1, 2),
                      kernel_size=(3, 3, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            mish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(64),
            mish(),
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=72, padding=(1, 1, 2),
                      kernel_size=(3, 3, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(72),
            mish(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=72, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(72),
            mish(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=144, out_channels=self.band, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.band),
            mish(),
        )

    def forward(self, x):
        x0 = torch.unsqueeze(x, -1)
        x_spe1 = self.conv2_1(x0)
        x_spe2 = self.conv2_2(x0)
        x1 = torch.cat((x_spe1, x_spe2), dim=1)
        x2 = self.conv3(x1)
        x_spe3 = self.conv4_1(x2)
        x_spe4 = self.conv4_2(x2)
        x3 = torch.cat((x_spe3, x_spe4), dim=1)
        x4 = self.conv5(x3)
        x5 = torch.squeeze(x4, -1)
        return x5


# 空间局部特征提取模块
class spa_LFEM(nn.Module):
    def __init__(self, band, classes):
        super(spa_LFEM, self).__init__()

        self.band = band
        self.classes = classes

        self.spaconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=64, padding=(2, 2),
                      kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(64),
            mish()
        )
        self.spaconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=64, padding=(1, 1),
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            mish()
        )
        self.spaconv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            mish()
        )
        self.spaconv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=72, padding=(2, 2),
                      kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(72),
            mish()
        )
        self.spaconv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=72, padding=(1, 1),
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(72),
            mish()
        )
        self.spaconv2 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=self.band, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.band),
            mish()
        )

    def forward(self, x):
        x_spa1 = self.spaconv1_1(x)
        x_spa2 = self.spaconv1_2(x)
        y1 = torch.cat((x_spa1, x_spa2), dim=1)
        y2 = self.spaconv1(y1)
        x_spa3 = self.spaconv2_1(y2)
        x_spa4 = self.spaconv2_2(y2)
        y3 = torch.cat((x_spa3, x_spa4), dim=1)
        y4 = self.spaconv2(y3)
        return y4
