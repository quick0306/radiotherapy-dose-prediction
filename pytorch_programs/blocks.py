""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels+mid_channels, out_channels, kernel_size=3, padding=1)
        self.BatchNorm3d_1 = nn.BatchNorm3d(mid_channels)
        self.BatchNorm3d_2 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        y = torch.cat([x, y], dim=1)
        z = self.conv2(y)
        z = self.act1(z)
        z = torch.cat([y, z], dim=1)
        return z




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, encode_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels , 64, kernel_size=2, stride=2)
            self.conv = DoubleConv(encode_channels+64, out_channels)


    def forward(self, x1, encode):
        x1 = self.up(x1)
        x = torch.cat([encode, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, out_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        return self.conv2(x)
