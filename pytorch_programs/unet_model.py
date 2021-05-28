import torch.nn.functional as F

from .blocks import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(n_channels+32, 16)
        self.down2 = Down(n_channels+64, 16)
        self.down3 = Down(n_channels+96, 16)
       # factor = 2 if bilinear else 1
        self.down4 = Down(n_channels+128, 16)
        self.up1 = Up(n_channels+160, 16, n_channels+128, bilinear)
        self.up2 = Up(n_channels+128+96, 16, n_channels+96, bilinear)
        self.up3 = Up(n_channels+96+96, 16, n_channels+64, bilinear)
        self.up4 = Up(n_channels+64+96, 16, n_channels+32, bilinear)
        self.outc = OutConv(n_channels+32+96, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        dose = self.outc(x)
        return dose