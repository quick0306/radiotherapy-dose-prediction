import torch.nn.functional as F

from .blocks import *


class Att_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Att_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(n_channels+32, 16)
        self.down2 = Down(n_channels+64, 16)
        self.down3 = Down(n_channels+96, 16)
        self.down4 = Down(n_channels+128, 16)
        self.gating = UnetGridGatingSignal3(n_channels+160, n_channels+160, kernel_size=(1, 1, 1), is_batchnorm=True)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(n_channels+64, n_channels+96+96, n_channels+64,
                                                   nonlocal_mode='concatenation',
                                                   sub_sample_factor=(2,2,2))
        self.attentionblock3 = MultiAttentionBlock(n_channels+96, n_channels+128+96, n_channels+96,
                                                   nonlocal_mode='concatenation', 
                                                   sub_sample_factor=(2,2,2))
        self.attentionblock4 = MultiAttentionBlock(n_channels+128, n_channels+160, n_channels+128,
                                                   nonlocal_mode='concatenation', 
                                                   sub_sample_factor=(2,2,2))

       # factor = 2 if bilinear else 1
        
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
        gating = self.gating(x5)
        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(x4, gating)
        x = self.up1(x5, g_conv4)
        g_conv3, att3 = self.attentionblock3(x3, x)
        x = self.up2(x, g_conv3)
        g_conv2, att2 = self.attentionblock2(x2, x)
        x = self.up3(x, g_conv2)
        x = self.up4(x, x1)
        dose = self.outc(x)
        return dose