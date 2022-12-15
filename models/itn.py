import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import *


class ITN(nn.Module):
    '''
    Image Transformation Network as mentioned in [https://arxiv.org/pdf/1703.10593.pdf, https://arxiv.org/abs/1603.08155]
    
    *pretty outdated generator, UNets work much better :/
    '''
    def __init__(self):
        super().__init__()
        
        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.residual_blocks = nn.ModuleList([ITN_ResidualBlock(channels=128) for i in range(9)])
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=1, bias=False)
        )
        
    def forward(self, x):
        x = self.down3(self.down2(self.down1(x)))
        for rb in self.residual_blocks:
            x = rb(x)
        out = self.up3(self.up2(self.up1(x)))
        return out