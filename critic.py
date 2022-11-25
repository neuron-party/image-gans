import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCritic(nn.Module):
    '''
    https://arxiv.org/pdf/1611.07004.pdf
    Input: 3 channel RGB image [b, 3, 256, 256]
    Output: Tensor of shape [b, 1, 30, 30] with values in [0, 1] representing the probability of a patch being fake/real
    '''
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False), # [256, 256] -> [128, 128]
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_chanels=128, kernel_size=4, stride=2, padding=1, bias=False), # [128, 128] -> [64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False), # [64, 64] -> [32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False), # [32, 32] -> [31, 31]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False), # [31, 31] -> [30, 30]
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layer5(x)
        return out