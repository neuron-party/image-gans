import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from modules import *


class DeepUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = self._init_encoder()
        self.upsample1 = UpsampleBlock(
            in_channels=512, out_channels=1024, bn_channels=256, hidden_in_channels=512, hidden_out_channels=768
        )
        self.upsample2 = UpsampleBlock(
            in_channels=768, out_channels=1536, bn_channels=128, hidden_in_channels=512, hidden_out_channels=768, attention=True
        )
        self.upsample3 = UpsampleBlock(
            in_channels=768, out_channels=1536, bn_channels=64, hidden_in_channels=448, hidden_out_channels=672
        )
        self.upsample4 = UpsampleBlock(
            in_channels=672, out_channels=1344, bn_channels=64, hidden_in_channels=400, hidden_out_channels=300
        )
        self.upsample5 = PixelShuffle(in_channels=300, out_channels=1200)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=303, out_channels=303, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=303, out_channels=303, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=303, out_channels=3, kernel_size=3, stride=1, padding=1),
            SigmoidRange(low=-3.0, high=3.0)
        )
        
    def forward(self, x):
        '''
        UpsampleBlock transformations:
            [b, 512, 16, 16], [b, 256, 32, 32] -> [b, 256, 32, 32], [b, 256, 32, 32] -> [b, 512, 32, 32] -> [b, 768, 32, 32]
            [b, 768, 32, 32], [b, 128, 64, 64] -> [b, 384, 64, 64], [b, 128, 64, 64] -> [b, 512, 64, 64] -> [b, 768, 64, 64]
            [b, 768, 64, 64], [b, 64, 128, 128] -> [b, 384, 128, 128], [b, 64, 128, 128] -> [b, 448, 128, 128] -> [b, 672, 128, 128]
            [b, 672, 128, 128], [b, 64, 256, 256] -> [b, 336, 256, 256], [b, 64, 256, 256] -> [b, 400, 256, 256] -> [b, 300, 256, 256]
            [b, 300, 256, 256] -> [b, 300, 512, 512]
            
            [b, 300, 512, 512] -> [b, 303, 512, 512]
            [b, 303, 512, 512] -> [b, 3, 512, 512]
            
            return out
        '''
        x_original = x
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x1 = self.encoder.relu(x) # [b, 64, 256, 256]
        xt = self.encoder.maxpool(x1) 
        
        x2 = self.encoder.layer1(xt) # [b, 64, 128, 128]
        x3 = self.encoder.layer2(x2) # [b, 128, 64, 64]
        x4 = self.encoder.layer3(x3) # [b, 256, 32, 32]
        x5 = self.encoder.layer4(x4) # [b, 512, 16, 16]
        
        x6 = self.upsample1(x5, x4) 
        x7 = self.upsample2(x6, x3)
        x8 = self.upsample3(x7, x2)
        x9 = self.upsample4(x8, x1)
        x = self.upsample5(x9)
        
        x = torch.cat([x, x_original], dim=1) # [b, 303, 
        out = self.conv_block(x)
        
        return out
        
    def _init_encoder(self):
        resnet34_weights = models.ResNet34_Weights.DEFAULT
        encoder = models.resnet34(weights=resnet34_weights)
        encoder.avgpool, encoder.fc = nn.Identity(), nn.Identity()
        return encoder