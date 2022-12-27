import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


# this implementation might be the most unreadable code i usually write :P
class VGG19_BN_LN(nn.Module):
    '''
    Pretrained VGG19 with BatchNorms 
    '''
    def __init__(self, instance_norm=False):
        super().__init__()
        
        self.splits = [[0, 3], [3, 10], [10, 17], [17, 30], [30, 31], [31, 52]]
        vgg19_weights = models.VGG19_BN_Weights.DEFAULT
        vgg19_features = models.vgg19_bn(weights=vgg19_weights).features
        self.convs = nn.ModuleList([nn.Sequential() for i in range(self.splits)])
        
        for i, split in enumerate(self.splits):
            for n in range(split[0], split[1]):
                # replacing maxpool with averagepool helps gradient flow
                if isinstance(vgg19_features[n], nn.modules.pooling.MaxPool2d):
                    vgg19_features[n] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
                self.convs[i].add_module(str(n), vgg19_features[n])
                
    def forward(self, x):
        '''
        Style Loss: [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
        Content Loss: [conv4_2]
        '''
        style_feature_maps = []
        for conv in self.convs:
            x = conv(x)
            style_feature_maps.append(x)
        
        content_feature_map = style_feature_maps.pop(4)
        return content_feature_map, style_feature_maps
    

class VGG19_LN(nn.Module):
    '''
    Standard pretrained VGG19
    '''
    def __init__(self):
        super().__init__()
        
        self.splits = [[0, 2], [2, 7], [7, 12], [12, 21], [21, 22], [22, 30]]
        vgg19_weights = models.VGG19_WEIGHTS.DEFAULT
        vgg19_features = models.vgg19(weights=vgg19_weights.feature)
        self.convs = nn.ModuleList([nn.Sequential() for i in range(len(self.splits))])
        
        for i, split in enumerate(self.splits):
            for n in range(split[0], split[1]):
                if isinstance(vgg19_features[n], nn.modules.pooling.MaxPool2d):
                    vgg19_features[n] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
                self.convs[i].add_module(str(n), vgg19_features[n])
                
    def forward(self, x):
        style_feature_maps = []
        for conv in self.convs:
            x = conv(x)
            style_feature_maps.append(x)
            
        content_feature_map = style_feature_maps.pop(4)
        return content_feature_map, style_feature_maps