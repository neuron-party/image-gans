import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffle(nn.Module):
    '''
    Applies an intial channel expansion and pixel shuffle upsampling for image expansion
    '''
    def __init__(self, in_channels, out_channels, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        out = self.shuffle(x)
        return out
    

class UpsampleBlock(nn.Module):
    '''
    Pixel shuffles and additional convolutions for channel reduction and image upsampling
    '''
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        hidden_in_channels,
        hidden_out_channels,
        attention=False,
        pix_bn=True,
        pix_bias=False
    ):
        super().__init__()
        self.shuffle = PixelShuffle(in_channels=in_channels, out_channels=out_channels, bn=pix_bn, bias=pix_bias)
        self.bn = nn.BatchNorm2d(bn_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_in_channels, out_channels=hidden_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_out_channels, out_channels=hidden_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_out_channels),
            nn.ReLU(inplace=True),
            Attention(in_channels=hidden_out_channels) if attention else nn.Identity()
        )
        
    def forward(self, x, y):
        x = self.shuffle(x)
        ssh = y.shape[-2:]
        if ssh != x.shape[-2:]:
            x = F.interpolate(x, y.shape[-2:], mode='nearest')
        x = torch.cat([x, self.bn(y)], dim=1)
        x = self.relu(x)
        out = self.conv2(self.conv1(x))
        return out
    

class Attention(nn.Module):
    '''
    Weird attention mechanism from SAGAN, try other variants?
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, bias=False)
        self.value = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=False)
        self.key = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, bias=False)
        self.gamma = nn.Parameter(torch.Tensor([0]))
        
    def forward(self, x):
        batch_size, shape = x.shape[0], x.size()
        x = x.flatten(2) # [b, c * 8, h * w]
        q, k, v = self.query(x), self.key(x), self.value(x) # [b, c, h * w],..., [b, c * 8, h * w]
        score = q.permute(0, 2, 1).contiguous() @ k # [b, h * w, c] @ [b, c, h * w] -> [b, h * w, h * w]
        score = F.softmax(score, dim=1)
        attention = self.gamma * (v @ score) + x # [b, c * 8, h * w] @ [b, h * w, h * w] -> [b, c * 8, h * w] + [b, c * 8, h * w]
        attention = attention.view(*shape).contiguous()
        return attention
    
    
class ITN_ResidualBlock(nn.Module):
    '''Double convolution residual block used in [https://arxiv.org/pdf/1703.10593.pdf, https://arxiv.org/abs/1603.08155]'''
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequenial(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = residual + x
        out = F.relu(x)
        return out
       

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high
        
    def forward(self, x):
        x = torch.sigmoid(x) * (self.high - self.low) + self.low
        return x