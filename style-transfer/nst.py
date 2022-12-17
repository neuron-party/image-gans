import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from models.lossnet import *
from utils import *

def main():
    device = torch.device('cuda:0')
    
    style_img_path, content_img_path = '/images/style1.jpeg', '/images/content1.jpeg'
    style_img, content_img = Image.open(style_img_path), Image.open(content_img_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    style_tensor, content_tensor = transform(style_img), transform(content_img)
    style_tensor, content_tensor = style_tensor.unsqueeze(0).to(device), content_tensor.unsqueeze(0).to(device)
    noise = torch.rand(1, 3, 256, 256, requires_grad=True, device=device) # 'white-noise' tensor to optimize
    noise = nn.Parameter(noise, requires_grad=True)
    
    criterion_mse = nn.MSELoss()
    model = VGG19_LN().to(device)
    for param in model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam([noise], lr=1e-1)
    
    for e in range(1500):
        content_pred, style_pred = model(noise)
        content_label, _ = model(content_tensor)
        _, style_label = model(style_tensor)
        
        # Content Loss
        content_loss = criterion_mse(content_pred, content_label) * 0.5
        
        # Style Loss
        style_pred_gm = [create_gram_matrix(x) for x in style_pred]
        style_label_gm = [create_gram_matrix(y) for y in style_pred]
        
        x1, x2, x3, x4, x5 = style_pred_gm
        y1, y2, y3, y4, y5 = style_label_gm
        
        # scaling factor for each loss component in the paper is 1 / (4(N^2 * M^2)) where N = # channels and M = height, width
        style_loss_1 = (1 / 4 * ((256 ** 2) * (64 ** 2))) * criterion_mse(x1, y1)
        style_loss_2 = (1 / 4 * ((128 ** 2) * (128 ** 2))) * criterion_mse(x1, y1)
        style_loss_3 = (1 / 4 * ((64 ** 2) * (256 ** 2))) * criterion_mse(x1, y1)
        style_loss_4 = (1 / 4 * ((32 ** 2) * (512 ** 2))) * criterion_mse(x1, y1)
        style_loss_5 = (1 / 4 * ((16 ** 2) * (512 ** 2))) * criterion_mse(x1, y1)
        
        style_loss = 0.2 * (style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4 + style_loss_5)
        
        # alpha/beta ratio is 1e-3 or 1e-4 for best results where alpha scales content and beta scales style
        loss = 1e-3 * content_loss + 1 * style_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 100 == 0:
            print(f'Epoch: {e}, Loss: {loss.detach().cpu().numpy()}')
            
if __name__ == '__main__':
    main()