import os
import cv2
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from models.masknet import *
from models.lossnet import *
from utils import *
from st_utils import *


def main():
    # if no masks, then use np.zeros((content/style_pil.shape))
    content_pil = Image.open('images/content.jpg')
    style_pil = Image.open('images/style.jpg')
    content_mask = Image.open('images/content_mask.jpg')
    style_mask = Image.open('images/style_mask.jpg')
    
    content_np, style_np = np.array(content_pil), np.array(style_pil)
    content_mask_np, style_mask_np = np.array(content_mask), np.array(style_mask)
    
    # resize as needed, placeholder should be same shape as content/style images and masks
    content_img, style_img = cv2.resize(content_np, (256, 256)), cv2.resize(style_np, (256, 256))
    content_mask, style_mask = cv2.resize(content_mask_np, (256, 256)), cv2.resize(style_mask_np, (256, 256))
    
    placeholder = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    masknet = build_mask_net(placeholder, 'inside')
    
    layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    
    content_masks = compute_layer_masks(content_mask, content_layers, masknet)
    style_masks = compute_layer_masks(style_mask, style_layers, masknet)
    
    content_tensor = torch.Tensor(content_img).permute(2, 0, 1).unsqueeze(0)
    style_tensor = torch.Tensor(style_img).permute(2, 0, 1).unsqueeze(0)
    
    device = torch.device('cuda:0')
    lossnet = VGG19_LN().to(device)
    for param in lossnet.parameters():
        param.requires_grad = False
        
    optimizing_tensor = torch.rand(1, 3, 256, 256, requires_grad=True, device=device)
    optimizing_tensor = nn.Parameter(optimizing_tensor, requires_grad=True)
    
    content_tensor, style_tensor = content_tensor.to(device), style_tensor.to(device)
    optimizer = torch.optim.Adam([optimizing_tensor], lr=1e-1)
    criterion = nn.MSELoss(reduction='sum')
    
    content_masks = {
        'relu1_1': torch.Tensor(content_masks['relu1_1']).to(device),
        'relu2_1': torch.Tensor(content_masks['relu2_1']).to(device),
        'relu3_1': torch.Tensor(content_masks['relu3_1']).to(device),
        'relu4_1': torch.Tensor(content_masks['relu4_1']).to(device),
        'relu5_1': torch.Tensor(content_masks['relu5_1']).to(device),
    }
    
    style_masks = {
        'relu1_1': torch.Tensor(style_masks['relu1_1']).to(device),
        'relu2_1': torch.Tensor(style_masks['relu2_1']).to(device),
        'relu3_1': torch.Tensor(style_masks['relu3_1']).to(device),
        'relu4_1': torch.Tensor(style_masks['relu4_1']).to(device),
        'relu5_1': torch.Tensor(style_masks['relu5_1']).to(device),
    }
    
    # Training Loop
    for e in range(500):
        pred_content_fm, pred_style_fm = model(optimizing_tensor)
        content_fm, _ = model(content_tensor)
        _, style_fm = model(style_tensor)
        
        # Content Loss
        content_loss = criterion(pred_content_fm, content_fm)
        content_loss = content_loss / (512 * 32 * 32)
        
        # Style Loss
        pred_style_1, pred_style_2, pred_style_3, pred_style_4, pred_style_5 = pred_style_fm
        style_1, style_2, style_3, style_4, style_5 = style_fm
        
        pred_gm1 = masked_gram_matrix(pred_style_1, content_masks['relu1_1'])
        pred_gm2 = masked_gram_matrix(pred_style_2, content_masks['relu2_1'])
        pred_gm3 = masked_gram_matrix(pred_style_3, content_masks['relu3_1'])
        pred_gm4 = masked_gram_matrix(pred_style_4, content_masks['relu4_1'])
        pred_gm5 = masked_gram_matrix(pred_style_5, content_masks['relu5_1'])
        
        gm1 = masked_gram_matrix(style_1, style_masks['relu1_1'])
        gm2 = masked_gram_matrix(style_2, style_masks['relu2_1'])
        gm3 = masked_gram_matrix(style_3, style_masks['relu3_1'])
        gm4 = masked_gram_matrix(style_4, style_masks['relu4_1'])
        gm5 = masked_gram_matrix(style_5, style_masks['relu5_1'])
        
        style_loss_1 = criterion(pred_gm1, gm1) / (4 * 64 ** 2)
        style_loss_2 = criterion(pred_gm2, gm2) / (4 * 128 ** 2)
        style_loss_3 = criterion(pred_gm3, gm3) / (4 * 256 ** 2)
        style_loss_4 = criterion(pred_gm4, gm4) / (4 * 512 ** 2)
        style_loss_5 = criterion(pred_gm5, gm5) / (4 * 512 ** 2)
        
        style_loss = 0.2 * (style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4 + style_loss_5)
        
        loss = content_loss + style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 50 == 0:
            print(f'Epoch: {e}, Loss: {loss.detach().cpu().numpy()}')