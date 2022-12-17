# perceptual loss network for style transfers
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
from models.itn import *


def main():
    device = torch.device('cuda:3')
    
    model = ITN().to(device)
    loss_network = VGG19_LN().to(device)
    for param in loss_network.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    
    # Since the loss network is fixed, we don't need to compute these at every iteration
    _, style_feature_maps = loss_network(style_tensor)
    style_gram_matrices = [create_gram_matrix(i) for i in style_feature_maps]
    
    for e in range(2): # supposedly 2 epochs is enough (~40,000 iterations with batch_size=4)
        for x, _ in trainloader:
            x = x.to(device)
            out = model(x)
            
            content_pred, style_pred = loss_network(out)
            content_label, _ = loss_network(x)
            style_pred_gram_matrices = [create_gram_matrix(i) for i in style_pred]
            
            # Content Loss
            # total variation regularization (TVR with a strength between 1e-6 and 1e-4 chosen via cross validation per style target)
            # not sure what lambda_TV should be though, gotta reskim paper again
            # the paper is missing a lot of information regarding scaling factors and loss function 
            content_loss = 0.5 * criterion_mse(content_pred, content_label)
            
            style_loss = 