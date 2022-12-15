import os
import random
import itertools
from PIL import Image
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


from models.unet import *
from models.critic import *


def main():
    # note: the original cyclegan for style transfer was trained using an image transformation network as the generator, i use a unet
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    X_dataset = torchvision.datasets.ImageFolder(root='...', transform=transform)
    Y_dataset = torchvision.datasets.ImageFolder(root='...', transform=transform)
    
    X_t, X_v = torch.utils.data.random_split(X_dataset, [6210, 690])
    Y_t, Y_v = torch.utils.data.random_split(Y_dataset, [6210, 690])
    
    X_trainloader = torch.utils.data.DataLoader(X_t, batch_size=1, shuffle=True)
    X_valloader = torch.utils.data.DataLoader(X_v, batch_size=1, shuffle=True)
    Y_trainloader = torch.utils.data.DataLoader(Y_t, batch_size=1, shuffle=True)
    Y_valloader = torch.utils.data.DataLoader(Y_v, batch_size=1, shuffle=True)
    
    device = torch.device('cuda:3') # 4 GPUs bro :D
    
    # todo: add parameters to model files for flexibility in the future
    generator_g = DeepUNet().to(device)
    generator_f = DeepUNet().to(device)
    discriminator_y = PatchCritic().to(device)
    discriminator_x = PatchCritic().to(device)
    
    optimizer_G = torch.optim.Adam(itertools.chain(generator_g.parameters(), generator_f.parameters()), lr=2e-4)
    optimizer_D = torch.optim.Adam(itertools.chain(discriminator_y.parameters(), discriminator_x.parameters()), lr=2e-4)
    
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    x_image_buffer, y_image_buffer = deque(maxlen=50), deque(maxlen=50)
    
    avg_generator_tl, avg_discriminator_y_tl, avg_discriminator_x_tl = [], [], []
    avg_generator_vl, avg_discriminator_y_vl, avg_discriminator_x_vl = [], [], []
    
    for e in range(100):
        generator_tl, discriminator_y_tl, discriminator_x_tl = [], [], []
        generator_vl, discriminator_y_tl, discriminator_x_tl = [], [], []
        
        for x, y in zip(X_trainloader, Y_trainloader):
            x, y = x[0].to(device), y[0].to(device)
            
            # No discriminator gradient propagation for generator training
            for py, px in zip(discriminator_y.parameters(), discriminator_x.parameters()):
                py.requires_grad, px.requires_grad = False, False
                
            y_fake = generator(x) # style transfer from domain X to domain Y
            x_fake = generator(y) # style transfer from domain Y to domain X
            
            generator_g_eval = discriminator_y(y_fake) # Y-domain discriminator prediction on fake Y-generated domain image
            generator_f_eval = discriminator_x(x_fake) # vice versa
            
            generator_g_label = torch.Tensor([1]).expand_as(generator_g_eval).to(device)
            generator_f_label = torch.Tensor([1]).expand_as(generator_f_eval).to(device)
            
            generator_g_loss = criterion_bce(generator_g_eval, generator_g_label) ** 2
            generator_f_loss = criterion_bce(generator_f_eval, generator_f_label) ** 2
            
            x_cycle = generator_f(y_fake) # cycling back from the Y-domain to the X-domain
            y_cycle = generator_g(x_fake) # vice versa
            
            x_cycle_loss = criterion_l1(x_cycle, x)
            y_cycle_loss = criterion_l1(y_cycle, y)
            cycle_loss = x_cycle_loss + y_cycle_loss
            
            generator_loss = generator_g_loss + generator_f_loss + 10.0 * cycle_loss # lambda parameter
            optimizer_G.zero_grad()
            generator_loss.backward()
            optimizer_G.step()
                                     
            # update image buffers
            y_image_buffer.append(y_fake)
            x_image_buffer.append(x_fake)
            
            
            # Discriminator Training
            for py, px in zip(discriminator_y.parameters(), discriminator_x.parameters()):
                py.requires_grad, px.requires_grad = True, True
                
            y_fake, x_fake = random.choice(y_image_buffer), random.choice(x_image_buffer)
            
            y_fake_pred, y_true_pred = discriminator_y(y_fake.detach()), discriminator_y(y)
            x_fake_pred, x_true_pred = discriminator_x(x_fake.detach()), discriminator_x(x)
            
            y_true_label = torch.Tensor([1]).expand_as(y_true_pred).to(device)
            y_fake_label = torch.Tensor([0]).expand_as(y_fake_pred).to(device)
            x_true_label = torch.Tensor([1]).expand_as(x_true_pred).to(device)
            x_fake_label = torch.Tensor([0]).expand_as(x_fake_pred).to(device)
            
            y_fake_loss = criterion_bce(y_fake_pred, y_fake_label) ** 2
            y_true_loss = criterion_bce(y_true_pred, y_true_label) ** 2
            x_fake_loss = criterion_bce(x_fake_pred, x_fake_label) ** 2
            x_true_loss = criterion_bce(x_true_pred, x_true_label) ** 2
            discriminator_y_loss = (y_fake_loss + y_true_loss) / 2
            discriminator_x_loss = (x_fake_loss + x_true_loss) / 2
            
            optimizer_D.zero_grad()
            discriminator_y_loss.backward()
            discriminator_x_loss.backward()
            optimizer_D.step()
            
        # Metrics
            generator_tl.append(generator_loss.detach().cpu().numpy())
            discriminator_y_tl.append(discriminator_y_loss.detach().cpu().numpy())
            discriminator_x_tl.append(discriminator_x_loss.detach().cpu().numpy())
        
        avg_generator_tl.append(np.mean(generator_tl))
        avg_discriminator_y_tl.append(np.mean(discriminator_y_tl))
        avg_discriminator_x_tl.append(np.mean(discriminator_x_tl))