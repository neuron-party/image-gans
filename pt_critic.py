import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from critic import *
from unet import *


def main():
    image_path = '/home/MXH82T4/research/ILSVRC/Data/CLS-LOC/generator_train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    imagefolder = torchvision.datasets.ImageFolder(
        root=image_path,
        transform=transform
    )
    
    train, val = torch.utils.data.random_split(imagefolder, [..., ...])
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=16)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=16)
    
    device = torch.device('cuda:0')
    model = PatchCritic(in_channels=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.BCEWithLogitsLoss()
    
    generator = DeepUNet()
    weights = torch.load('weights/pt_generator_best.pth')
    generator.load_state_dict(weights['model'])
    generator = generator.to(device)
    
    avg_train_loss, avg_val_loss = [], []
    lowest_val_loss = np.inf
    
    for e in range(100):
        t_loss, v_loss = [], []
        for x, _ in trainloader:
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            fake_color = generator(y)
            
            real_img = torch.cat([y, uv], dim=1)
            fake_img = torch.cat([y, fake_color], dim=1)
            
            real_pred = model(real_img)
            fake_pred = model(fake_img.detach())
            
            real_labels = torch.Tensor([1]).expand_as(real_pred)
            fake_labels = torch.Tensor([0]).expand_as(fake_pred)
            
            loss_real = criterion(real_pred, real_labels)
            loss_fake = criterion(fake_pred, fake_labels)
            loss = (loss_real + loss_fake) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss.append(loss.detach().cpu().numpy())
            
        for x, _ in valloader:
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            fake_color = generator(y)
            
            real_img = torch.cat([y, uv], dim=1)
            fake_img = torch.cat([y, fake_color], dim=1)
            
            real_pred = model(real_img)
            fake_pred = model(fake_img.detach())
            
            real_labels = torch.Tensor([1]).expand_as(real_pred)
            fake_labels = torch.Tensor([0]).expand_as(fake_pred)
            
            loss_real = criterion(real_pred, real_labels)
            loss_fake = criterion(fake_pred, fake_labels)
            loss = (loss_real + loss_fake) / 2
            
            v_loss.append(loss.detach().cpu().numpy())
            
        avg_train_loss.append(np.mean(t_loss))
        avg_val_loss.append(np.mean(v_loss))
        
        if np.mean(v_loss) < lowest_val_loss:
            lowest_val_loss = np.mean(v_loss)
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict()
            }, 'weights/pt_critic_best.pth')
            
        print(f'Epoch: {e}, Train Loss: {np.mean(avg_train_loss)}, Val Loss: {np.mean(avg_val_loss)}')