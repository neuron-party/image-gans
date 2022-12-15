import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.models as models
import torchvision.transforms as transforms

from models.unet import *


def main():
    # Data
    folder_path = '/home/MXH82T4/research/ILSVRC/Data/CLS-LOC/generator_train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    imagefolder = torchvision.datasets.ImageFolder(
        root=folder_path,
        transform=transform
    )
    
    train, val = torch.utils.data.random_split(imagefolder, [..., ...])
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=16)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=16)
    
    device = torch.device('cuda:0')
    model = DeepUNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    avg_train_loss, avg_val_loss = [], []
    lowest_val_loss = np.inf
    
    for e in range(100):
        t_loss, v_loss = [], []
        for x, _ in trainloader:
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            pred = model(y)
            loss = criterion(pred, uv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss.append(loss.detach().cpu().numpy())
            
        for x, _ in valloader:
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            pred = model(y)
            loss = criterion(pred, uv)
            
            v_loss.append(loss.detach().cpu().numpy())
            
        avg_train_loss.append(np.mean(t_loss))
        avg_val_loss.append(np.mean(v_loss))
        
        if np.mean(v_loss) < lowest_val_loss:
            lowest_val_loss = np.mean(v_loss)
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, 'weights/pt_generator_best.pth')
        
        print(f'Epoch: {e}, Train Loss: {np.mean(avg_train_loss)}, Val Loss: {np.mean(avg_val_loss)}')