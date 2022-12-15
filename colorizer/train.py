import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from models.critic import *
from models.unet import *


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
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    generator = DeepUNet()
    generator_weights = torch.load('weights/pt_generator_best.pth')
    generator.load_state_dict(generator_weights['model'])
    generator = generator.to(device)
    
    critic_weights = torch.load('weights/pt_critic_best.pth')
    critic = PatchCritic(in_channels=3)
    critic.load_state_dict(critic_weights['model'])
    critic = critic.to(device)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=2e-4)
    
    avg_train_loss_g, avg_train_loss_d, avg_val_loss_g, avg_val_loss_d = [], [], [], []
    lambda_l1 = 100.0
    
    for e in range(100):
        generator_tl, critic_tl, generator_vl, critic_vl = [], [], [], []
        
        for x, _ in trainloader:
            # Generator Training
            for param in critic.parameters():
                param.requires_grad = False
                
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            fake_color = generator(y)
            
            real_img = torch.cat([y, uv], dim=1)
            fake_img = torch.cat([y, fake_color], dim=1)
            
            gen_eval = critic(fake_img)
            gen_label = torch.Tensor([1]).expand_as(gen_eval).to(device)
            
            gen_eval_loss = criterion_bce(gen_eval, gen_label)
            gen_l1_loss = criterion_l1(fake_img, real_img)
            gen_loss = gen_eval_loss + lambda_l1 * gen_l1_loss
            
            optimizer_g.zero_grad()
            gen_loss.backward()
            optimizer_g.step()
            
            # Critic Training
            for param in critic.parameters():
                param.requires_grad = True
            
            real_pred = critic(real_img)
            fake_pred = critic(fake_img)
            
            real_labels = torch.Tensor([1]).expand_as(real_pred)
            fake_labels = torch.Tensor([0]).expand_as(fake_pred)
            
            loss_real = criterion_bce(real_pred, real_labels)
            loss_fake = criterion_bce(fake_pred, fake_labels)
            critic_loss = (loss_real + loss_fake) / 2
            
            optimizer_d.zero_grad()
            critic_loss.backward()
            optimizer_d.step()
            
            generator_tl.append(gen_loss.detach().cpu().numpy())
            critic_tl.append(critic_loss.detach().cpu().numpy())
            
        for x, _ in valloader:
            # Generator Validation
            y, uv = channel_transform(x)
            y, uv = y.to(device), uv.to(device)
            fake_color = generator(y)
            
            real_img = torch.cat([y, uv], dim=1)
            fake_img = torch.cat([y, fake_color], dim=1)
            
            gen_eval = critic(fake_img)
            gen_label = torch.Tensor([1]).expand_as(gen_eval).to(device)
            
            gen_eval_loss = criterion_bce(gen_eval, gen_label)
            gen_l1_loss = criterion_l1(fake_img, real_img)
            gen_loss = gen_eval_loss + lambda_l1 * gen_l1_loss
            
            # Critic Validation
            real_pred = critic(real_img)
            fake_pred = critic(fake_img)
            
            real_labels = torch.Tensor([1]).expand_as(real_pred)
            fake_labels = torch.Tensor([0]).expand_as(fake_pred)
            
            loss_real = criterion_bce(real_pred, real_labels)
            loss_fake = criterion_bce(fake_pred, fake_labels)
            critic_loss = (loss_real + loss_fake) / 2
            
            generator_vl.append(gen_loss.detach().cpu().numpy())
            critic_vl.append(critic_loss.detach().cpu().numpy())
            
        # Metrics
        avg_train_loss_g.append(np.mean(generator_tl))
        avg_train_loss_d.append(np.mean(critic_tl))
        avg_val_loss_g.append(np.mean(generator_vl))
        avg_val_loss_d.append(np.mean(critic_vl))
        
        if e % 10 == 0:
            torch.save({
                'model': generator.state_dict(),
                'optim': optimizer_g.state_dict()
            }, 'weights/generator_checkpoint.pth')
            torch.save({
                'model': critic.state_dict(),
                'optim': optimizer_d.state_dict()
            }, 'weights/critic_checkpoint.pth')
            
        print(f'Epoch: {e}, TL G: {np.mean(avg_train_loss_g)}, TL D: {np.mean(avg_train_loss_d)}, VL G: {np.mean(avg_val_loss_g)}, VL D: {np.mean(avg_val_loss_d)}')
        
if __name__ == '__main__':
    main()