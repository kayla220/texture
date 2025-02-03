from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt

import os
import time
from PIL import Image
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from dataset import FabricDataset
from net import GeneratorUNet, Discriminator
from utils import initialise_weights

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1,512//2**4,512//2**4) #patch 32

lr = 2e-4
beta1 = 0.5
beta2 = 0.999

def train(train_dataloader, weight_path):
    model_gen = GeneratorUNet().to(device)
    model_dis = Discriminator().to(device)
    
    opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))
    
    model_gen.train()
    model_dis.train()

    batch_count = 0
    num_epochs = 100
    start_time = time.time()

    loss_hist = {'gen':[],
                'dis':[]}

    for epoch in range(num_epochs):
        for a, b in tqdm(train_dataloader):
            ba_si = a.size(0)

            # real image
            real_a = a.to(device)
            real_b = b.to(device)

            # patch label
            real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

            # generator
            model_gen.zero_grad()

            fake_b = model_gen(real_a) # 가짜 이미지 생성
            out_dis = model_dis(fake_b, real_b) # 가짜 이미지 식별

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_b, real_b)

            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            opt_gen.step()

            # discriminator
            model_dis.zero_grad()

            out_dis = model_dis(real_b, real_a) # 진짜 이미지 식별
            real_loss = loss_func_gan(out_dis,real_label)
            
            out_dis = model_dis(fake_b.detach(), real_a) # 가짜 이미지 식별
            fake_loss = loss_func_gan(out_dis,fake_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())

            batch_count += 1
            if epoch % 10 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
               
                os.makedirs(weight_path, exist_ok=True)
                weights_gen = os.path.join(weight_path, f'weights_gen_e{epoch}.pt')
                weights_dis = os.path.join(weight_path, f'weights_dis_e{epoch}.pt')

                torch.save(model_gen.state_dict(), weights_gen)
                torch.save(model_dis.state_dict(), weights_dis)
    
        
if __name__=='__main__':
    image_size=512
    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Resize((image_size,image_size))
    ])

    data_path = r"/mnt/storage/dataset/Pascal/pix2pix"
    weight_path = '/workspace/texture/pix2pix/models'
    train_dataset = FabricDataset(data_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(train_dataset.__len__())
    img, label = train_dataset[0]
    print(img.size())
    
    train(train_dataloader, weight_path)
    
    
    
    