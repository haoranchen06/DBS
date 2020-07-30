# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:42:10 2020

@author: Haoran6
"""

import os
import glob
import h5py
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from smpl_torch_batch import SMPLModel
from dmpl_torch_batch import DMPLModel
from dbs_lstm import DBS_lstm
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# def train():
dbslayer = DBS_lstm(input_dim=82,hidden_dim=1024).cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(dbslayer.parameters(), lr=0.01)
# training_data_0 = [[torch.rand(500,82).double().cuda(),torch.rand(500,6890,3).double().cuda()] for i in range(10)]
# training_data_1 = [[torch.rand(150,82).double().cuda(),torch.rand(150,6890,3).double().cuda()] for i in range(10)]
# training_data = training_data_0 + training_data_1

training_data = torch.load('male_pbs2dbs.pt')
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)
del training_data
total_loss = []

for epoch in range(100):
    epoch_loss = []
    for data in tqdm(data_loader):
        pose_beta_seq = data[0].squeeze().cuda()
        target_dbs = data[1].squeeze().cuda()
        
        dbslayer.zero_grad()
        dbs_ver = dbslayer(pose_beta_seq)
        
        loss = loss_function(dbs_ver, target_dbs)
        epoch_loss.append(loss.data)
        # print(loss.data)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    
    total_loss.append(epoch_loss)

# total_loss = torch.load('total_loss.pt')
torch.save(dbslayer.state_dict(), './body_models/dbs/male/dbs_lstm_1024.pt')

total_loss = torch.tensor(total_loss)
total_loss = torch.mean(total_loss, 1)
total_loss = total_loss.numpy()

plt.figure()
plt.plot(total_loss)

# if __name__ == '__main__':
#     train()