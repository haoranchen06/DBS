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
from dbs_model_torch import DBSModel
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt



def train(dbs_type='tcn',num_c=[512,768,1024],hd=1024,num_sk=5,num_epochs=10,\
          training_data=None,val_data=None):

  device = torch.device('cuda')
  print(torch.cuda.get_device_name(0))

  dbsmodel = DBSModel(device=device, model_path='/content/drive/My Drive/model.pkl',\
               dbs_type=dbs_type,num_c=num_c,hd=hd,num_sk=num_sk)

  loss_function = nn.MSELoss()
  coef = 1e-5
  optimizer = optim.SGD(dbsmodel.parameters(), lr=10)
  # optimizer = optim.Adam(dbsmodel.parameters(), lr=0.001)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,90], gamma=0.5)

  data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)
  # del training_data
  total_loss = []
  val_total_loss = []
  
  for epoch in range(num_epochs):
    epoch_loss = []
    val_epoch_loss = []
    
    for data in val_loader:
      beta_pose_trans_seq = data[0].squeeze()
      betas = beta_pose_trans_seq[:,:10].cuda()
      pose = beta_pose_trans_seq[:,10:82].cuda()
      trans = beta_pose_trans_seq[:,82:].cuda()

      target_bs = data[1].squeeze().cuda()
      
      dbs_ver = dbsmodel(betas, pose, trans)
      loss = loss_function(dbs_ver, target_bs) + \
          coef * torch.norm(dbsmodel.dbs_layer.hidden2dbs.weight)
      val_epoch_loss.append(loss.data)
    
    for data in tqdm(data_loader):
      beta_pose_trans_seq = data[0].squeeze()
      betas = beta_pose_trans_seq[:,:10].cuda()
      pose = beta_pose_trans_seq[:,10:82].cuda()
      trans = beta_pose_trans_seq[:,82:].cuda()

      target_bs = data[1].squeeze().cuda()
      
      optimizer.zero_grad()
      dbs_ver = dbsmodel(betas, pose, trans)
      
      loss = loss_function(dbs_ver, target_bs) + \
          coef * torch.norm(dbsmodel.dbs_layer.hidden2dbs.weight)
      epoch_loss.append(loss.data)
      # print(loss.data)
      loss.backward()
      optimizer.step()
      torch.cuda.empty_cache()
    
    scheduler.step()
      
    total_loss.append(epoch_loss)
    val_total_loss.append(val_epoch_loss)
    
  # total_loss = torch.load('total_loss.pt')
  
  if dbs_type == 'tcn':
      model_name = 'dbs_{}_{}'.format(dbs_type,'-'.join([str(i) for i in num_c]))
  else:
      model_name = 'dbs_{}_{}hd_{}stack'.format(dbs_type, hd, num_sk)
  
  torch.save(dbsmodel.state_dict(),"/content/drive/My Drive/{}.pt".format(model_name),\
             _use_new_zipfile_serialization=False)
  

  total_loss = torch.tensor(total_loss)
  total_loss = torch.mean(total_loss, 1)
  total_loss = total_loss.numpy()
  
  val_total_loss = torch.tensor(val_total_loss)
  val_total_loss = torch.mean(val_total_loss, 1)
  val_total_loss = val_total_loss.numpy()

  plt.style.use('ggplot')
  plt.figure()
  plt.title('{}'.format(model_name))
  plt.xlabel('num of epoch')
  plt.ylabel('loss')
  plt.plot(total_loss)
  plt.plot(val_total_loss)
  plt.legend(['train','validate'])
  plt.savefig('/content/drive/My Drive/{}.png'.format(model_name))

  return total_loss, val_total_loss
