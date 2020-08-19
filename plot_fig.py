# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:58:57 2020

@author: Haoran6
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch import nn
from smpl_torch_batch import SMPLModel
from dmpl_torch_batch import DMPLModel
from dbs_model_torch import DBSModel
from time import time
from tqdm import tqdm
from verts_animation import verts_animation
import matplotlib.pyplot as plt


tcn_list = []
lstm_list = []
gru_list = []

x_axis = [16,64,256,1024]

for i in [16,64,256,1024]:
    tcn_loss = np.mean(torch.load('./analysis_fig/loss/tcn_loss_{}.pt'.format(i)))
    lstm_loss = np.mean(torch.load('./analysis_fig/loss/lstm_loss_{}.pt'.format(i)))
    gru_loss = np.mean(torch.load('./analysis_fig/loss/gru_loss_{}.pt'.format(i)))
    tcn_list.append(tcn_loss)
    lstm_list.append(lstm_loss)
    gru_list.append(gru_loss)

plt.style.use('ggplot')
plt.figure()
plt.title('tcn lstm gru loss versus hidden dimension')
plt.xlabel('hidden dimension (number of components)')
plt.ylabel('loss')
# plt.plot(smpl_loss_list,'o',linestyle='-')
# plt.plot(dmpl_loss_list,'^',linestyle='-.')
plt.plot(x_axis,tcn_list,'.',linestyle='--')
plt.plot(x_axis,lstm_list,'o',linestyle='-')
plt.plot(x_axis,gru_list,'^',linestyle='-.')

# plt.legend(['smpl','dmpl','dbs'])
plt.legend(['dbs_tcn','dbs_lstm','dbs_gru'])

plt.savefig('./analysis_fig/analysis.png')