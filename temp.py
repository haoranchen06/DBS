# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:58:41 2020

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


# total_loss = torch.load('total_loss.pt')
# total_loss = torch.tensor(total_loss)
# total_loss = torch.mean(total_loss, 1)
# total_loss = total_loss.numpy()

# plt.figure()
# plt.plot(total_loss)

# torch.cuda.empty_cache()
# bdata = np.load('./DFaust_67/50002/50002_running_on_spot_poses.npz')
# betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64).cuda()
# pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
# pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1).cuda()
# trans = torch.Tensor(bdata['trans']).type(torch.float64).cuda()
# num_frame = pose_body.shape[0]
# betas = betas.repeat(num_frame,1)

training_data = torch.load('male_pbs2dbs.pt')

# smplmodel = SMPLModel(device=torch.device('cuda'))
# smpl_vert = smplmodel(betas, pose_body, trans)
# std_vert = smpl_vert + training_data[10][1].cuda()

# dbsmodel = DBSModel(device=torch.device('cuda'))
# dbs_vert = dbsmodel(betas, pose_body, trans)
# verts_animation(dbs_vert)

traing_data_sample = training_data[0] + training_data[2]
torch.save(traing_data_sample, 'male_pbs2dbs_sample.pt')



