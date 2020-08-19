# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:09:52 2020

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


dataset = torch.load('male_bpts2dbs.pt')
valset = dataset[-11:]
del dataset

data_dir = './DFaust_67'
maleid = '50027'
data_fnames = glob.glob(os.path.join(data_dir, maleid, '*_poses.npz'))

# smplmodel = SMPLModel(device=torch.device('cpu'))
# dmplmodel = DMPLModel(device=torch.device('cpu'))
dbs_model_path = {}

dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-64-16.pt'
dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_16hd_5stack.pt'
dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_16hd_5stack.pt'

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-128-64.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_64hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_64hd_5stack.pt'

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-256-256.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_256hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_256hd_5stack.pt'

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_512-768-1024.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_1024hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_1024hd_5stack.pt'

dbsmodel_tcn = DBSModel(device=torch.device('cpu'),\
                    dbs_type='tcn',num_c=[256,64,16],hd=16,num_sk=5,\
                    # dbs_type='tcn',num_c=[256,128,64],hd=64,num_sk=5,\
                    # dbs_type='tcn',num_c=[256,256,256],hd=256,num_sk=5,\
                    # dbs_type='tcn',num_c=[512,768,1024],hd=1024,num_sk=5,\
                    dbs_model_path=dbs_model_path['tcn'])

dbsmodel_lstm = DBSModel(device=torch.device('cpu'),\
                    dbs_type='lstm',num_c=[256,64,16],hd=16,num_sk=5,\
                    # dbs_type='lstm',num_c=[256,128,64],hd=64,num_sk=5,\
                    # dbs_type='lstm',num_c=[256,256,256],hd=256,num_sk=5,\
                    # dbs_type='lstm',num_c=[512,768,1024],hd=1024,num_sk=5,\
                    dbs_model_path=dbs_model_path['lstm'])

dbsmodel_gru = DBSModel(device=torch.device('cpu'),\
                    dbs_type='gru',num_c=[256,64,16],hd=16,num_sk=5,\
                    # dbs_type='gru',num_c=[256,128,64],hd=64,num_sk=5,\
                    # dbs_type='gru',num_c=[256,256,256],hd=256,num_sk=5,\
                    # dbs_type='gru',num_c=[512,768,1024],hd=1024,num_sk=5,\
                    dbs_model_path=dbs_model_path['gru'])

i = 0
j = 0
smpl_loss_list = []
dmpl_loss_list = []
tcn_loss_list = []
lstm_loss_list = []
gru_loss_list = []

for data_fname in tqdm(data_fnames):
    if i == 9 or i == 10:
        i += 1
        continue
    bdata = np.load(data_fname)
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
    pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
    pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
    trans = torch.Tensor(bdata['trans']).type(torch.float64)
    dmpls = torch.Tensor(bdata['dmpls']).type(torch.float64)
    num_frame = pose_body.shape[0]
    betas = betas.repeat(num_frame,1)
    
    
    std_vert = valset[j][1].cpu()
    # smpl_vert = smplmodel(betas, pose_body, trans)
    # dmpl_vert = dmplmodel(betas, pose_body, trans, dmpls)
    dbs_tcn = dbsmodel_tcn(betas, pose_body, trans)
    dbs_lstm = dbsmodel_lstm(betas, pose_body, trans)
    dbs_gru = dbsmodel_gru(betas, pose_body, trans)
    
    loss_fn = nn.MSELoss()
    # smpl_loss = loss_fn(smpl_vert,std_vert)
    # dmpl_loss = loss_fn(dmpl_vert,std_vert)
    tcn_loss = loss_fn(dbs_tcn,std_vert)
    lstm_loss = loss_fn(dbs_lstm,std_vert)
    gru_loss = loss_fn(dbs_gru,std_vert)
    
    # smpl_loss_list.append(smpl_loss.data.numpy())
    # dmpl_loss_list.append(dmpl_loss.data.numpy())
    tcn_loss_list.append(tcn_loss.data.numpy())
    lstm_loss_list.append(lstm_loss.data.numpy())
    gru_loss_list.append(gru_loss.data.numpy())
    
    del bdata
    
    i+=1
    j+=1

torch.save(tcn_loss_list,'./analysis_fig/loss/tcn_loss_16.pt')
torch.save(lstm_loss_list,'./analysis_fig/loss/lstm_loss_16.pt')
torch.save(gru_loss_list,'./analysis_fig/loss/gru_loss_16.pt')

# plt.style.use('ggplot')
# plt.figure()
# plt.title('Comparison of tcn lstm gru')
# plt.xlabel('index of pose sequence')
# plt.ylabel('loss')
# # plt.plot(smpl_loss_list,'o',linestyle='-')
# # plt.plot(dmpl_loss_list,'^',linestyle='-.')
# plt.plot(tcn_loss_list,'.',linestyle='--')
# plt.plot(lstm_loss_list,'o',linestyle='-')
# plt.plot(gru_loss_list,'^',linestyle='-.')

# # plt.legend(['smpl','dmpl','dbs'])
# plt.legend(['dbs_tcn','dbs_lstm','dbs_gru'])

# plt.savefig('./analysis_fig/analysis.png')