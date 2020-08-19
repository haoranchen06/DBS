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
from verts_animation import verts_animation, png2video
import matplotlib.pyplot as plt


torch.cuda.empty_cache()
# bdata = np.load('./DFaust_67/50027/50027_jumping_jacks_poses.npz')
# bdata = np.load('./DFaust_67/50002/50002_jumping_jacks_poses.npz')
bdata = np.load('./HumanEva/S1/Box_3_poses.npz')
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
trans = torch.Tensor(bdata['trans']).type(torch.float64)
dmpls = torch.Tensor(bdata['dmpls']).type(torch.float64)
num_frame = pose_body.shape[0]
betas = betas.repeat(num_frame,1)
del bdata

# dataset = torch.load('male_bpts2dbs.pt')
# # std_vert = dataset[-9][1].cpu()
# std_vert = dataset[3][1].cpu()
# del dataset


smplmodel = SMPLModel(device=torch.device('cpu'))
smpl_vert = smplmodel(betas, pose_body, trans)

dmplmodel = DMPLModel(device=torch.device('cpu'))
dmpl_vert = dmplmodel(betas, pose_body, trans, dmpls)


dbs_model_path = {}

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-64-16.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_16hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_16hd_5stack.pt'

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-128-64.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_64hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_64hd_5stack.pt'

# dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_256-256-256.pt'
# dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_256hd_5stack.pt'
# dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_256hd_5stack.pt'

dbs_model_path['tcn'] = './body_models/dbs/male/dbs_tcn_512-768-1024.pt'
dbs_model_path['lstm'] = './body_models/dbs/male/dbs_lstm_1024hd_5stack.pt'
dbs_model_path['gru'] = './body_models/dbs/male/dbs_gru_1024hd_5stack.pt'

dbs_type = 'tcn'
dbsmodel = DBSModel(device=torch.device('cpu'),\
                    # dbs_type=dbs_type,num_c=[256,64,16],hd=16,num_sk=5,\
                    # dbs_type=dbs_type,num_c=[256,128,64],hd=64,num_sk=5,\
                    # dbs_type=dbs_type,num_c=[256,256,256],hd=256,num_sk=5,\
                    dbs_type=dbs_type,num_c=[512,768,1024],hd=1024,num_sk=5,\
                    dbs_model_path=dbs_model_path[dbs_type])
dbs_vert = dbsmodel(betas, pose_body, trans)

# verts_animation(smpl_vert,'./dbs_obj/smpl/visualize_{}.png')
# verts_animation(dmpl_vert)
verts_animation(dbs_vert)
# verts_animation(std_vert,'./dbs_obj/std/visualize_{}.png')


'''
To check the loss of SMPL, DMPL, DBS
'''

# loss_fn = nn.MSELoss()
# smpl_loss = loss_fn(smpl_vert,std_vert)
# dmpl_loss = loss_fn(dmpl_vert,std_vert)
# dbs_loss = loss_fn(dbs_vert,std_vert)

# print(smpl_loss,dmpl_loss,dbs_loss)
# print(smpl_loss - dbs_loss)


'''
Save whole sequence obj files
'''

# outmesh_path = './dbs_obj/smpl/visualize_{}.obj'
# for i in tqdm(range(num_frame)):
#     dbsmodel.write_obj(smpl_vert[i], outmesh_path.format(i))


# traing_data_sample = dataset[0] + dataset[2]
# torch.save(traing_data_sample, 'male_bpts2dbs_sample.pt')


'''
Check loss frame by frame
'''

# smpl_frame = []
# dmpl_frame = []
# dbs_frame = []

# for i in range(num_frame):
#     smpl_f_l = loss_fn(smpl_vert[i],std_vert[i])
#     dmpl_f_l = loss_fn(dmpl_vert[i],std_vert[i])
#     dbs_f_l = loss_fn(dbs_vert[i],std_vert[i])
#     smpl_frame.append(smpl_f_l.data.numpy())
#     dmpl_frame.append(dmpl_f_l.data.numpy())
#     dbs_frame.append(dbs_f_l.data.numpy())

# plt.style.use('ggplot')
# plt.figure()
# plt.title('SMPL DMPL DBS loss in each frame')
# plt.xlabel('the xth frame')
# plt.ylabel('loss')
# plt.plot(smpl_frame,linestyle='-')
# plt.plot(dmpl_frame,linestyle='-.')
# plt.plot(dbs_frame)

# plt.legend(['SMPL','DMPL','DBS_TCN'])

# plt.savefig('./analysis_fig/analysis.png')




