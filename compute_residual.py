# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:31:11 2020

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
from time import time
from tqdm import tqdm
from verts_animation import verts_animation


sids = ['50004', '50020', '50021', '50022', '50025',
        '50002', '50007', '50009', '50026', '50027']
pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
        'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
        'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
        'one_leg_jump', 'running_on_spot']
femaleids = sids[:5]
maleids = sids[5:]

# sidpid = sid + '_' + pid

regis_dir = './dyna_registrations/dyna_male.h5'
data_dir = './DFaust_67'

f = h5py.File(regis_dir, 'r')

comp_device = torch.device('cpu')
smplmodel = SMPLModel(device=comp_device)
dmplmodel = DMPLModel(device=comp_device)

dataset = []

for maleid in maleids:
    # print('\n{} now is processing:'.format(maleid))
    data_fnames = glob.glob(os.path.join(data_dir, maleid, '*_poses.npz'))
    for data_fname in tqdm(data_fnames):
        sidpid = data_fname[18:-10]
        verts = f[sidpid].value.transpose([2, 0, 1])
        verts = torch.Tensor(verts).type(torch.float64)
        bdata = np.load(data_fname)
        betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
        pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
        pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
        if pose_body.shape[0]-verts.shape[0] != 0:
            # print(data_fname, pose_body.shape[0]-verts.shape[0])
            continue
        trans = torch.Tensor(bdata['trans']).type(torch.float64)
        dmpls = torch.Tensor(bdata['dmpls']).type(torch.float64)
        num_frame = pose_body.shape[0]
        betas = betas.repeat(num_frame,1)
        
        # s = time()
        smpl_vert = smplmodel(betas, pose_body, trans)
        # print(time()-s)
        
        # s = time()
        dmpl_vert = dmplmodel(betas, pose_body, trans, dmpls)
        # print(time()-s)
        
        # outmesh_path = './dmpl_batch_obj/dmpl_torch_{}.obj'
        # for i in range(dmpl_vert.shape[0]):
        #      dmplmodel.write_obj(dmpl_vert[i], outmesh_path.format(i))
        
        translation = torch.mean(verts - smpl_vert, 1).view(num_frame,1,3)
        tar_dbs = verts - (smpl_vert + translation)
        criterion = nn.MSELoss()
        smpl_loss = criterion(smpl_vert+translation, verts)
        dmpl_loss = criterion(dmpl_vert+translation, verts)
        
        pbs = torch.cat((pose_body,betas),1)
        dataset.append((pbs,tar_dbs))
        
        # del verts, betas, pose_body, trans, dmpls, smpl_vert, dmpl_vert
        torch.cuda.empty_cache()
        # print(num_frame)
        break
    # print('\n{} is done.\n'.format(maleid))
    break
# torch.save(dataset, 'male_pbs2dbs.pt')
verts_animation(verts)

    
        
        
        