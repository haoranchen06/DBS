# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:43:53 2020

@author: Haoran6
"""

import numpy as np
import torch
from torch import nn
from smpl_torch import SMPLModel
from dmpl_torch import DMPLModel
import h5py

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bdata = np.load("./DFaust_67/50002/50002_chicken_wings_poses.npz")
fId = 100
pose_body = torch.Tensor(bdata['poses'][fId, 3:72]).squeeze().type(torch.float64).to(comp_device)
pose_body = torch.cat((torch.zeros(3).type(torch.float64).to(comp_device),pose_body),0)
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64).to(comp_device)
dmpls = torch.Tensor(bdata['dmpls'][fId]).type(torch.float64).to(comp_device)
# trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(comp_device)
trans = torch.Tensor(bdata['trans'][fId]).type(torch.float64).to(comp_device)
smplmodel = SMPLModel(device=comp_device)
dmplmodel = DMPLModel(device=comp_device)
smpl_vert = smplmodel(betas, pose_body, trans)
dmpl_vert = dmplmodel(betas, pose_body, trans, dmpls)
with h5py.File('./dyna_registrations/dyna_male.h5', 'r') as f:
    verts = f['50002_chicken_wings'].value.transpose([2, 0, 1])
standard_vert = torch.Tensor(verts[fId]).type(torch.float64).to(comp_device)

translation = torch.mean(standard_vert-smpl_vert,0)
criterion = nn.MSELoss()
smpl_loss = criterion(smpl_vert+translation, standard_vert)
dmpl_loss = criterion(dmpl_vert+translation, standard_vert)
print(smpl_loss, dmpl_loss)


# smplmodel.write_obj(smpl_vert,'./smpl_ver.obj')
# dmplmodel.write_obj(dmpl_vert,'./dmpl_ver.obj')
# dmplmodel.write_obj(standard_vert,'./std_ver.obj')