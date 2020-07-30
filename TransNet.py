# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:09:28 2020

@author: Haoran6
"""

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from smpl_torch import SMPLModel
from dmpl_torch import DMPLModel
import h5py



class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        # self.device = torch.device('cuda')
        self.trans = nn.Parameter(torch.Tensor(3).type(torch.float64).to(torch.device('cuda')))
    def forward(self, input_vertice):
        return input_vertice + self.trans
        

if __name__ == '__main__':
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdata = np.load("./DFaust_67/50002/50002_chicken_wings_poses.npz")
    fId = 100
    pose_body = torch.Tensor(bdata['poses'][fId, 3:72]).squeeze().type(torch.float64).to(comp_device)
    pose_body = torch.cat((torch.zeros(3).type(torch.float64).to(comp_device),pose_body),0)
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64).to(comp_device)
    dmpls = torch.Tensor(bdata['dmpls'][fId]).type(torch.float64).to(comp_device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(comp_device)
    smplmodel = SMPLModel(device=comp_device)
    dmplmodel = DMPLModel(device=comp_device)
    simu_vert = smplmodel(betas, pose_body, trans)
    with h5py.File('./dyna_registrations/dyna_male.h5', 'r') as f:
        verts = f['50002_chicken_wings'].value.transpose([2, 0, 1])
    standard_vert = torch.Tensor(verts[fId]).type(torch.float64).to(comp_device)
        
    transnet = TransNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(transnet.parameters(), lr=0.01)
    
    for t in range(1500):
        
        # zero the parameter gradients
        optimizer.zero_grad()
            
        # forward + backward + optimize
        output = transnet(simu_vert)
        loss = criterion(output, standard_vert)
        loss.backward()
        optimizer.step()
        
        # print(loss.item())
    print('Finished Training')
    translation = transnet.trans.data
    print(translation)
    print(torch.mean(standard_vert-simu_vert,0))
    


