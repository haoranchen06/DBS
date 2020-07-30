# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:42:50 2020

@author: Haoran6
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DBS_lstm(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DBS_lstm, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim).double()
        self.hidden2ver = nn.Linear(hidden_dim, 6890*3).double()

    def forward(self, pose_beta_seq):
        num_frames = pose_beta_seq.shape[0]
        lstm_out, _ = self.lstm(pose_beta_seq.view(num_frames, 1, -1))
        ver = self.hidden2ver(lstm_out.view(num_frames, -1)).view(num_frames, 6890, 3)
        return ver

if __name__ == '__main__':
    
    dbslayer = DBS_lstm(input_dim=82,hidden_dim=1024).cuda()
    
    # pose_beta_seq = torch.rand(10,82)
    # dbs_ver = dbslayer(pose_beta_seq=pose_beta_seq)
    
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(dbslayer.parameters(), lr=0.01)
    training_data_0 = [[torch.rand(100,82).double().cuda(),torch.rand(100,6890,3).double().cuda()] for i in range(10)]
    training_data_1 = [[torch.rand(150,82).double().cuda(),torch.rand(150,6890,3).double().cuda()] for i in range(10)]
    training_data = training_data_0 + training_data_1
    
    for epoch in range(10):
        for pose_beta_seq, target_ver in tqdm(training_data):
            dbslayer.zero_grad()
            
            dbs_ver = dbslayer(pose_beta_seq)
            
            loss = loss_function(dbs_ver, target_ver)
            # print(loss.data)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()