# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:42:50 2020

@author: Haoran6
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tcn import TemporalConvNet

class DBS_lstm(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_stacks):
        super(DBS_lstm, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, num_stacks).double()
        self.hidden2dbs = nn.Linear(hidden_dim, 6890*3, bias=False).double()

    def forward(self, pose_beta_seq):
        num_frames = pose_beta_seq.shape[0]
        lstm_out, _ = self.lstm_layer(pose_beta_seq.view(num_frames, 1, -1))
        dbs = self.hidden2dbs(lstm_out).view(num_frames, 6890, 3)
        return dbs

class DBS_gru(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_stacks):
        super(DBS_gru, self).__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_dim, num_stacks).double()
        self.hidden2dbs = nn.Linear(hidden_dim, 6890*3, bias=False).double()

    def forward(self, pose_beta_seq):
        num_frames = pose_beta_seq.shape[0]
        gru_out, _ = self.gru_layer(pose_beta_seq.view(num_frames, 1, -1))
        dbs = self.hidden2dbs(gru_out).view(num_frames, 6890, 3)
        return dbs

class DBS_tcn(nn.Module):
    
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):
        super(DBS_tcn, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout).double()
        self.hidden2dbs = nn.Linear(num_channels[-1], output_size, bias=False).double()

    def forward(self, pose_beta_seq):
        """ Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len """
        num_frames = pose_beta_seq.shape[0]
        tcn_out = self.tcn(pose_beta_seq.transpose(0, 1).view(1,-1,num_frames)).transpose(1, 2).squeeze()
        dbs = self.hidden2dbs(tcn_out).view(num_frames, 6890, 3)
        return dbs
    

if __name__ == '__main__':
    
    # dbslayer = DBS_lstm(input_dim=82,hidden_dim=1024,num_stacks=3).cuda()
    
    # # pose_beta_seq = torch.rand(10,82).double().cuda()
    # # dbs_ver = dbslayer(pose_beta_seq=pose_beta_seq)
    
    # loss_function = nn.MSELoss()
    # optimizer = optim.SGD(dbslayer.parameters(), lr=0.01)
    # training_data_0 = [[torch.rand(100,82).double().cuda(),torch.rand(100,6890,3).double().cuda()] for i in range(10)]
    # training_data_1 = [[torch.rand(150,82).double().cuda(),torch.rand(150,6890,3).double().cuda()] for i in range(10)]
    # training_data = training_data_0 + training_data_1
    
    # for epoch in range(10):
    #     for pose_beta_seq, target_ver in tqdm(training_data):
    #         dbslayer.zero_grad()
            
    #         dbs_ver = dbslayer(pose_beta_seq)
            
    #         loss = loss_function(dbs_ver, target_ver)
    #         # print(loss.data)
    #         loss.backward()
    #         optimizer.step()
    #         torch.cuda.empty_cache()
    
    dbslayer = DBS_tcn(input_size=289, output_size=6890*3, num_channels=[512,768,1024]).cpu()
    pose_beta_seq = torch.rand(100,289).double().cpu()
    dbs_ver = dbslayer(pose_beta_seq=pose_beta_seq)
    