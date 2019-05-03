#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:46:25 2019

@author: rongzhao
"""

import torch
import torch.nn as nn
import numpy as np
import bnn
from tensorboardX import SummaryWriter
from trainer import Trainer, batch_loader, timestr
from data_generation import regress_1d_data
import os
import matplotlib.pyplot as plt

# structure of BNN
n_in = 4
n_hidden = [32]
n_out = 1

bias = True
prior_std = 0.5
likelihood_ds = 5.0

lr = 0.1
extra_w_kl = 0.01
batch_size = 100
device = 'cuda'
max_epoch = 500
dataset_size = 1000

train_set = regress_1d_data(dataset_size, n_in, n_out, [-np.pi, np.pi])
test_set = None

model = bnn.BNN(n_in, n_hidden, n_out, bias, prior_std, likelihood_ds, nla=bnn.ReLU())
experiment_id = 'Reg1D_ds%d_%dhidden_w%d_relu_withKL_MSE_extra_w_kl%.2f_%s' % \
(dataset_size, len(n_hidden), n_hidden[0], extra_w_kl, timestr('mdhm'))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#trainX = [(1,1,1,1,9)] # inputs of BNN: list of (s_t, a_t)
#trainY = [(2,2,2,2)] # targets of BNN: list of (s_{t+1})
#train_set = (trainX, trainY)

log_cfg = {
        'root': './snap/%s' % experiment_id, 
        'display_interval': 20,
        'val_interval': 100,
        'snapshot_interval': 100,
        'writer': SummaryWriter(log_dir='./tboard/%s' % (experiment_id)),
        }
os.makedirs(log_cfg['root'], exist_ok=True)

extra_arg = {
        'external_criterion': nn.MSELoss(),
        }

trainer = Trainer(model, train_set, batch_size, optimizer, max_epoch, log_cfg, 
                  extra_w_kl, device, test_set, extra_arg)
trainer.train()

test_set = regress_1d_data(1000, n_in, n_out, [-np.pi*1.5, np.pi*1.5])
testX, testY = test_set
predictions1, predictions2 = [], []
for xi, yi in batch_loader(testX, testY, 1, False):
    xi = torch.tensor(xi, dtype=torch.float32, device=device)
    yi = torch.tensor(yi, dtype=torch.float32, device=device)
    preds1 = trainer.inference(xi, False)
    preds2 = trainer.inference(xi, True)
    predictions1.append(preds1.squeeze(dim=0).item())
    predictions2.append(preds2.squeeze(dim=0).item())
plt.plot(testX[:,0], testY, testX[:,0], predictions1)
plt.plot(testX[:,0], predictions2)





