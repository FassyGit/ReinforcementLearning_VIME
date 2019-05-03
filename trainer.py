#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:39:03 2019

@author: rongzhao
"""

import torch
import numpy as np
import time

def timestr(form=None):
    if form is None:
        return time.strftime("<%Y-%m-%d %H:%M:%S>", time.localtime())
    if form == 'mdhm':
        return time.strftime('%m%d%H%M', time.localtime())

def batch_loader(X, y, batch_size, shuffle=True, drop_last=False):
    if shuffle:
        shuffled_indices = np.random.permutation(len(X))
    else:
        shuffled_indices = np.arange(len(X))
    mini_batch_index = 0
    num_remain = len(X)
    num_remain -= batch_size
    while num_remain >= 0:
        indices = shuffled_indices[mini_batch_index:mini_batch_index + batch_size]
        mini_batch_index += batch_size
        num_remain -= batch_size
        yield X[indices], y[indices]
    
    if not drop_last:
        if mini_batch_index < X.shape[0]:
            indices = shuffled_indices[mini_batch_index:]
            yield X[indices], y[indices]

class Trainer(object):
    '''A wrapper class encapsuling the training of a BNN for VIME'''
    def __init__(self, bnn, train_set, batch_size, optimizer, max_epoch, log_cfg, 
                 extra_w_kl=1.0, device='cuda', test_set=None, extra_arg=dict()):
        ''' train_set = (trainX, trainY) with len(trainX) == len(trainY)
            similar to test_set 
        '''
        self.model = bnn
        self.train_set = train_set
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = device
        self.test_set = test_set
        length = len(train_set[0])
        self.weight_kl = batch_size / len(train_set[0]) * extra_w_kl
        self.max_epoch = max_epoch
        self.log_cfg = log_cfg
        self.writer = self.log_cfg.get('writer', None)
        self.parse_extra_arg(extra_arg)
        
        self.start_epoch = 1
        self.model.to(device)
    
    def train(self):
        loss_all = []
        is_new_metric = False
        for epoch in range(1, self.max_epoch+1):
            loss_all.append(self.train_epoch())
            
            if epoch % self.log_cfg['display_interval'] == 0 or epoch == self.start_epoch:
                N = self.log_cfg['display_interval']
                loss_avg = np.array(loss_all[-N:]).mean()
                first_epoch = epoch if epoch == self.start_epoch else epoch+1-N
                print('%s Epoch %d ~ %d: loss = %.7f, current lr = %.7e' %
                      (timestr(), first_epoch, epoch, loss_avg, self._get_lr()))
            
            if epoch % self.log_cfg['snapshot_interval'] == 0 or epoch == self.start_epoch:
                self._snapshot(epoch)
            
            if epoch % self.log_cfg['val_interval'] == 0 or epoch == self.start_epoch:
                is_new_metric = True
                metric_train, kl_train = self.validate(self.train_set, is_prob=False)
                print('%s Epoch %d: metric_train = %.7f, kl_train = %.7f' %
                      (timestr(), epoch, metric_train, kl_train))
                if self.test_set:
                    metric_val, kl_val = self.validate(self.test_set, is_prob=False)
                    print('%s Epoch %d: metric_val = %.7f, kl_val = %.7f' %
                          (timestr(), epoch, metric_val, kl_val))
            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                self.writer.add_scalar('loss', loss_all[-1], epoch)
                if is_new_metric:
                    self.writer.add_scalar('train/metric', metric_train, epoch)
                    self.writer.add_scalar('train/kl', kl_train, epoch)
                    if self.test_set:
                        self.writer.add_scalar('val/metric', metric_val, epoch)
                        self.writer.add_scalar('val/kl', kl_val, epoch)
                    is_new_metric = False

        return loss_all
    
    def train_epoch(self):
        self.model.train()
        loss_arr = []
        for inputs, targets in batch_loader(self.train_set[0], self.train_set[1], self.batch_size):
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            predictions = self.model(inputs)
            self.optimizer.zero_grad()
            if self.external_criterion:
                loss = self.external_criterion(predictions, targets) + self.weight_kl*self.model.kl_new_prior()
            else:
                loss = self.model.loss(predictions, targets, self.weight_kl)
            loss.backward()
            self.optimizer.step()
            loss_arr.append(loss.item())
        return np.array(loss_arr).mean()
    
    def validate(self, dataset, is_prob=True):
        self.model.eval()
        if dataset is None:
            raise RuntimeError('Data set is empty.')
        metric_arr = []
        kl_arr = []
        with torch.no_grad():
            for inputs, targets in batch_loader(dataset[0], dataset[1], self.batch_size, shuffle=False):
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    targets = torch.tensor(targets, dtype=torch.float32)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs, is_prob)
                if self.external_criterion_val:
                    metric = self.external_criterion_val(predictions, targets) # + self.weight_kl*self.model.kl_new_prior()
                elif self.external_criterion:
                    metric = self.external_criterion(predictions, targets) # + self.weight_kl*self.model.kl_new_prior()
                else:
                    metric = self.model.loss(predictions, targets, 0.0)
                kl = self.model.kl_new_prior()
                metric_arr.append(metric.item())
                kl_arr.append(kl.item())
        return np.array(metric_arr).mean(), np.array(kl_arr).mean()
    
    def inference(self, inputs, is_prob=True):
        with torch.no_grad():
            predictions = self.model(inputs, is_prob)
        return predictions
    
    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
    
    def _snapshot(self, epoch, name=None):
        '''Take snapshot of the model, save to root dir'''
        root = self.log_cfg['root']
        state_dict = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if name is None:
            filename = '%s/state_%04d.pkl' % (root, epoch)
        else:
            filename = '%s/state_%s.pkl' % (root, name)
        print('%s Snapshotting to %s' % (timestr(), filename))
        torch.save(state_dict, filename)
    
    def parse_extra_arg(self, extra_arg):
        self.external_criterion = extra_arg.get('external_criterion', None)
        self.external_criterion_val = extra_arg.get('external_criterion_val', None)
    
    
    
    