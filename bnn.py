#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:07:49 2019

@author: rongzhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np

def Tanh():
    def nla():
        return nn.Tanh()
    return nla

def ReLU(inplace=True):
    def nla():
        return nn.ReLU(inplace)
    return nla
    
def LeakyReLU(negative_slope=1e-2, inplace=True):
    def nla():
        return nn.LeakyReLU(negative_slope, inplace)
    return nla

def PReLU(num_parameters=1, init=0.25):
    '''num_parameter = 1 or nChannels (learn 1 parameter for each channel)'''
    def nla():
        return nn.PReLU(num_parameters, init)
    return nla

def ELU(alpha=1., inplace=True):
    '''ELU(x) = max(0,x) + min(0,α∗(exp(x)−1))'''
    def nla():
        return nn.ELU(alpha, inplace)
    return nla

def pho_to_std(pho):
    if isinstance(pho, float):
        return math.log(1 + math.exp(pho))
    return (1+pho.exp()).log()

def std_to_pho(std):
    if isinstance(std, float):
        return math.log(math.exp(std) - 1)
    return (std.exp()-1).log()

class BayesLinearLayer(nn.Module):
    '''Bayesian linear layer'''
    def __init__(self, in_chans, out_chans, bias=False, prior_std=0.5):
        super(BayesLinearLayer, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.bias = bias
        self.prior_std = prior_std
        w_shape = (out_chans, in_chans)
        self.w_mean = Parameter(torch.randn(*w_shape)) # initialize by N(0,1)
        self.w_pho = Parameter(torch.Tensor(*w_shape).fill_(std_to_pho(prior_std)))
        if bias:
            b_shape = (out_chans, 1)
            self.b_mean = Parameter(torch.Tensor(*b_shape).fill_(0))
            self.b_pho = Parameter(torch.Tensor(*b_shape).fill_(std_to_pho(prior_std)))
        
    def forward(self, x, is_prob=True):
        '''x should be of size (N, in_chans), output will be (N, out_chans)
           y = W*x + b
        '''
        w_var = pho_to_std(self.w_pho)**2
        if self.bias:
            b_var = pho_to_std(self.b_pho)**2
        out = []
        for xi in x:
            xi = xi.unsqueeze(dim=1) # => (in_chans, 1)
            gamma = torch.mm(self.w_mean, xi)
            delta = torch.mm(w_var, xi**2)
            if self.bias:
                gamma += self.b_mean
                delta += b_var
            
            # local reparametrization trick
            zeta = torch.randn(gamma.size(), device=x.device) # (out_chans, 1)
            if is_prob:
                yi = gamma + delta.sqrt() * zeta # (out_chans, 1)
            else: # work as a normal linear layer with mean weights
                yi = gamma
            out.append(yi.squeeze(dim=1))
        
        return torch.stack(out, dim=0) # (N, out_chans)
    
    def extra_repr(self):
        s = ('{in_chans}, {out_chans}')
        if self.bias is True:
            s += ', bias=True'
        if self.prior_std != 0.5:
            s += ', prior_std={prior_std}'
        return s.format(**self.__dict__)
    
class BNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, bias=False, prior_std=0.5, likelihood_sd=5.0, nla=ReLU(True)):
        super(BNN, self).__init__()
        self.in_layer = BayesLinearLayer(n_in, n_hidden[0], bias, prior_std)
        self.inter_layers = nn.ModuleList([])
        for i in range(len(n_hidden)-1):
            self.inter_layers.append(BayesLinearLayer(n_hidden[i], n_hidden[i+1], bias, prior_std))
        self.out_layer = BayesLinearLayer(n_hidden[-1], n_out, bias, prior_std)
        self.relu = nla()
        self.prior_std = prior_std
        self.likelihood_sd = likelihood_sd
        
    def forward(self, x, is_prob=True):
        out = self.in_layer(x, is_prob)
        out = self.relu(out)
        for layer in self.inter_layers:
            out = self.relu(layer(out))
        out = self.out_layer(out, is_prob)
        
        return out
    
    @staticmethod
    def kl_div_p_q(p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        if isinstance(p_std, float):
            p_std = torch.tensor(p_std, dtype=torch.float32)
        if isinstance(q_std, float):
            q_std = torch.tensor(q_std, dtype=torch.float32)
        numerator = (p_mean - q_mean)**2 + \
            p_std**2 - q_std**2
        denominator = 2 * q_std**2 + 1e-8
        return torch.sum(
            numerator / denominator + torch.log(q_std) - torch.log(p_std))
    
    @staticmethod
    def _log_prob_normal(input, mu=0., sigma=1.):
        log_normal = - \
            math.log(sigma) - math.log(math.sqrt(2 * np.pi)) - \
            (input - mu)**2 / (2 * sigma**2)
        return torch.sum(log_normal)
    
    @staticmethod
    def likelihood_criterion(predictions, targets, likelihood_sd=5.0):
        '''predictions: (N, out_chans) float
           targets: (N, out_chans) float
           return value: scalar'''
        assert len(predictions) == len(targets)
        likelihood = 0
        for pred, tgt in zip(predictions, targets):
            likelihood += BNN._log_prob_normal(pred, tgt, likelihood_sd)
        return likelihood / len(predictions)
    
    def kl_new_prior(self):
        """KL divergence KL[params||prior] for a fully factorized Gaussian
           prior is given by prior_std at initialization of BNN
        """
        means = self.get_means()
        phos = self.get_phos()
        assert len(means) == len(phos)
        kl = 0
        for mean, pho in zip(means, phos):
            kl += self.kl_div_p_q(mean, pho_to_std(pho), 0.0, self.prior_std)
        
        return kl
    
    def loss(self, predictions, targets, weight_kl=1.):
        kl = self.kl_new_prior()
        log_p_D_given_w = self.likelihood_criterion(predictions, targets, self.likelihood_sd)
        return weight_kl * kl - log_p_D_given_w
        
    def kl_new_old(self):
        raise NotImplementedError
        
    def get_means(self):
        return [p[1] for p in self.named_parameters() if 'mean' in p[0]]
    
    def get_phos(self):
        return [p[1] for p in self.named_parameters() if 'pho' in p[0]]
        
    def loss_last_sample(self, inputs, targets):
        predictions = self.forward(inputs)
        loss = self.likelihood_criterion(predictions, targets, self.likelihood_sd)
        return loss

    # intrinsic reward
    # step_size 1e-2
    def kl_second_order_approx(self, step_size, inputs, targets):
        self.zero_grad()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        means = self.get_means()
        phos = self.get_phos()
        kl = 0
        for mu, pho in zip(means, phos): # Note: different mu/pho's have different shapes
            grad_mu = mu.grad
            grad_pho = pho.grad
            pho = pho.detach() # no need to retain computation graph
            H_mu = 1 / ((1+pho.exp()).log()) ** 2
            H_pho = ( (2*(2*pho).exp()) / (1+pho.exp())**2 ) * H_mu
            kl += torch.dot(grad_mu.pow(2).flatten(), 1 / H_mu.flatten())
            kl += torch.dot(grad_pho.pow(2).flatten(), 1 / H_pho.flatten())
        kl *= 0.5 * step_size**2

        return kl
    
    def get_parameters(self):
        return self.get_means + self.get_phos
        
    def get_gradient_flatten(self, inputs, targets):
        pass
    
    def get_diag_hessian(self, param=None):
        pass
    
    def second_order_update(self, step_size):
        raise NotImplementedError
        
    
        
    
    
    
        



            
        
            


