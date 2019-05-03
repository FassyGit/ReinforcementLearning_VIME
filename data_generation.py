#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:25:36 2019

@author: rongzhao
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def regress_1d_data(length, n_in, n_out, limits=(-2*np.pi,2*np.pi)):
    dataX = []
    dataY = []
    for x in np.linspace(*limits, length):
        dataX.append([x**(i+1) for i in range(n_in)])
        dataY.append([np.sin(x)**(i+1) for i in range(n_out)])
    
    return np.array(dataX), np.array(dataY)
    





