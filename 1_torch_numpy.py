#coding:utf-8
'''
data convert between numpy and torch
'''
import numpy as np
import torch


#numpy to torch

numpy_data = np.arange(6).reshape(2,3)
torch_data = torch.from_numpy(numpy_data)
print numpy_data
print torch_data

#torch to numpy
data = torch_data.numpy()
print data
