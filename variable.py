#coding:utf-8
'''
Variable
'''
import numpy as np
import torch
from torch.autograd import Variable
data1 = [[1,-2],[3,-5]]

#to numpy
data_numpy1 = np.array(data1)

#to torch

variable =  Variable(torch.IntTensor(data1),requires_grad = True)
variable1 =  Variable()# build a empty variable

v_out = torch.mean(variable*2)
v_out.backward()

print variable.grad
