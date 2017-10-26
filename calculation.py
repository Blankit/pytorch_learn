#coding:utf-8
'''
basic calculation
'''
import numpy as np
import torch
#abs
data1 = [[1,-2],[3,-5]]
data2 = [[2,3],[-3,-5]]
#to numpy
data_numpy1 = np.array(data1)
data_numpy2 = np.array(data2)

# torch_data1 = torch.FloatTensor(data1)
# torch_data2 = torch.FloatTensor(data2)

#to torch
torch_data1 = torch.from_numpy(data_numpy1)
torch_data2 = torch.from_numpy(data_numpy2)
data_abs = torch.abs(torch_data1)

print torch_data1
print torch_data2
#
# print data_abs
torch_add = torch.add(torch_data1,torch_data2)
print torch_add
torch_mul = torch.mm(torch_data1,torch_data2)
print torch_mul
torch_dot = torch.dot(torch.IntTensor([1,2]),torch.IntTensor([4,1]))
print torch_dot