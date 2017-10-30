#coding:utf-8
'''
save and restore model
'''
import numpy as np
import torch
import torch.utils.data as Data

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

BATCH_SIZE = 5

torch_dataset = Data.TensorDataset(data_tensor = x, target_tensor = y)#put data to dataset

loader = Data.DataLoader(             # implement batchsize training
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

for epoch in range(3):
    for step,(x_batch,y_batch) in enumerate(loader):
        print 'epoch = ',epoch, '|step = ', step, '|x_batch = ',x_batch.numpy(), '|y_batch = ' ,y_batch.numpy()

        print '*'*10
