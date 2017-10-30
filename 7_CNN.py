#coding:utf-8
'''
CNN
'''
import numpy as np
import torch
import torch.utils.data as Data
import torchvision #Dataset
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBAL_DEVICES'] = '0'

BATCH_SIZE = 128
LR = 0.01
EPOCH = 1

train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train = True,
    transform = torchvision.transforms.ToTensor(),#(0,255) -->(0,1)
    download = False
)

loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

test_dataset = torchvision.datasets.MNIST(root = './mnist',train = False)

# test_x = Variable(torch.unsqueeze(test_dataset.test_data,dim = 0), volatile = True).type(torch.FloatTensor)
test_x = Variable(torch.unsqueeze(test_dataset.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_dataset.test_labels[:2000]# the first 2000examples

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.out = torch.nn.Linear(32*7*7,10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)#platten
        output = self.out(x)
        return output


cnn = CNN()
optimizer = torch.optim.SGD(cnn.parameters(),lr = 0.01)
loss_func = torch.nn.CrossEntropyLoss()

# print cnn
#training
loss_ = []
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(loader):
        # print step
        b_x = Variable(b_x)
        b_y = Variable(b_y)

        predit_y = cnn(b_x)
        loss = loss_func(predit_y,b_y)
        # print loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_.append(loss.data)


        if step%50 == 0:
            test_out = cnn(test_x)
            predit_out = torch.max(test_out,1)[1].data.squeeze()
            accuracy = torch.sum(predit_out == test_y)/float(test_y.size(0))#convert to float. otherwise ,the result is 0
            print 'epoch = ', epoch, '|step =', step,  '|train loss = %.4f'%loss.data[0], '|accuracy = ', accuracy
loss_ = np.array(loss_)
plt.plot(loss_)
plt.show()
print 'done'

