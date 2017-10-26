#coding:utf-8
'''
Activition and plot
'''
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

data = torch.ones(100,2)
x0 = torch.normal(2*data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*data,1)
y1 = torch.ones(100)
# t = [i for i in range(200)]
# plt.scatter(t, x0.numpy())
# plt.scatter(t, x1.numpy())
# plt.show()
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),0).type(torch.LongTensor)

x,y = Variable(x),Variable(y)


# plt.scatter(x_data,y_data,c = 'orange')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predit = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        y = self.predit(x)##
        return y

net = Net(2,10,2)
# print net
# a = net(x)
# print a.size()

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.CrossEntropyLoss()##loss function

for t in range(100):
    out = net(x)
    # print '*'*10
    # print predition.size()
    # print y.size()
    loss = loss_func(out,y)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2 == 0:
        plt.cla()
        predition = torch.max(F.softmax(out),1)[1]
        pred_y = predition.data.numpy()
        tart_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1],c = pred_y, s = 100, lw = 0 )
        accuracy = sum(pred_y == tart_y)/200
        plt.text(1.5,-4,'accuracy = %.4f'%accuracy,fontdict = {'size':20, 'color' :'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()