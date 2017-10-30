#coding:utf-8
'''
Activition and plot
'''
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-2,2,100),dim=1)#x.size() = (100L,1L)# unsqueeze可以按行，也可以按列展开
y = pow(x,2) + 2*x + 0.2 * torch.rand(x.size())

x,y = Variable(x),Variable(y)
x_data = x.data.numpy()
y_data = y.data.numpy()

# plt.scatter(x_data,y_data,c = 'orange')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net, self).__init__()
        #按这样可以搭建多层的网络
        self.hidden = torch.nn.Linear(n_feature,n_hidden)#构建隐藏层，输入维度（N,n_feature）,输出维度（N,n_hidden）
        self.predit = torch.nn.Linear(n_hidden,n_output)#构建输出层

    def forward(self, x):
        x = F.relu(self.hidden(x))#对隐藏层的输出做非线性变换
        y = self.predit(x)##
        return y

net = Net(1,10,1)#构建网络
# print net
# a = net(x)
# print a.size()
#
plt.ion()#打开交互模式
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)#使用SGD训练
loss_func = torch.nn.MSELoss()#损失函数是MSE

for t in range(200):
    predition = net(x)
    # print '*'*10
    # print predition.size()
    # print y.size()
    loss = loss_func(predition,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 == 0:
        plt.cla()#Clear the current axes
        plt.scatter(x.data.numpy(),y.data.numpy())#画散点图
        plt.plot(x.data.numpy(),predition.data.numpy(),'r-',lw = 5)#拟合的图像
        plt.text(0.5,0,'loss = %.4f'%loss.data[0],fontdict = {'size':20, 'color' :'red'})#加文本描述
        plt.pause(0.1)#

plt.ioff()#关闭交互模式
plt.show()
