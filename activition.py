#coding:utf-8
'''
Activition and plot
'''
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

import matplotlib.pyplot as plt

x = torch.linspace(-5,5,200)
x_variable = Variable(x)
x_data = x_variable.data.numpy()

y_relu = F.relu(x)#torch.autograd.variable.Variable
y_relu_data = y_relu.data.numpy()
y_sigmoid_data = F.sigmoid(x).data.numpy()
y_tanh_data = F.tanh(x).data.numpy()
y_softplus_data = F.softplus(x).data.numpy()
# print type(x)
# print type(x_data)
# print type(y_relu)

plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.plot(x_data,y_relu_data,c = 'red',label = 'relu')
plt.ylim(-1,5)#set range
plt.legend(loc = 'best')

plt.subplot(222)
plt.plot(x_data,y_sigmoid_data,label = 'sigmoid')
plt.ylim(-0.2,1.2)
plt.legend(loc = 'best')

plt.subplot(223)
plt.plot(x_data,y_tanh_data,c = 'black',label = 'tanh')
plt.ylim(-1.2,1.2)
plt.legend(loc = 'best')


plt.subplot(224)
plt.plot(x_data,y_softplus_data,c = 'orange',label = 'softplus')
plt.ylim(-0.2,5.2)
plt.legend(loc = 'best')

plt.show()

