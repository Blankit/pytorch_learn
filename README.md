# 记录Pytorch的学习过程
- 学习资料是在b站上的莫烦的Pytorch教程
https://www.bilibili.com/video/av10495320/index_7.html?t=625#page=1
---
### Day 1（2017/10/25）
- 了解Pytorch中的基本语法，以及常用的包
- 实现的函数  
&emsp; **torch_numpy** 实现numpy与torch中张量间数据的转换  
&emsp; **numpy --> torch** torch_data.numpy()  
&emsp; **torch --> numpy** torch.from_numpy(numpy_data)

---
### Day 2（2017/10/26）

1. Python转换成Torch数据
1. Torch中的数学计算
2. 激活函数
1. 变量及自动求解梯度
1. LR实现
1. 分类问题

- **Python转换成Torch数据**

```
torch_data = torch.FloatTensor(python_data)
torch_data = torch.IntTensor(data1)
```

- **Torch中的数学计算**  
&emsp; 加（add）,矩阵乘（mm）以及比较特殊的dot(对应位置上元素相乘后相加)

- **激活函数**  
&emsp; 主要是导入torch.nn中的functional来实现  
&emsp; 涉及到子图以及添加图例
- **变量及自动求解梯度**   
&emsp; Pytorch在求解梯度的过程中，都是通过变量实现的（需要在设置变量时定义为可自动求解）

- **LR实现**  
&emsp; 包括数据的定义，网络的搭建，损失函数的定义，训练方法以及结果的可视化
  - 数据的定义  
  定义为变量形式。torch.unsqueeze()？
  - 网络的搭建  
  搭建的是一个线性的网络，使用torch.nn.Module中的torch.nn.Linear(in_features, out_features, bias=True)　　

    网络的输入: (N,∗,in_features) where * means any number of additional dimensions  

    网络的输出: (N,∗,out_features) where all but the last dimension are the same shape as the input.
  - 损失函数的定义
  　
  - 训练方法
  
  - 结果的可视化
  
 check 
 ```
 strings /home/zxf/anaconda2/lib/python2.7/site-packages/torch/lib/../../../../libstdc++.so.6 | grep GLIBCXX 
 ```
 install 
 ```
 conda install libgcc 
```
