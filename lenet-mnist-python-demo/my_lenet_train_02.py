# -*-coding: utf-8 -*-
"""
    @Project: caffe-learning-tutorials
    @File   : my_lenet_train.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-25 14:28:13
    @desc   :
     [1]solver.net.forward() 和 solver.test_nets[0].forward() 是将batch_size个图片送到网络中去，只有前向传播(Forward Propagation，BP)
     solver.net.forward()作用于训练集，solver.test_nets[0].forward() 作用于测试集，一般用于获得测试集的正确率。
     [2]solver.step(1) 也是将batch_size个图片送到网络中去，不过 solver.step(1) 不仅有FP，而且还有反向传播(Back Propagation，BP)！
     这样就可以更新整个网络的权值(weights)， 同时得到该batch的loss。
     url:https://blog.csdn.net/u012762410/article/details/78917540
"""

from pylab import *
import caffe
import matplotlib.pyplot as plt

# 设置GPU，不是GPU环境的使用caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

solver_path='mnist/lenet_auto_solver.prototxt'
solver =caffe.SGDSolver(solver_path)

# 查看一下网络的层次：
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

# train net(只是前向转播)
solver.net.forward()
# 查看第一层“data”中batch图像中的第一个图像是哪个数字
data_layer = solver.net.blobs['data']
print(data_layer.data.shape)
image=data_layer.data[0,0]
plt.imshow(image, cmap='gray');
plt.show()