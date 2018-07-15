#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

#import os, sys
#CAFFE_HOME = '.../caffe/'
#sys.path.insert(0, CAFFE_HOME + 'caffe/python')

import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P

def net(dbfile, batch_size, mean_value=0):
    n = caffe.NetSpec()
    # 定义带数据和标签的数据输入层
    n.data, n.label=L.Data(source=dbfile, backend = P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(scale=0.00390625))
    # 定义全连接层
    n.ip1 = L.InnerProduct(n.data, num_output=500, weight_filler=dict(type='xavier'))
    # 定义激活函数
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # 定义全连接层
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    # 生成误差损失层
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    # 添加一个精确层
    n.accu = L.Accuracy(n.ip2, n.label, include={'phase':caffe.TEST})
    # 返回网络结构
    # return n.to_proto()
    return n

#若返回前，已经n.to_proto()，则可以使用下面的方式保存网络结构：
# with open( 'auto_train00.prototxt', 'w') as f:
#     f.write(str(net( '/home/hbk/caffe/examples/mnist/mnist_train_lmdb', 64)))
# with open('auto_test00.prototxt', 'w') as f:
#     f.write(str(net('/home/hbk/caffe/examples/mnist/mnist_test_lmdb', 100)))

# 若没有to_proto()，则需要to_proto()后才能保存，注意write时，需要将to_proto()的结果转为字符串的形式
with open('train.prototxt','w') as f:
    dbfile='/home/hbk/caffe/examples/mnist/mnist_train_lmdb'
    batch_size=64
    mean_value = 0
    train_net=net(dbfile=dbfile,batch_size=batch_size,mean_value=mean_value)
    dd=train_net.to_proto()
    f.write(str(train_net.to_proto()))

solver = caffe.SGDSolver("hbk_mnist_solver_py.prototxt")
solver.net.forward()
solver.test_nets[0].forward()

solver.step()

#按照配置文件进行训练
#solver.solve()


