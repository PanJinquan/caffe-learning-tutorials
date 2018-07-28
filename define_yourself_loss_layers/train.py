# -*-coding: utf-8 -*-
"""
    @Project: caffe-learning-tutorials
    @File   : train.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-17 15:11:38
"""


from pylab import *
import matplotlib.pyplot as plt
import sys
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

solver_config_path = 'config/letnet5_regression/solver.prototxt'

# 设置GPU，不是GPU环境的使用caffe.set_mode_cpu()
caffe.set_device(1)
caffe.set_mode_gpu()

# 载入solver文件
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(solver_config_path)
# [1]按照配置文件进行训练
solver.solve()

##########################################################
# # [2]自定义训练过程,可以方便获得loss等信息
# niter = 250                # 迭代次数
# test_interval = niter / 10 #测试间隔
#
# # losses will also be stored in the log
# train_loss = zeros(niter)
# test_acc = zeros(int(np.ceil(niter / test_interval)))
#
# # the main solver loop
# for it in range(niter):
#     # 取一个batch_size的训练数据训练一下
#     solver.step(1)
#
#     # 保存每次迭代的loss值
#     train_loss[it] = solver.net.blobs['loss'].data
#
#     if it % test_interval == 0:
#         print 'Iteration', it, 'testing...'
#         correct = 0
#         for test_it in range(100):
#             solver.test_nets[0].forward() #取一个batch_size的测试数据计算各层结果。
#                                           #不像solver.step()，它没有反向传播，所以
#                                           #不会改变权重，只是单纯地计算一下而已。
#             correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
#                            == solver.test_nets[0].blobs['label'].data)
#         test_acc[it // test_interval] = correct / 1e4
#
# # 绘制loss acc等信息
# _, ax1 = subplots()
# ax2 = ax1.twinx()
# ax1.plot(arange(niter), train_loss)
# ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# ax2.set_ylabel('test accuracy')
# ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
# plt.show()