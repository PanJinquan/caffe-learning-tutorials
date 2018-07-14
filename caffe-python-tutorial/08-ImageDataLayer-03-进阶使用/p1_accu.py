#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe

from pylab import *
from caffe import layers as L

imgdata_mean = 108

def net(img_list, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.ImageData(source=img_list,batch_size=batch_size,new_width=28,new_height=28,ntop=2,transform_param=dict(scale=0.00390625, mean_value=mean_value))
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.accu = L.Accuracy(n.ip2, n.label)
    return n.to_proto()

with open('auto_train00.prototxt', 'w') as f:
    f.write(str(net('train00.imglist', 200, imgdata_mean)))
with open('auto_test00.prototxt', 'w') as f:
    f.write(str(net('test00.imglist', 50, imgdata_mean)))


solver = caffe.SGDSolver('auto_solver00_step.prototxt')

solver.net.forward()

niter = 301
plot_interval = 10
train_loss = zeros(niter)
test_acc = zeros(niter)
train_acc = zeros(niter)

# The main solver loop
for it in range(niter):
    solver.step(10)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    train_acc[it] = solver.net.blobs['accu'].data
    test_acc[it] = solver.test_nets[0].blobs['accu'].data



# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(arange(niter), test_acc, 'r', arange(niter), train_acc, 'm')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('accuracy')
_.savefig('converge01.png')

