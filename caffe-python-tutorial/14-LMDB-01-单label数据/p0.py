#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe

from pylab import *
from caffe import layers as L
from caffe import params as P
imgdata_mean = 20

def net(dbfile, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.Data(source=dbfile, backend = P.Data.LMDB, batch_size=batch_size, ntop=2, transform_param=dict(scale=1.0/30.0, mean_value=mean_value))
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=4, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open( 'auto_train00.prototxt', 'w') as f:
    f.write(str(net( 'hbk_lmdb_train', 200, imgdata_mean)))
with open('auto_test00.prototxt', 'w') as f:
    f.write(str(net('hbk_lmdb_test', 50, imgdata_mean)))

solver = caffe.SGDSolver('auto_solver00.prototxt')


niter = 2000
test_interval = 100
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
print len(test_acc)


# The main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='data')


    if it % test_interval == 0:
        correct = 0
        data = solver.test_nets[0].blobs['ip2'].data
        label = solver.test_nets[0].blobs['label'].data
        for test_it in range(100):
            solver.test_nets[0].forward()
            # Positive values map to label 1, while negative values map to label 0
            for i in range(len(data)):
                    if np.argmax(data[i]) == label[i]:
                        correct += 1

        test_acc[int(it / test_interval)] = correct * 1.0 / (len(data)  * 100)
        print 'Iteration', it, 'testing accuracy is ', str(correct * 1.0 / (len(data)  * 100))
# output graph
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
_.savefig('converge00.png')

