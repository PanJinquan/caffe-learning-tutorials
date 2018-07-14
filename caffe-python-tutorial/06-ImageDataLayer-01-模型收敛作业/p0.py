#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe

from pylab import *
from caffe import layers as L

def net(img_list, batch_size, mean_value=0):
    n = caffe.NetSpec()
    n.data, n.label=L.ImageData(source=img_list,batch_size=batch_size,new_width=28,new_height=28,ntop=2, transform_param=dict(scale=1.0/255.0))
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open( 'auto_train00.prototxt', 'w') as f:
    f.write(str(net( 'train00.imglist', 200)))
with open( 'auto_test00.prototxt', 'w') as f:
    f.write(str(net( 'test00.imglist', 50)))


#solver = caffe.SGDSolver( 'auto_solver00.prototxt')
solver = caffe.get_solver( 'auto_solver00.prototxt')
#按照配置文件进行训练
#solver.solve()

iternum = 1000
loss_iter = np.zeros(iternum)
for it in range(iternum):
    solver.step(1)  # SGD by Caffe
    loss_iter[it] = solver.net.blobs['loss'].data

print loss_iter