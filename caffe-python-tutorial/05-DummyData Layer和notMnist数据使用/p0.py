#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001


import caffe
from caffe import layers as L

def net():
    n = caffe.NetSpec()
    n.data=L.DummyData(dummy_data_param=dict(num=10, channels=1, height=28, width=28, data_filler=dict(type='gaussian')))
    n.label=L.DummyData(dummy_data_param=dict(num=10, channels=1, height=1, width=1, data_filler=dict(type='gaussian')))
    n.ip1 = L.InnerProduct(n.data, num_output=50, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=4, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    return n.to_proto()


with open('dm_py.prototxt', 'w') as f:
    f.write(str(net()))


#载入solver文件
solver = caffe.SGDSolver('dm_solver.prototxt')

#solver.net.forward()
#solver.step(1)
#solver.solve()

print solver.net.blobs['data'].data.shape
print solver.net.blobs['label'].data.shape
