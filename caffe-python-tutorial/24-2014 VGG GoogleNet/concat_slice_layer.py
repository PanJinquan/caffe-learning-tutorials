#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe
from caffe import layers as L
from caffe import params as P
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# 配置文件路径和文件名
path=''             
net_file = path+'concat_slice_py.prototxt'


def concat_slice_net():
    n = caffe.NetSpec()
    n.data = L.DummyData(dummy_data_param=dict(num=20,channels=50,height=64,width=64,data_filler=dict(type="gaussian")))
    # 将输入的data层分为a,b,c输出,slice_point比Slice的个数少1
    # 如本例将输入的data层分为a,b,c输出,即top有三个,slice_point则有两个,
    # 其中第一个slice_point=20是top:"a"的个数,第二个slice_point=30是top:"b"+top:"a"的个数
    # 而top:"c"的个数:channels-第二个slice_point=50-30=20,
    # 因此a,b,c的channels分别是:20,10,20
    n.a, n.b,n.c = L.Slice(n.data, ntop=3, slice_point=[20,30],axis=0)
    n.d = L.Concat(n.a,n.b,axis=0)

    # Eltwise层的操作有三个：product（点乘）， sum（相加减） 和 max（取大值），其中sum是默认操作
    n.e = L.Eltwise(n.a,n.c)

    return n.to_proto()


def draw_net(net_file, jpg_file):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(), net)
    caffe.draw.draw_net_to_file(net, jpg_file, 'BT')


with open( net_file, 'w') as f:
    f.write(str(concat_slice_net()))


draw_net(net_file, "a.jpg")




