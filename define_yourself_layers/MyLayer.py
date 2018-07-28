# -*-coding: utf-8 -*-
"""
    @Project: caffe-learning-tutorials
    @File   : MyLayer.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-26 10:59:26
    @desc   :
"""
import sys
import caffe
import numpy as np
import yaml
import cv2

class MyLayer(caffe.Layer):
    '''
    类直接继承caffe.Layer，必须重写setup()，reshape()，forward()，backward()函数，其他的函数可以自己定义
    setup()  是类启动时该做的事情，比如层所需数据的初始化。
    reshape()就是取数据然后把它规范化为四维的矩阵。每次取数据都会调用此函数。
    forward()就是网络的前向运行，这里就是把取到的数据往前传递，因为没有其他运算。
    backward()就是网络的反馈，data层是没有反馈的
    url:  https://blog.csdn.net/thesby/article/details/51264439
    '''
    def setup(self, bottom, top):
        self.num = yaml.load(self.param_str)["num"]
        print "Parameter num : ", self.num

    def reshape(self, bottom, top):
        pass

    # 前向转播
    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        print bottom[0].data.shape
        print bottom[0].data
        top[0].data[...] = bottom[0].data + self.num# 实现输入数据加上num
        print top[0].data[...]

    # 反向转播
    def backward(self, top, propagate_down, bottom):
        pass