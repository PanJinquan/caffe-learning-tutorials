# -*-coding: utf-8 -*-
"""
    @Project: define_yourself_layers
    @File   : 01-learning-lenet.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-17 15:11:38
"""
import sys
import caffe
import numpy as np                                                                                                                                                       
import yaml
import cv2   

net = caffe.Net('conv.prototxt',caffe.TEST)
#im = np.array(Image.open('timg.jpeg'))
im = np.array(cv2.imread('timg.jpeg'))
print im.shape
#im_input = im[np.newaxis, np.newaxis, :, :]
im_input = im[np.newaxis, :, :]
print im_input.shape
#print im_input.transpose((1,0,2,3)).shape
im_input2 = im_input.transpose((0,3,1,2))
print im_input2.shape
#print im_input.shape
net.blobs['data'].reshape(*im_input2.shape)
net.blobs['data'].data[...] = im_input2
net.forward()



