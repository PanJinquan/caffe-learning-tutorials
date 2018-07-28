# -*- coding:utf-8 -*-

import sys
import numpy as np
sys.path.append('/path/to/caffe/python')
import caffe
import glob
import shutil
import os

WEIGHTS_FILE = 'models/letnet5_regression/letnet5_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'config/letnet5_regression/deploy.prototxt'
test_files='dataset/test.txt'
mean_file='dataset/mean/lenet_train_mean.txt'

MEAN_VALUE = []
with open(mean_file) as f:
    ls=f.readlines()
    for l in ls:
        num=float(l)
        MEAN_VALUE.append(num)
#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.array([MEAN_VALUE]))
# transformer.set_raw_scale('data', 255)
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension将图像通道移动到最外层
transformer.set_mean('data', np.array(MEAN_VALUE))            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
batch_size = net.blobs['data'].data.shape[0]
print("batch_size=",batch_size)

# 读取测试样本
image_dir='dataset/images'
image_list=[]
with open(test_files) as f:
    for line in f.readlines():
        name = line[:-1]
        image_list.append(name)
labels_num=2
test_len=10
test_len=min(test_len,len(image_list))

labels = np.zeros([test_len,labels_num], dtype=np.float32)
score = np.zeros([test_len,labels_num], dtype=np.float32)
for i in range(len(image_list[0:test_len])):
    filename=image_list[i]
    image = caffe.io.load_image(os.path.join(image_dir,filename))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0, ...] = transformed_image

    output = net.forward()
    freqs = output['score']
    pre=freqs[0]
    print('Predicted frequencies for %s is %3.4f and %3.4f '%(filename, pre[0], pre[1]))
    # print('Predicted frequencies for %s is %f '%(filename, pre[0]))
    score[i]=pre
    label1=float(filename.split('_')[1])
    label2=float(filename.split('_')[2].split('.jpg')[0])
    labels[i]=[label1,label2]

diff2 = score - labels
diff = diff2
loss = np.sum(diff ** 2) / test_len / 2.
print('loss=%f' % (loss))
