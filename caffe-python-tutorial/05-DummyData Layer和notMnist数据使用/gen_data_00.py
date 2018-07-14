#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import os
import cv2
import numpy as np
import pdb

def write_img_list(data, filename):
    with open(filename, 'w') as f:
        for i in xrange(len(data)):
            f.write(data[i][0]+' '+str(data[i][1])+'\n')


image_size = 28
s='ABCDEFGHIJ'

filedir='/home/hbk/caffe/data/notMNIST/notMNIST_small/'


# 1. read file
filedir2 = os.listdir(filedir)

datasets=[]
data=[]
for subdir in filedir2:
    if os.path.isdir(filedir+subdir):
        files=os.listdir(filedir+subdir)
        dataset = np.ndarray(shape=(len(files), image_size, image_size),
                         dtype=np.float32)
        
        num_image = 0
        for file in files:
            if file[-3:]=='png':
                tmp=cv2.imread(filedir+subdir+'/'+file,cv2.IMREAD_GRAYSCALE)
                #判断图像大小是否符合要求，不符合则跳过
                try:
                    if tmp.shape==(image_size,image_size):
                        datasets.append((filedir+subdir+'/'+file, s.rfind(subdir)))
                        data.append(tmp)
                        num_image+=1
                    else:
                        print subdir,file,tmp.shape
                except:
                    print subdir,file,tmp
            else:
                print file

#随机化数据序列，计算均值
np.random.shuffle(datasets)
print np.mean(np.array(data))

TRAIN_NUM = 4*len(datasets)/5

write_img_list(datasets[0:TRAIN_NUM], 'train00.imglist')
write_img_list(datasets[TRAIN_NUM:], 'test00.imglist')


