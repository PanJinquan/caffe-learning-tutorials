#!/usr/bin/env python
# coding: utf-8
#copyRight by heibanke 
#如需转载请注明出处
#<<用Python做深度学习2-caffe>>
#http://study.163.com/course/courseMain.htm?courseId=1003491001

import caffe
import numpy as np
from pylab import *

solver = caffe.SGDSolver('hbk_mnist_solver.prototxt')

N = 1000
t = np.linspace(0, 2*np.pi, N)
x1 = np.array([t, 30+np.cos(t)+0.3*np.random.rand(N)])
x2 = np.array([t, 29+np.cos(t)+0.3*np.random.rand(N)])
y1 = np.zeros((N,1))
y2 = np.ones((N,1))
X = np.concatenate((x1.T, x2.T)).astype('float32')
y = np.concatenate((y1, y2)).astype('float32')


for i in range(2):
    X[:,i] = (X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))

idx = np.arange(len(y))
np.random.shuffle(idx)


# input graph
_, ax1 = subplots()

ax1.set_xlabel('t')
ax1.set_ylabel('x')

X_train = X[idx,:].reshape(X.shape[0],2,1,1)
y_train = y[idx].reshape(y.shape[0],1,1,1)

X_train = np.require(X_train,dtype='float32',requirements='C')
solver.net.set_input_arrays(X_train, y_train)
solver.test_nets[0].set_input_arrays(X_train, y_train)

for i in range(1001):
    solver.step(1)


x_min, x_max = np.min(X[:,0])-0.5, np.max(X[:,0])+0.5
y_min, y_max = np.min(X[:,1])-0.5, np.max(X[:,1])+0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
	                      np.arange(y_min, y_max, 0.02))

X_test = np.array([xx.ravel(), yy.ravel()]).T
X_test = np.require(X_test, dtype='float32', requirements='C').reshape(X_test.shape[0],2,1,1)
y_test = np.zeros((X_test.shape[0],1,1,1),dtype='float32')
solver.test_nets[0].set_input_arrays(X_test, y_test)

Z = np.zeros(xx.shape).ravel()
for i in range(int(X_test.shape[0]/50.0)):
    solver.test_nets[0].forward()
    data = solver.test_nets[0].blobs['ip2'].data
    label = np.argmax(data, axis=1)

    Z[i*50: (i+1)*50] = label


ax1.contourf(xx,yy,Z.reshape(xx.shape), levels=[0,0.5,1],colors=('r','g'))
ax1.scatter(X[0:N,0],X[0:N,1],c='r')
ax1.scatter( X[N:,0],X[N:,1],c='b')
ax1.set_xticks([])
ax1.set_yticks([])
_.savefig('output.png')
