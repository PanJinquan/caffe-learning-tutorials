# -*-coding: utf-8 -*-
"""
    @Project: caffe-learning-tutorials
    @File   : 01-learning-lenet.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-17 15:11:38
"""


from pylab import *
import matplotlib.pyplot as plt
import sys
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

train_net_path = 'my_lenet/my_lenet_train.prototxt'
test_net_path = 'my_lenet/my_lenet_test.prototxt'
solver_config_path = 'my_lenet/my_lenet_solver.prototxt'
snapshot_prefix='my_lenet/my_lenet'
# 定义网络
def my_lenet(lmdb, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

# 定义solver
def my_solver(train_net_path, test_net_path, snapshot_prefix):
    '''
    功能：定义solver参数（用来优化），并写入solver_config_path。
        train_net_path：训练网络路径
        test_net_path：测试网络路径
        solver_config_path：solver路径
    '''
    s = caffe_pb2.SolverParameter() #结构化初始

    # 设置一个种子，这样可以重复实验
    # 这控制训练过程中的随机性
    s.random_seed = 0xCAFFE

    # 指明训练和测试网络的位置
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 500  # 500次训练迭代后，测试一下
    s.test_iter.append(100)  # 每次测试时测试100个
    s.max_iter = 10000  # 训练10000次结束

    s.base_lr = 0.01  # 初始学习率
    s.momentum = 0.9  #设置动量
    s.weight_decay = 5e-4 #设置weight decay，可预防过拟合

    s.lr_policy = 'inv'  #学习率如何变化的策略，还有'fixed'等
    s.gamma = 0.0001
    s.power = 0.75

    s.display = 1000 #每训练1000次就显示一下loss和accuracy


    s.snapshot = 250  #每训练250次就保存一下训练好的权重，
                      #这里为演示，设置有点小，一般为5000.
    s.snapshot_prefix = snapshot_prefix

    s.type = "SGD"  # 还可以选择 "SGD"、"Adam"、"Nesterov"等
    s.solver_mode = caffe_pb2.SolverParameter.GPU #设置为GPU模式
    return s

# 保存定义的网络结构
with open(train_net_path, 'w') as f:
    f.write(str(my_lenet('mnist_data/mnist_train_lmdb', 64)))

with open(test_net_path, 'w') as f:
    f.write(str(my_lenet('mnist_data/mnist_test_lmdb', 100)))

# 保存solver文件
with open(solver_config_path, 'w') as f:
    f.write(str(my_solver(train_net_path,test_net_path,snapshot_prefix)))

# 设置GPU，不是GPU环境的使用caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

# 载入solver文件
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(solver_config_path)
# [1]按照配置文件进行训练
# solver.solve()


# [2]自定义训练过程,可以方便获得loss等信息
niter = 250                # 迭代次数
test_interval = niter / 10 #测试间隔

# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    # 取一个batch_size的训练数据训练一下
    solver.step(1)

    # 保存每次迭代的loss值
    train_loss[it] = solver.net.blobs['loss'].data

    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward() #取一个batch_size的测试数据计算各层结果。
                                          #不像solver.step()，它没有反向传播，所以
                                          #不会改变权重，只是单纯地计算一下而已。
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

# 绘制loss acc等信息
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()