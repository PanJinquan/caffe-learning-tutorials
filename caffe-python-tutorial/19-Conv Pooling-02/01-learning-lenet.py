# -*-coding: utf-8 -*-
"""
    @Project: caffe-learning-tutorials
    @File   : 01-learning-lenet.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-17 15:11:38
"""

'''
# Solving in Python with LeNet
In this example, we'll explore learning with Caffe in Python, using the fully-exposed `Solver` interface.
### 1. Setup
# Set up the Python environment: we'll use the `pylab` import for numpy and plot inline.
'''
from pylab import *
import matplotlib.pyplot as plt
import sys
import caffe
# We'll be using the provided LeNet example data and networks
# (make sure you've downloaded the data and created the databases, as below).
# run scripts from caffe root
import os
# Download data下载mnist数据
# 在caffe根目录:sh data/mnist/get_mnist.sh
# Prepare data,生成lmdb数据
# 在caffe根目录:examples/mnist/create_mnist.sh
# back to examples

'''
### 2. Creating the net 
Now let's make a variant of LeNet, the classic 1989 convnet architecture.
We'll need two external files to help out:
* the net `prototxt`, defining the architecture and pointing to the train/test data
* the solver `prototxt`, defining the learning parameters
We start by creating the net. We'll write the net in a succinct and natural way as Python code 
that serializes to Caffe's protobuf model format.
This network expects to read from pregenerated LMDBs, but reading directly from `ndarray`s is also possible using `MemoryDataLayer`.

'''
from caffe import layers as L, params as P


def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
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


with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))
'''
The net has been written to disk in a more verbose but human-readable serialization format using Google's protobuf library.
You can read, write, and modify this description directly. Let's take a look at the train net.
Now let's see the learning parameters, which are also written as a `prototxt` file (already provided on disk). '
'We're using SGD with momentum, weight decay, and a specific learning rate schedule.
'''

'''
###3. Loading and checking the solver
* Let's pick a device and load the solver. We'll use SGD (with momentum),
 but other methods (such as Adagrad and Nesterov's accelerated gradient) are also available.
'''
caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
# To get an idea of the architecture of our net, we can check the dimensions of the intermediate features (blobs) and parameters
# (these will also be useful to refer to when manipulating data later).

# each output is (batch size, feature dim, spatial dim)
net_arch= [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print(net_arch)

# just print the weight sizes (we'll omit the biases)
net_para=[(k, v[0].data.shape) for k, v in solver.net.params.items()]
print(net_para)

# 可以采用以下函数print_net_shape,直接打印网络参数
def print_net_shape(net):
    # 查看每层的输出
    print "======data and diff output shape======"
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + ' out \t' + str(blob.data.shape)
        print layer_name + ' diff\t' + str(blob.diff.shape)

    print "======   weight and bias shape  ======"
    for layer_name, param in net.params.iteritems():
        print layer_name + ' weight\t' + str(param[0].data.shape), str(param[1].data.shape)
        print layer_name + ' diff  \t' + str(param[0].diff.shape), str(param[1].diff.shape)
print_net_shape(solver.net)

# Before taking off, let's check that everything is loaded as we expect.
# We'll run a forward pass on the train and test nets and check that they contain our data.
# 直接打印训练solver.net.forward() 或 solver.test_nets[0].forward()会输出最后一层的结果
print solver.net.forward()           # train net
print solver.test_nets[0].forward()  # test net (there can be more than one)
# 运行结果:{'loss': array(2.4669013, dtype=float32)}

# we use a little trick to tile the first eight images
# 获取net输入层的数据,并显示前8张图
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
plt.show()
print 'train labels:', solver.net.blobs['label'].data[:8]

plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
plt.show()
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]
'''
### 4. Stepping the solver
Both train and test nets seem to be loading data, and to have correct labels.
Let's take one step of (minibatch) SGD and see what happens.
Do we have gradients propagating through our filters? Let's see the updates to the first layer, 
shown here as a 4x5 grid of 5x5 filters.
'''
# 迭代一次,并显示conv1的diff图像
solver.step(1)
plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
plt.show()


'''
  下面是进行niter=200次迭代,并每隔25次进行一个测试
### 5. Writing a custom training loop
Something is happening. Let's run the net for a while, keeping track of a few things as it goes.
Note that this process will be the same as if training through the `caffe` binary. In particular:
* logging will continue to happen as normal
* snapshots will be taken at the interval specified in the solver prototxt (here, every 5000 iterations)
* testing will happen at the interval specified (here, every 500 iterations)

Since we have control of the loop in Python, we're free to compute additional things as we go, as we show below. We can do many other things as well, for example:
* write a custom stopping criterion
* change the solving process by updating the net in the loop
'''
# time
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    # 在conv1开始正向传递以避免加载新数据
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

# * Let's plot the train loss and test accuracy.
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()
'''
The loss seems to have dropped quickly and coverged (except for stochasticity), 
while the accuracy rose correspondingly. Hooray!
* Since we saved the results on the first test batch, 
we can watch how our prediction scores evolved. We'll plot 
time on the $x$ axis and each possible label on the $y$, 
with lightness indicating confidence.
'''


'''
We started with little idea about any of these digits, 
and ended up with correct classifications for each. 
If you've been following along,
 you'll see the last digit is the most difficult, 
 a slanted "9" that's (understandably) most confused with "4".

* Note that these are the "raw" output scores rather than 
the softmax-computed probability vectors. T
he latter, shown below, make it easier to see the confidence of our net 
(but harder to see the scores for less likely digits).
'''

for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
    plt.show()