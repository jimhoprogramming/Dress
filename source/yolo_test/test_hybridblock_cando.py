# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon
from mxnet import nd
from matplotlib import pyplot as plt
from mxnet import autograd
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet import init

def make_data_label():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)    
    return features,labels

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的。
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take 函数根据索引返回对应元素。

def squared_loss(y_hat, y):  # 本函数已保存在 gluonbook 包中方便以后使用。
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  # 本函数已保存在 gluonbook 包中方便以后使用。
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def linreg(X, w, b):  # 本函数已保存在 gluonbook 包中方便以后使用。
    return nd.dot(X, w) + b    

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(256)
            self.fc2 = gluon.nn.Dense(128)
            self.fc3 = gluon.nn.Dense(1)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        #print('type(x): {}, F: {}'.format(type(x).__name__, F.__name__))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def predict_Net():
    net = Net()
    net.hybridize()
    net.collect_params().initialize()
    x = nd.random_normal(shape=(1, 512))
    print('=== 1st forward ===')
    y = net(x)
    print('=== 2nd forward ===')
    y = net(x)
    print(y)

def model_train():
    # creat data
    features, labels = make_data_label()

    # define loop
    batch_size = 10
    # 将训练数据的特征和标签组合。
    dataset = gdata.ArrayDataset(features, labels)
    # 随机读取小批量。
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    # define net and init it
    net = Net()
    net.collect_params().initialize()
    net.hybridize()
    input_symbol = mx.symbol.Variable('input_data')

    # output computation graph and summary
##    net = net(input_symbol)
##    digraph = mx.viz.plot_network(net,save_format='dot')
##    digraph.view()
##    mx.visualization.print_summary(net)
    
    # define loss
    loss = gloss.L2Loss()  # 平方损失又称 L2 范数损失。

    # define Trainer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    # run loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                predict = net(X)
                print(predict)
                l = loss(predict, y)
##            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))    

if __name__== '__main__':
    model_train()
