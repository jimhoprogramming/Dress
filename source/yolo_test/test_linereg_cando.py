# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon
from mxnet import nd
from matplotlib import pyplot as plt
from mxnet import autograd
import random


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
            self.fc3 = gluon.nn.Dense(2)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        print('type(x): {}, F: {}'.format(type(x).__name__, F.__name__))
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

    # init w b
    num_inputs = 2
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
        
    # loop batch
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):  # 训练模型一共需要 num_epochs 个迭代周期。
        # 在一个迭代周期中，使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
        # X 和 y 分别是小批量样本的特征和标签。
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)  # l 是有关小批量 X 和 y 的损失。
            l.backward()  # 小批量的损失对模型参数求梯度。
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数。
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))


if __name__== '__main__':
    model_train()
