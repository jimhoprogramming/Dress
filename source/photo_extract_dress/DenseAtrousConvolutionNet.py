# DeneAtrousConvolutionNet

from mxnet.gluon import Block, nn
from mxnet import nd, init 
import numpy as np
from GetData import create_dataloader
import d2l

LAYOUT = 'NCHW'

def conv_block(c,k,s,p,r):
    blk = nn.Sequential()
    blk.add(#nn.BatchNorm(),
            #nn.Activation('relu'),
            nn.Conv2D(channels=c, kernel_size=k, strides=s, padding=p, dilation=r, layout = LAYOUT))
    return blk

#定义密集带孔卷积块DenseAtrous
class dense_block(nn.Block):
    def __init__(self, **kwargs):
        super(dense_block, self).__init__(**kwargs)
        self.parameter_ckspr = []
        self.parameter_ckspr.append([8,3,1,3,3])
        self.parameter_ckspr.append([8,3,1,6,6])
        self.parameter_ckspr.append([8,3,1,12,12])
        self.parameter_ckspr.append([16,3,2,0,1])
        #
        self.net = nn.Sequential()
        for i in range(4):
            c,k,s,p,r = self.parameter_ckspr[i]
            #print('out->{}'.format(np.floor((512+2*p-r*(k-1)-1)/s)+1))
            self.net.add(conv_block(c,k,s,p,r))   

    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            if y.shape[2] == x.shape[2]:
                x = nd.concat(x, y, dim=1)
            else:
                x = y
        return x



# 连接层
def transition_block():
    # 参数
    eps = 1.1e-5
    concat_axis = 1 # channel 
    nb_filter = 3 # 64
    compression = 1.0
    dropout_rate = 0.0
    #
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(epsilon = eps, axis = concat_axis, scale = True),
            # x = Scale(axis=concat_axis, name=conv_name_base+'_scale'),
            nn.Activation('relu'),
            nn.Conv2D(int(nb_filter * compression), kernel_size = (1,1), padding = (1,1), use_bias = False, layout = LAYOUT))       
    if dropout_rate:
        blk.add(nn.Dropout(dropout_rate))
    blk.add(nn.AvgPool2D(pool_size = (2, 2), strides = (2, 2), layout = LAYOUT))      
    return blk
    

# 定义密集带孔卷积网络
def dense_atrous_conv_net():
    net = nn.Sequential()
    net.add(dense_block())
    net.add(transition_block())
    return net



# 定义密集全局带孔平均金字塔池化模型
class dense_global_atrous_spatial_pyramid_pooling(nn.Block):
    def __init__(self, **kwargs):
        super(dense_global_atrous_spatial_pyramid_pooling, self).__init__(**kwargs)
        # init var
        self.parameter_ckspr = []
        self.parameter_ckspr.append([3,3,1,3,3])
        self.parameter_ckspr.append([3,3,1,6,6])
        self.parameter_ckspr.append([3,3,1,12,12])
        self.parameter_ckspr.append([3,3,1,18,18])
        self.parameter_ckspr.append([3,1,1,0,1])
        self.net = nn.Sequential()
        layers = 5
        #
        for i in range(layers):
            c,k,s,p,r = self.parameter_ckspr[i]
            print('out->{}'.format(np.floor((64+2*p-r*(k-1)-1)/s)+1))
            self.net.add(conv_block(c,k,s,p,r))
            if i == 4:
                self.net.add(nn.Conv2D(channels = 3, kernel_size = 3, strides = 1, padding = 6, dilation=6, layout = LAYOUT))
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            if y.shape[2] == x.shape[2]:
                x = nd.concat(x, y, dim=1)
            else:
                x = y
        return x
    
def dgaspp_other():
    blk = nn.Sequential()
    blk.add(nn.Conv2D(channels = 3, kernel_size = 3, strides = 2, padding = 12, dilation = 12 ,layout = LAYOUT))
    blk.add(nn.AvgPool2D(pool_size = 3, strides = 1, padding = 1, layout = LAYOUT))
    blk.add(nn.Conv2D(channels = 3, kernel_size = 1, strides = 1, layout = LAYOUT))
    return blk
                                    
def dgaspp_net():
    net = nn.Sequential()
    net.add(dense_global_atrous_spatial_pyramid_pooling())
    net.add(dgaspp_other())
    return net

def Encoder_Net():
    net = nn.Sequential()
    net.add(dense_atrous_conv_net())
    

    
if __name__== '__main__':
    ## 生成输入数据
##    x = nd.random.uniform(shape=(8,3,64,64))
##    print(x.shape)
    t,v = create_dataloader()
    
    ## 代入神经网络生成y_hat

    net = dense_atrous_conv_net()
    #net = dgaspp_net()
    net.initialize(init=init.Xavier())

    # 看网络结构
    print(net)

    # 显示中间结果
    for x,y in t:
        # 输入并转换通道
        print('input x shape = {}'.format(x.shape))
        y_hat = net(nd.transpose(x,axes = (0,3,1,2)))
        print('output y_hat shape = {}'.format(y_hat.shape))
        # 通道转换
        y_hat = nd.transpose(y_hat, axes = (0,2,3,1))
        # 显示热量图
        d2l.show_images_ndarray([x,y,y_hat],3,8,2)
        break


