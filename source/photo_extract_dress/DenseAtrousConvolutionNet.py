# DeneAtrousConvolutionNet

from mxnet.gluon import Block, nn
from mxnet import nd, init 
import numpy as np
from GetData import create_dataloader
import d2l

LAYOUT = 'NCHW'

def conv_block(c,k,s,p,r):
    blk = nn.HybridSequential()
    blk.add(#nn.BatchNorm(),
            #nn.Activation('relu'),
            nn.Conv2D(channels=c, kernel_size=k, strides=s, padding=p, dilation=r, layout = LAYOUT))
    return blk

#定义密集带孔卷积块DenseAtrous
class dense_block(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(dense_block, self).__init__(**kwargs)
        self.parameter_ckspr = []
        self.parameter_ckspr.append([8,3,1,3,3])
        self.parameter_ckspr.append([8,3,1,6,6])
        self.parameter_ckspr.append([8,3,1,12,12])
        self.parameter_ckspr.append([16,3,2,0,1])
        #
        self.net = nn.HybridSequential()
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
    blk = nn.HybridSequential()
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
    net = nn.HybridSequential()
    net.add(dense_block())
    net.add(transition_block())
    return net



# 定义密集全局带孔平均金字塔池化模型
class dense_global_atrous_spatial_pyramid_pooling(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(dense_global_atrous_spatial_pyramid_pooling, self).__init__(**kwargs)
        # init var
        self.parameter_ckspr = []
        self.parameter_ckspr.append([3,3,1,3,3])
        self.parameter_ckspr.append([3,3,1,6,6])
        self.parameter_ckspr.append([3,3,1,12,12])
        self.parameter_ckspr.append([3,3,1,18,18])
        self.parameter_ckspr.append([3,1,1,0,1])
        self.net = nn.HybridSequential()
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
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(channels = 3, kernel_size = 3, strides = 2, padding = 12, dilation = 12 ,layout = LAYOUT))
    blk.add(nn.AvgPool2D(pool_size = 3, strides = 1, padding = 1, layout = LAYOUT))
    blk.add(nn.Conv2D(channels = 3, kernel_size = 1, strides = 1, layout = LAYOUT))
    return blk
                                    
def dgaspp_net():
    net = nn.HybridSequential()
    net.add(dense_global_atrous_spatial_pyramid_pooling())
    net.add(dgaspp_other())
    return net

# 定义有多个中间输出的复合金字塔编码器
class Encoder_Net(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Encoder_Net, self).__init__(**kwargs)
        # setup main net
        self.dense_atrous_conv_net = dense_atrous_conv_net()
        self.dgaspp_net = dgaspp_net()
        self.conv2d_1 = nn.Conv2D(channels = 3, kernel_size = 1, strides = 1)
        self.conv2d_2 = nn.Conv2D(channels = 3, kernel_size = 3, strides = 1, padding = 1)
        self.conv2d_3 = nn.Conv2D(channels = 3, kernel_size = 3, strides = 2, padding = 0)
        self.conv2d_4 = nn.Conv2D(channels = 3, kernel_size = 1, strides = 1)
    def forward(self, x):
        feature_b = self.dense_atrous_conv_net(x)
        # load 1
        feature_h = self.conv2d_1(feature_b)
        # load 2
        feature_e = self.conv2d_2(feature_b)
        feature_c = self.conv2d_3(feature_e)
        feature_d = self.dgaspp_net(feature_c)
        # load 3
        feature_f = self.dgaspp_net(feature_e)
        feature_f = self.dgaspp_net(feature_f)
        # concat 2,3
        feature_g = nd.concat(feature_f, feature_d, dim = 1)
        feature_g = self.conv2d_4(feature_g)
        return feature_h, feature_g
    

    
if __name__== '__main__':
    ## 生成输入数据
##    x = nd.random.uniform(shape=(8,3,64,64))
##    print(x.shape)
    t,v = create_dataloader()
    
    ## 代入神经网络生成y_hat

    #net = dense_atrous_conv_net()
    #net = dgaspp_net()
    net = Encoder_Net()
    net.initialize(init=init.Xavier())

    # 看网络结构
    print(net)

    # 显示中间结果
    for x,y in t:
        # 输入并转换通道
        print('input x shape = {}'.format(x.shape))
        y_hat_1, y_hat_2 = net(nd.transpose(x,axes = (0,3,1,2)))
        print('output y_hat_1 shape = {}'.format(y_hat_1.shape))
        print('output y_hat_2 shape = {}'.format(y_hat_2.shape))
        # 通道转换
        y_hat_1 = nd.transpose(y_hat_1, axes = (0,2,3,1))
        y_hat_2 = nd.transpose(y_hat_2, axes = (0,2,3,1))
        # 显示热量图
        d2l.show_images_ndarray([x,y,y_hat_2],3,8,2)
        break


