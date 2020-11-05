# DeneAtrousConvolutionNet

from mxnet.gluon import Block, nn
from mxnet import nd, init 
import numpy as np
from GetData import create_dataloader
import d2l

# 参数
LAYOUT = 'NCHW'
dropout_rate = 0.1
activation = 'PReLU'
eps = 1.1e-5
concat_axis = 1 # channel 
nb_filter = 3 # 64
compression = 1.0


def conv_block(c,k,s,p,r):
    blk = nn.HybridSequential()
    blk.add(#nn.BatchNorm(),
            nn.PReLU(),
            nn.Conv2D(channels=c, kernel_size=k, strides=s, padding=p, dilation=r, layout = LAYOUT),
            nn.Dropout(rate = dropout_rate))
    return blk

#定义密集带孔卷积块DenseAtrous
class dense_block(nn.HybridBlock):
    def __init__(self, parameters, **kwargs):
        super(dense_block, self).__init__(**kwargs)
        #
        self.net = nn.HybridSequential()
        for i in range(len(parameters)):
            #c,k,s,p,r = self.parameter_ckspr[i]
            c,k,s,p,r = parameters[i]
            print('dense_block output shize ->{}'.format(np.floor((512+2*p-r*(k-1)-1)/s)+1))
            self.net.add(conv_block(c,k,s,p,r))   

    def forward(self, x):
        current = x
        y = None
        for layer in self.net:
            if y is not None:
                current = nd.concat(current, y, dim = 1)
            y = layer(current)
            
        return y



# 连接层
def transition_block(parameters):
    #
    blk = nn.HybridSequential()
    blk.add(nn.BatchNorm(epsilon = eps, axis = concat_axis, scale = True),
            # x = Scale(axis=concat_axis, name=conv_name_base+'_scale'),
            nn.PReLU(),
            nn.Conv2D(int(nb_filter * compression), kernel_size = (1,1), padding = (1,1), use_bias = False, layout = LAYOUT))       
    if dropout_rate:
        blk.add(nn.Dropout(dropout_rate))
    k,s,p = parameters
    print('transition output size = {}'.format(np.floor((512+2*p-k)/s)+1))
    # blk.add(nn.AvgPool2D(pool_size = (2, 2), strides = (2, 2), layout = LAYOUT))
    blk.add(nn.AvgPool2D(pool_size = k, strides = s, padding = p, layout = LAYOUT)) 
    return blk
    

# 定义密集带孔卷积网络
def dense_atrous_conv_net():
    net = nn.HybridSequential()
    parameters = [[8,3,1,3,3],\
                  [8,3,1,6,6],\
                  [8,3,1,12,12],\
                  [16,3,1,18,18]]
    net.add(dense_block(parameters = parameters))
    parameters = [2,2,0]
    net.add(transition_block(parameters = parameters))
    parameters = [[16,3,1,3,3],\
                  [16,3,1,6,6],\
                  [32,3,1,12,12],\
                  [32,3,1,18,18]]
    net.add(dense_block(parameters = parameters))
    parameters = [2,2,0]
    net.add(transition_block(parameters = parameters))
    parameters = [[32,3,1,3,3],\
                  [64,3,1,6,6],\
                  [64,3,1,12,12],\
                  [64,3,1,17,18]]
    net.add(dense_block(parameters = parameters))
    parameters = [2,1,0]
    net.add(transition_block(parameters = parameters))
    return net



# 定义密集全局带孔平均金字塔池化模型
class dense_global_atrous_spatial_pyramid_pooling(nn.HybridBlock):
    def __init__(self, parameters, **kwargs):
        super(dense_global_atrous_spatial_pyramid_pooling, self).__init__(**kwargs)
        # init var
        self.net = nn.HybridSequential()
        layers = 5
        #
        for i in range(layers):
            c,k,s,p,r = parameters[i]
            print('dgaspp output size->{}'.format(np.floor((64+2*p-r*(k-1)-1)/s)+1))
            self.net.add(conv_block(c,k,s,p,r))
            if i == 4:
                self.net.add(nn.Conv2D(channels = c, kernel_size = 3, strides = 1, padding = 6, dilation=6, layout = LAYOUT))
    def forward(self, x):
        current = x
        y = None
        for blk in self.net:
            if y is not None:
                current = nd.concat(current, y, dim=1)
            y = blk(current)
        return y
    
def dgaspp_other(channels):
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(channels = channels[0], kernel_size = 3, strides = 2, padding = 12, dilation = 12 ,layout = LAYOUT))
    blk.add(nn.AvgPool2D(pool_size = 3, strides = 1, padding = 1, layout = LAYOUT))
    blk.add(nn.Conv2D(channels = channels[1], kernel_size = 1, strides = 1, layout = LAYOUT))
    return blk
                                    
def dgaspp_net(channels):
    net = nn.HybridSequential()
    parameters = [[channels,3,1,3,3],\
                  [channels,3,1,6,6],\
                  [channels,3,1,12,12],\
                  [channels,3,1,18,18],\
                  [channels,1,1,0,1]]
    net.add(dense_global_atrous_spatial_pyramid_pooling(parameters = parameters))
    parameters = [channels, channels * 2]
    net.add(dgaspp_other(channels = parameters))
    return net


# 定义有多个中间输出的复合金字塔编码器
class Encoder_Net(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Encoder_Net, self).__init__(**kwargs)
        # setup main net
        self.dense_atrous_conv_net = dense_atrous_conv_net()
        self.dgaspp_net_128 = dgaspp_net(channels = 128)
        self.dgaspp_net_512 = dgaspp_net(channels = 512)
        self.conv2d_1 = nn.Conv2D(channels = 64, kernel_size = 1, strides = 1)
        self.conv2d_2 = nn.Conv2D(channels = 128, kernel_size = 3, strides = 1, padding = 1)
        self.conv2d_3 = nn.Conv2D(channels = 256, kernel_size = 3, strides = 2, padding = 0)
        #self.conv2d_4 = nn.Conv2D(channels = 3, kernel_size = 1, strides = 1) # for show
    def forward(self, x):
        feature_b = self.dense_atrous_conv_net(x)   # 64x128x128
        # load 1
        feature_h = self.conv2d_1(feature_b)        # 64x128x128=image
        # load 2
        feature_e = self.conv2d_2(feature_b)        # 128x128x128
        feature_c = self.conv2d_3(feature_e)        # 256x64x64
        feature_d = self.dgaspp_net_512(feature_c)  # 1024x32x32
        # load 3
        feature_f = self.dgaspp_net_128(feature_e)  # 512x64x64
        feature_f = self.dgaspp_net_512(feature_f)  # 1024x32x32
        # concat 2,3
        feature_g = nd.concat(feature_f, feature_d, dim = 1) # 2048x32x32
        #feature_h = self.conv2d_4(feature_h)        # for show
        return feature_g, feature_h
    
# 初级反卷积网络
def primery_decoder():
    net = nn.HybridSequential()
    #
    net.add(nn.Conv2DTranspose(channels = 2048, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 2048, kernel_size = 3, strides = 2, padding = 1, output_padding = 1))
    net.add(nn.Conv2DTranspose(channels = 1024, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 1024, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 1024, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 512, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 512, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 512, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 512, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 256, kernel_size = 3, strides = 2, padding = 1, output_padding = 1))
    net.add(nn.Conv2DTranspose(channels = 256, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 256, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 256, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 128, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 128, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 128, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 128, kernel_size = 3, strides = 1, padding = 1))
    return net


# 特征还原器
def return_feature():
    net = nn.HybridSequential()
    #
    net.add(nn.Conv2DTranspose(channels = 64, kernel_size = 3, strides = 2, padding = 1, output_padding = 1))
    net.add(nn.Conv2DTranspose(channels = 64, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 64, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 64, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 32, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 32, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 32, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 32, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 16, kernel_size = 3, strides = 2, padding = 1, output_padding = 1))
    net.add(nn.Conv2DTranspose(channels = 16, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 16, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 16, kernel_size = 3, strides = 1, padding = 1))
    #
    net.add(nn.Conv2DTranspose(channels = 8, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 8, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 8, kernel_size = 3, strides = 1, padding = 1))
    net.add(nn.Conv2DTranspose(channels = 8, kernel_size = 3, strides = 1, padding = 1))
    return net



class Decoder_Net(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Decoder_Net, self).__init__(**kwargs)
        # setup main net    
        self.primery_decoder = primery_decoder()
        self.return_feature = return_feature()
    def forward(self, x1, x2):
        x2 = self.primery_decoder(x2)
        middle_feature = nd.concat(x1, x2, dim = 1)
        y_hat = self.return_feature(middle_feature)
        return y_hat

class model(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(model, self).__init__(**kwargs)
        self.Encoder_Net = Encoder_Net()
        self.Decoder_Net = Decoder_Net()
        #self.flatten = nn.Flatten()
        self.conv2d_last = nn.Conv2D(channels = 20, kernel_size = 1, strides =1, padding = 0, layout = LAYOUT)
    def forward(self, x):
        x_y_hat, x_image = self.Encoder_Net(x)
        image = self.Decoder_Net(x_image, x_y_hat)
        y_hat = self.conv2d_last(image)
        y_hat = nd.SoftmaxActivation(y_hat, mode = 'channel')
        return y_hat, image

if __name__== '__main__':
    ## 生成输入数据
##    x = nd.random.uniform(shape=(8,3,512,512))
##    print(x.shape)
##    x1 = nd.random.uniform(shape=(8,3,128,128))
##    x2 = nd.random.uniform(shape=(8,3,32,32))
##    print(x1.shape)
##    print(x2.shape)
    
    t,v = create_dataloader()
    
    ## 代入神经网络生成y_hat

    #net = dense_atrous_conv_net()
    #net = dgaspp_net(512)
    #net = Encoder_Net()
    #net = Decoder_Net()
    net = model()
    net.initialize(init=init.Xavier())
    

    # 看网络结构
    print(net)
##    image, y_hat = net(x)
##    print(image.shape, y_hat.shape)
##    net.hybridize()

    # 显示中间结果
    for x,y in t:
        # 输入并转换通道
        print('input x shape = {}'.format(x.shape))
        y_hat, image = net(nd.transpose(x,axes = (0,3,1,2)))
        print('output y_hat shape = {}'.format(y_hat.shape))
        print('output image shape = {}'.format(image.shape))
        # 通道转换
        image = nd.transpose(image, axes = (0,2,3,1))
        #y_hat = nd.transpose(y_hat, axes = (0,2,3,1))
        # 显示热量图
        d2l.show_images_ndarray([x,y,image],3,8,2)
        break
    net.hybridize()


