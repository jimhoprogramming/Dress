import os, numpy as np, time
from mxnet.gluon import nn, loss as gloss, Trainer
from mxnet import autograd
from mxnet import initializer as init
from mxnet import nd
from mxboard import *
#
from DenseAtrousConvolutionNet import model 
from GetData import create_dataloader

class Train(object):
    def __init__(self):
        object.__init__(self)
        self.model = model()
        self.loss = None
        self.trainer = None
        self.learning_rate = 0.01
        self.weight_url = '/home//jim//Dress//Data//weights.params'
        self.logdir_url = '/home//jim//Dress//log'
        self.sw = None

    def set_weight_name(self, name):
        self.weight_url = self.weight_url.replace('fill_it', name)
        
    def load_weights(self, weight_url):
        if os.path.exists(self.weight_url):
            self.model.load_parameters(self.weight_url)        

    def init_model(self):
        # 新模型初始化
        if os.path.exists(self.weight_url):
            print(u'已含有旧权重文件，正在载入继续训练并更新')
            self.model.load_parameters(self.weight_url, allow_missing = True, ignore_extra = True)
        else:
            #self.model.collect_params("conv*|dense*").initialize(init.Xavier())
            self.model.initialize(init=init.Xavier())
        #self.model.hybridize()

        # 显示模型
        print(u'model construct display:')
        print(self.model)

    def model_compile(self):
        # 编译模型
        self.loss = gloss.L2Loss()
        self.trainer = Trainer(self.model.collect_params(), optimizer = 'adam')
        self.trainer.set_learning_rate = self.learning_rate

    def predict(self, x):
        x = nd.expand_dims(x, axis = 0)
        x = nd.transpose(x, axes = (0,3,1,2))
        #print(u'单个样本预测输入的形状：{}'.format(x.shape))
        # 初始化及编译模型准备训练
        self.init_model()
        self.model_compile()
        #self.setup_debug()
        # 进入预测
        with autograd.predict_mode():
            #print(u'训练模式：{}'.format(autograd.is_training()))
            output = self.model(x)
        return output

    def acc(self, output, label):
        # output: (batch, num_output) float32 ndarray
        # label: (batch, label) float32 ndarray
        return (output == label.flatten()).mean().asscalar()    

    def fit(self, dataset_train, dataset_val, epochs = 100, verbose = 1, validation_split = 0.3, batch_size = 64):
        # 初始化及编译模型准备训练
        self.init_model()
        self.model_compile()
        self.setup_debug()
        # 进入训练
        train_data, valid_data = dataset_train, dataset_val
        global_step = 0
        for epoch in nd.arange(epochs):
            train_loss, train_acc, val_acc = 0., 0., 0.
            tic = time.time()
            train_step = 0
            for data, label in train_data:
                data = nd.transpose(data,axes = (0,3,1,2))
                label = nd.transpose(label,axes = (0,3,1,2))
                # show img as y
                #show_data(data, label)
                # forward + backward
                with autograd.record():
                    #
                    print(u'训练模式：{}'.format(autograd.is_training()))
                    '''
                    print(u'输入数据形态{}, type = {}'.format(data.shape, data.dtype))
                    print('input max:{},min:{}'.format(data.max(),data.min()))
                    '''
                    #
                    output,train_image = self.model(data)
                    '''
                    print(u'输出y_hat数据形态{}, type = {}'.format(output.shape, output.dtype))
                    print(u'输出label数据形态{}, type = {}'.format(label.shape, label.dtype))
                    print('output max:{},min:{}'.format(output.max().asscalar(), output.min().asscalar()))
                    print('label max:{},min:{}'.format(label.max().asscalar(), label.min().asscalar()))
                    '''    
                    #
                    loss = self.loss(output, label.flatten())
                    #print('loss shape : {}'.format(loss.shape))
                    print('loss value: {}'.format(loss))
                    #
                    loss.backward()
                # update parameters
                self.trainer.step(batch_size)
                # calculate training metrics
                train_loss += np.mean(loss.asnumpy())
                #print('train_loss shape:{}'.format(train_loss.shape))
                #print('train_loss:{}'.format(train_loss))
                train_acc += self.acc(output, label)
                #print('train_acc:{}'.format(train_acc))
                # caculate countor
                train_step += 1
                
                # check_bug
                #print(u'train set check:')
                #self.check_bug(label, output)
                # callback
                train_loss_mean = train_loss/train_step
                train_acc_mean = train_acc/train_step
                #
                self.call_back(global_step = global_step, \
                               train_step = train_step, \
                               train_loss_mean = train_loss_mean, \
                               train_acc_mean = train_acc_mean, \
                               train_data = data, \
                               train_image = train_image, \
                               train_label = label, \
                               val_step = 0, \
                               val_acc_mean = None, \
                               val_data = None, \
                               val_image = None, \
                               val_label = None \
                               )
                


            # calculate validation accuracy
            val_step = 0
            for val_data, val_label in valid_data:
                val_data = nd.transpose(val_data,axes = (0,3,1,2))
                val_label = nd.transpose(val_label,axes = (0,3,1,2))
                with autograd.predict_mode():
                    print(u'训练模式：{}'.format(autograd.is_training()))
                    val_output,val_image = self.model(val_data)   
                #
                val_acc += self.acc(val_output, val_label)
                #print(u'val set check:')
                #self.check_bug(label, output)
                val_step += 1
                val_acc_mean = val_acc/val_step
                #
                self.call_back(global_step = global_step, \
                               train_step = train_step, \
                               train_loss_mean = train_loss_mean, \
                               train_acc_mean = train_acc_mean, \
                               train_data = data, \
                               train_image = train_image, \
                               train_label = label, \
                               val_step = val_step, \
                               val_acc_mean = val_acc_mean, \
                               val_data = val_data, \
                               val_image = val_image, \
                               val_label = val_label \
                               )
                
            # display rel
            print("Epoch {}: loss {}, train acc {}, test acc {}, using {} sec" .format(
                    epoch, train_loss/train_step, train_acc/train_step,
                    val_acc/val_step, time.time()-tic))

            # save weights
            self.model.save_parameters(self.weight_url)
            #
            global_step += 1
            
    def check_bug(self, label, output):
        # 真阴、真阳、假阴、假阳统计
        np_label = label.asnumpy()
        output = output.argmax(axis=1)
        np_output = output.asnumpy()
        for i in [0,1,2]:
            index = np.ix_(np_label == i)[0]
            np_label_class_number = index.shape[0]
            post_in_output = np_output[index]
            #print(post_in_output)
            for p in [0,1,2]:
                np_output_class_number = post_in_output[post_in_output == p].shape[0]
                if i == p:
                    print(u'action = {}, real have {} predict have {}'.format(i, np_label_class_number, np_output_class_number))
        print('---onetime check over---')
        
        
    def setup_debug(self):
        self.sw = SummaryWriter(logdir = self.logdir_url, flush_secs=10)

    def image_255_to_01(self, img):
        img = nd.transpose(img, axes = (0,2,3,1))
        for i in nd.arange(img.shape[0]):
            image = img[i,:,:,:]            
            rgb_mean = nd.array([0.485, 0.456, 0.406])
            rgb_std = nd.array([0.229, 0.224, 0.225])
            image = (image.astype('float32') / 255.0 - rgb_mean) / rgb_std
        img = img.clip(0.0,1.0)
        img = nd.transpose(img, axes = (0,3,1,2))
        return img

        
    def call_back(self,
                  global_step = 0,
                  train_step = 0,
                  train_loss_mean = None,
                  train_acc_mean = None,
                  train_data = None,
                  train_image = None,
                  train_label = None,
                  val_step = 0,
                  val_acc_mean = None,
                  val_data = None,
                  val_image = None,
                  val_label = None,
                  ):
        # 加入模型框架图
        #if global_step == 0:
            #self.sw.add_graph(self.model)
        
        # 加入某层输出图像变化
        if train_data is not None:
            train_image = self.image_255_to_01(train_image)
            train_label = self.image_255_to_01(train_label)
            #
            self.sw.add_image('train_input_image', train_data, train_step)
            self.sw.add_image('train_output_image', train_image, train_step)
            self.sw.add_image('train_output_label', train_label, train_step)
        if val_data is not None:
            val_data = self.image_255_to_01(val_data)
            val_image = self.image_255_to_01(val_image)
            val_label = self.image_255_to_01(val_label)
            #
            self.sw.add_image('val_input_image', val_data, val_step)
            self.sw.add_image('val_output_image', val_image, val_step)
            self.sw.add_image('val_output_label', val_label, val_step)
            

        # 加入学习次数-loss、acc变化曲线图
        if train_loss_mean is not None:
            self.sw.add_scalar(tag = 'L2Loss_and_acc', \
                               value = {'train_loss': train_loss_mean, 'train_acc': train_acc_mean}, \
                               global_step = train_step)
        if val_acc_mean is not None:
            self.sw.add_scalar(tag = 'val_acc', \
                               value = {'val_acc' : val_acc_mean}, \
                               global_step = val_step)            
            

        # 加入某个层权重分布变化等高图
        grads = [i.grad() for i in self.model.collect_params('.*weight|.*bias').values()]
        param_names = [name for name in self.model.collect_params('.*weight|.*bias').keys()]
        assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        for i, name in enumerate(param_names):
            self.sw.add_histogram(tag = name, values = grads[i], global_step = train_step, bins = 20)

if __name__=='__main__':
    t,v = create_dataloader(64)
    Trainer_Obj = Train()
    Trainer_Obj.fit(t,v)
