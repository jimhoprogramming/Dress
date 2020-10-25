from mxnet.gluon import nn, loss as gloss, Trainer
from mxnet import autograd
from mxnet import initializer as init
from mxnet import nd
#



class Train(object):
    def __init__(self):
        object.__init__(self)
        self.pretrain_model = None
        self.model = None
        self.loss = None
        self.trainer = None
        self.full_model()
        self.fine_tune()
        

        
    def set_weight_name(self, name):
        self.weight_url = self.weight_url.replace('fill_it', name)
        
    def load_weights(self, weight_url):
        if os.path.exists(self.weight_url):
            self.model.load_parameters(self.weight_url)        
        
    
    def full_model(self):

        # 只下权重。没有的话会自动下载zip文件到%mxnethome%/model/目录里，而模型在本地用代码构建
        self.pretrain_model = vision.mobilenet_v2_1_0(pretrained = False)
        # 显示模型全部形态
        #print(self.pretrain_model)
        
    def fine_tune(self):
        '''
           参考：https://zhuanlan.zhihu.com/p/42441251
        '''
        # 截取模型部分层训练
        pre_layers = self.pretrain_model.features[:24]

        # 自补充完最后收尾层
        append_layers = [
                         #nn.BatchNorm(), \
                         #nn.Dropout(rate = 0.2), \
                         #nn.Activation('relu'), \
                         #nn.GlobalAvgPool2D(), \
                         #nn.Flatten(), \
                         nn.Dense(units = self.output_dim), \
                         nn.Flatten() \
                         ]
        # 建一个新的net
        self.model = nn.HybridSequential()

        # 新旧层建构
        for layer in pre_layers:
            '''
            # 加入dropout层
            #print(type(layer))
            if isinstance(layer, nn.basic_layers.HybridSequential):
                for child_layer in layer:
                    if isinstance(child_layer, vision.resnet.BottleneckV2):
                        print(child_layer)
            '''
            self.model.add(layer)
        #    
        for layer in append_layers:
            self.model.add(layer)

    def init_model(self):
        # 新模型初始化
        if os.path.exists(self.weight_url):
            self.model.load_parameters(self.weight_url, allow_missing = True, ignore_extra = True)
        else:
            #self.model.collect_params("conv*|dense*").initialize(init.Xavier())
            self.model.initialize(init=init.Xavier())
        self.model.hybridize()

        # 显示模型
        #print(u'after fine tune model:')
        #print(self.model)

    def model_compile(self):
        # 编译模型
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.trainer = Trainer(self.model.collect_params(), optimizer = 'RMSProp')
        self.trainer.set_learning_rate = self.learning_rate

    def predict(self, x):
        x = x[:,:,:3]
        x = nd.array(x)
        x = nd.expand_dims(x, axis = 0)
        data_transforms = create_transformer()
        x = data_transforms(x)
        #print(u'单个样本预测输入的形状：{}'.format(x.shape))
        # 初始化及编译模型准备训练
        self.init_model()
        self.model_compile()
        self.setup_debug()
        # 进入预测
        with autograd.predict_mode():
            #print(u'训练模式：{}'.format(autograd.is_training()))
            output = self.model(x)
        return output.argmax(axis=1)
        

    def fit(self, dataset, epochs = config.EPOCHS, verbose = 1, validation_split = config.VAL_PERCEND, batch_size = config.BATCH_SIZE):
        # 初始化及编译模型准备训练
        self.init_model()
        self.model_compile()
        #self.setup_debug()
        # 进入训练
        train_data, valid_data = dataset
        global_step = 0
        for epoch in np.arange(epochs):
            train_loss, train_acc, valid_acc = 0., 0., 0.
            tic = time.time()
            for data, label in train_data:
                # show img as y
                #show_data(data, label)
                # forward + backward
                with autograd.record():
                    print(u'训练模式：{}'.format(autograd.is_training()))
                    print(u'数据形态{}'.format(data.shape))
                    output = self.model(data)
                    loss = self.loss(output, label)
                    loss.backward()
                # update parameters
                self.trainer.step(batch_size)
                # calculate training metrics
                train_loss += loss.mean().asscalar()
                train_acc += acc(output, label)
                '''
                # check_bug
                print(u'train set check:')
                self.check_bug(label, output)
                # callback
                train_loss_mean = train_loss/len(train_data)
                train_acc_mean = train_acc/len(train_data)
                
                self.call_back(global_step = global_step, \
                              train_loss_mean = train_loss_mean, \
                              train_acc_mean = train_acc_mean, \
                              train_data = data)
                '''
                global_step += 1

            # calculate validation accuracy
            for data, label in valid_data:
                with autograd.predict_mode():
                    print(u'训练模式：{}'.format(autograd.is_training()))
                    output = self.model(data)   
                #
                valid_acc += acc(output, label)
                #print(u'val set check:')
                #self.check_bug(label, output)
                
            # display rel
            print("Epoch {}: loss {}, train acc {}, test acc {}, using {} sec" .format(
                    epoch, train_loss/len(train_data), train_acc/len(train_data),
                    valid_acc/len(valid_data), time.time()-tic))

            # save weights
            self.model.save_parameters(self.weight_url)
            
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
        self.sw = SummaryWriter(logdir = self.logdir_url, flush_secs=5)

    
    def call_back(self, global_step = 0 , train_loss_mean = None, train_acc_mean = None, train_data = None,  val_acc_mean = None, val_data = None):
        # 加入模型框架图
        if global_step == 0:
            self.sw.add_graph(self.model)
        
        # 加入某层输出图像变化
        if train_data is not None:
            self.sw.add_image('train_minist_first_minibatch', train_data, global_step)
        if val_data is not None:
            self.sw.add_image('val_minist_first_minibatch', val_data, global_step)

        # 加入学习次数-loss、acc变化曲线图
        if train_loss_mean is not None:
            self.sw.add_scalar(tag = 'cross_entropy_and_acc', \
                               value = {'train_loss': train_loss_mean, 'train_acc': train_acc_mean}, \
                               global_step = global_step)
        if val_acc_mean is not None:
            self.sw.add_scalar(tag = 'val_acc', \
                               value = {'val_acc' : val_acc_mean}, \
                               global_step = global_step)            
            

        # 加入某个层权重分布变化等高图
##        grads = [i.grad() for i in self.model.collect_params().values()]
##        assert len(grads) == len(param_names)
##        # logging the gradients of parameters for checking convergence
##        for i, name in enumerate(param_names):
##            self.sw.add_histogram(tag = name, values = grads[i], global_step = global_step, bins = 1000)
