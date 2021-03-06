# 提取pascal 2012voc 语义分类图片数据
# 平台 mxnet
from mxnet import image, gluon, nd
#npx.set_np()
import d2l
import random

class P2012Voc_DataSet(gluon.data.Dataset):
    def __init__(self, is_train = True):
        super(P2012Voc_DataSet, self).__init__()
        url_file_train = '/home//jim//Data//VOCdevkit//VOC2012//ImageSets//Segmentation//train.txt'
        url_file_val = '/home//jim//Data//VOCdevkit//VOC2012//ImageSets//Segmentation//trainval.txt'
        self.pre_x_path = '/home//jim//Data//VOCdevkit//VOC2012//JPEGImages'
        self.pre_y_path = '/home//jim//Data//VOCdevkit//VOC2012//SegmentationClass'
        self.crop_size = [512,512]
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        if is_train:
            self.url = url_file_train
        else:
            self.url = url_file_val
        with open(self.url, "r") as file:
            self.image_list = file.readlines()
        self.image_list = [im.strip() for im in self.image_list]
        self.colormap2label = d2l.create_colormap2label()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.crop_size == [512,512]:
            x, y = self.__filter(idx,pass_filter = True)
            x, y = self.__resize(feature = x, label = y, height = self.crop_size[0], width = self.crop_size[1])
        else:
            x, y = self.__filter(idx,pass_filter = False)
            x, y = self.__voc_rand_crop(feature = x, label = y, height = self.crop_size[0], width = self.crop_size[1])
        # to class id
        label_ind = nd.array(d2l.voc_label_indices(y.astype('float32'), self.colormap2label))
        print('out x = {}'.format(x.shape))
        print('out y = {}'.format(y.shape))
        print('out label_ind = {}'.format(label_ind.shape))
        return x, y, label_ind

    def __voc_rand_crop(self, feature, label, height, width):
        feature, rect = image.random_crop(feature, (width, height))
        label = image.fixed_crop(label, *rect)
        return feature, label
    
    def __resize(self, feature, label, height, width):
        img_h, img_w = feature.shape[0], feature.shape[1]     # 行代表高 ， 列代表宽
        w, h = width, height
        scale = max(w * 1.0/img_w, h * 1.0/img_h)
        print('x_h = {},x_w = {},to_w = {},to_h = {}, scale = {}'.format(img_h,img_w,w,h,scale))
        new_w = int(img_w * scale)+1
        new_h = int(img_h * scale)+1
        resized_image = image.imresize(src = feature, w = new_w,  h = new_h, interp=1)   # 改变图像尺寸[fx，fy]
        resized_label = image.imresize(src = label, w = new_w,  h = new_h, interp=1)   # 改变图像尺寸[fx，fy]
        print('resized_image = {}'.format(resized_image.shape))
        print('resized_label = {}'.format(resized_label.shape))
        return self.__voc_rand_crop(resized_image, resized_label, height, width)
              
    def __filter(self, idx, pass_filter = True):
        while True:
            x = image.imread(self.pre_x_path + '//' + self.image_list[idx] + '.jpg')
            y = image.imread(self.pre_y_path + '//' + self.image_list[idx] + '.png')
            print(x.shape)
            #x = nd.array(x)
            #y = nd.array(y)
            if (x.shape[0] >= self.crop_size[0] and x.shape[1] >= self.crop_size[1]) or pass_filter:
                if x.shape == y.shape:
                    break
            else:
                idx = random.randint(0,len(self.image_list))
        
        return self.__normalize_image(x), y.astype('float32')
    
    def __normalize_image(self, img):
        img = (img.astype('float32') / 255.0 - self.rgb_mean) / self.rgb_std
        img = img.clip(0.0,1.0)
        return img

def create_dataloader(batch_size = 8):
    train_dataset = P2012Voc_DataSet(is_train = True)
    train_dataloader = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = P2012Voc_DataSet(is_train = False)
    val_dataloader = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


    
if __name__ == '__main__':
    t,v = create_dataloader()
    for x,y in t:
        print('x input shape = {}'.format(x.shape))
        print('x intput type = {}'.format(x.dtype))
        print('input min:{},max:{}'.format(x.min().asscalar(), x.max().asscalar()))
        #
        print('label_image shape = {}'.format(y.shape))
        print('label_image type = {}'.format(y.dtype))
        print('label_image min:{},max:{}'.format(y.min().asscalar(), y.max().asscalar()))
        #
        d2l.show_images_ndarray([x,y],2,8,2)
        
        
