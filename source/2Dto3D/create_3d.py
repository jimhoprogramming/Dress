# -*- coding:utf-8 -*-
# 由已经标识好区域的的图片文件生成空间形状物体
#

from mxnet import image, nd
from matplotlib import pyplot as plt
import numpy as np

# 导入图得有效矩阵
def open_img():
    url = 'd://VOCdevkit//VOC2012//SegmentationClass//2007_006076.png'
    img = image.imread(url)
    return img

# 过滤有效区域
def filter_usage(img):
    data_ndarray = voc_label_indices(img)  
    return data_ndarray


# 本函数已保存在d2lzh包中方便以后使用
def voc_label_indices(img):
    # 该常量已保存在d2lzh包中方便以后使用
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]
    # 该常量已保存在d2lzh包中方便以后使用
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    #
    colormap2label = nd.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    #
    img = img.astype('int32')
    idx = ((img[:, :, 0] * 256 + img[:, :, 1]) * 256
           + img[:, :, 2])
    return colormap2label[idx]

# 生成空间点坐标
def create_3d_vertex(nd_array):
    rel = []
    print(nd_array)
    print('nd_array min = {} ,max = {}'.format(nd.min(nd_array),nd.max(nd_array)))
    # 可视化
    plt.imshow(nd_array.asnumpy())
    plt.colorbar()
    plt.show()
    # create 3d point
    vertex = np.where(nd_array.asnumpy()>0)
    #print(vertex)
    for i in np.arange(vertex[0].shape[0]):
        rel.append([vertex[1][i],vertex[0][i],0])
        #print('point {} position is {}'.format(i,rel[i]))
    return rel
                
##
if __name__ == '__main__':
    img = open_img()
    nd_array = filter_usage(img)
    vertex = create_3d_vertex(nd_array)
    print('vertex_list len = {}'.format(len(vertex)))
    
