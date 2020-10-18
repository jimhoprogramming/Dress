import numpy as np
import cv2
from matplotlib import pyplot as plt

def getRect():
    x0,y0 = (67,5 )
    xm,ym = (410,511 )
    return (x0,y0,xm,ym)
img = cv2.imread('E:\\warm_up_train_20180201\\web\\Images\\skirt_length_labels\\0a0c5475316629d56813419f14798c8e.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#rect = (50,50,450,290)
rect=getRect()
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()
