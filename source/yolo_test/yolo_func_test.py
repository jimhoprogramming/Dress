# -*- coding: cp936 -*-
import sys
import numpy as np
import cv2
##import KmeanToIm
##
###Im=cv2.imread('d:\\robots\\robotscripts\\data\\road1.jpg',0)
##Im=cv2.imread('d:\\dress\\768x847.jpg',0)
####print Im.dtype
##print str(type(Im))
###改变输出图象的确大小400x300dpi 4:3
##resizeIm=cv2.resize(Im,(400,300),interpolation=cv2.INTER_LINEAR) 
##
##
####KmeanIm=KmeanToIm.doKmean('d:\\robots\\robotscripts\\data\\road.jpg')
##GaussianIm=cv2.GaussianBlur(resizeIm,(3,3),4)
##
##
### 阀值化处理得到二值化图
##
##
###CANNY边缘提取
##edges=cv2.Canny(GaussianIm,5,50, apertureSize = 3)
##
###HOUGH变换得到直线
##minLineLength = 1
##maxLineGap = 1
##lines = cv2.HoughLines(edges,1,np.pi/180,10)
####result = KmeanIm.copy()
##result = resizeIm.copy()
##lineps = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
##for x1,y1,x2,y2 in lineps[0]:
##	cv2.line(Im,(x1,y1),(x2,y2),(0,255,0),2)
##
###显示结果
##
###霍夫变换的输出
##result = resizeIm.copy()
##for line in lines[0]:
##	rho = line[0] #第一个元素是距离rho
##	theta= line[1] #第二个元素是角度theta
##	print rho
##	print theta
##	if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
##                #该直线与第一行的交点
##		pt1 = (int(rho/np.cos(theta)),0)
##		#该直线与最后一行的焦点
##		pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
##		#绘制一条白线
##		cv2.line( result, pt1, pt2, (255))
##	else: #水平直线
##		# 该直线与第一列的交点
##		pt1 = (0,int(rho/np.sin(theta)))
##		#该直线与最后一列的交点
##		pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
##		#绘制一条直线
##		cv2.line(result, pt1, pt2, (255), 1)
cv2.namedWindow('Canny',flags=cv2.WINDOW_NORMAL)
##cv2.imshow('orginIm',resizeIm)
##cv2.imshow('Gaussian',GaussianIm)




edges = np.full((50,50,3),128,dtype = np.uint8)
cv2.imshow('Canny', edges )
##edges2=edges.copy()
##edges2[200:400,:]=0
##cv2.imshow('canny2',edges2)
##cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
###保存结果
##
##cv2.imwrite('d:\\robots\\robotscripts\\data\\cannySave.jpg',edges)
##
##
##
##
