# coding=utf-8
# 读取图片 返回图片某像素点的b，g，r值
import cv2
import numpy as np
import numpy

img = cv2.imread('./images/4.png')
px = img[1024, 561]
print (px)#输出
blue = img[10, 10, 0]
print (blue)
green = img[10, 10, 1]
print (green)
red = img[10, 10, 2]
print (red)

#1.获取像素点并输出
BGR =numpy.array(px)
upper =BGR+5
lower =BGR-5
mask =cv2.inRange(img,lower,upper)
cv2.imshow("NewWindow",mask)
cv2.waitKey(0)