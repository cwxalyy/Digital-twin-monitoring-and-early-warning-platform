import numpy as np
import cv2
#导入numpy及cv2文件包，导入方法
# 读取图片
# 二、修复算法介绍

#1、算法INPAINT_TELEA介绍：
        #基于快速行进算法（FMM）,从待修补区域的边界向区域内部前进，先填充区域边界像素。
        #选待修补区域小的领域，使用领域归一化加权和更新修复像素。（先修复待修改区域的边界，依据边界外正常的像素向内修复）
#2、算法INPAINT_NS介绍：
        # 通过匹配待修复区域的梯度相邻来延伸等光强线，灰度相等的点连成线，通过填充颜色使区域内的灰度值变化最小
img = cv2.imread('./images/1.png')
# 图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度二值化
_, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY_INV)
#创建掩模图像
_,mask = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
#添加修复算法（两个方法）
#INPAINT_TELEA算法
#dst = cv2.inpaint(img,mask,10,cv2.INPAINT_TELEA) #10为领域大小
#INPAINT_NS算法
dst = cv2.inpaint(img,mask,5,cv2.INPAINT_NS) #10为领域大小


# 【2】显示图像+窗口操作
cv2.namedWindow('image', cv2.WINDOW_NORMAL)# 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
cv2.imshow('image',mask) #掩模图
cv2.namedWindow('img1', cv2.WINDOW_NORMAL)# 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
cv2.imshow('img1',dst) #修改处理后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()


