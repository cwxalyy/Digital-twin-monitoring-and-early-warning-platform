#cv.imread()
#cv.imshow()
#cv.imwrite()
#GUI的实现，图片的读取，展示，写出  灰度模式读取图像
#自己搭建GUI，因为要处理县志图片，初始化self，是摄像机照射下的图片
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def __init__(self):
    # 我的图片文件夹路径
    self.path = '/Users/chenwangxin/Desktop/mapimg'
    filelist = os.listdir(self.path)
    count = 1
    for file in filelist:
        print(file)
    for file in filelist:  # 遍历所有文件
        Olddir = os.path.join(self.path, file)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        count += 1
def read_show_save_img(self, outfile): # self先用具体路径代替，而后再由之后创建数组储存图片，河流
    # 【1】读入图像
    img = cv.imread(self.infile, 0)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道; 第一个参数是一个具体的图片路径
    # 【2】显示图像+窗口操作
    cv.namedWindow('image', cv.WINDOW_NORMAL) # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
    cv.imshow('image', img)  # 窗口会自动调整为图像大小。第一个参数是窗口的名字，第二个参数就是我们的图像
    k = cv.waitKey(0) # 等待特定的几毫秒，看是否有键盘输入。有输入，返回键ascll码，无输入，返回-1.
                      # 一个键盘绑定函数，若参数为0，则无限期等待键盘输入。
    if k == 27:  # esc键
        cv.destroyWindow('image')  # 清除某个窗口
        cv.destroyAllWindows()  # 清除所有窗口
    elif k == ord('s'):
    # 【3】保存图像 's'保存退出
        cv.imwrite(outfile, img) #第一个是文件名，第二个是你想保存的图片
        cv.destroyAllWindows()  # 清除所有窗口

    #显示灰度图像
def show_gray_img_in_plt(self, img):
    img = cv.imread(self.infile, 0)  # 【1】opencv读入图片,0是以灰度读入
    plt.imshow(img, cmap='gray', interpolation='bicubic')  # 【2】plt读入图片
    plt.show()  # 【3】plt显示图片

