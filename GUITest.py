import cv2
import matplotlib.pyplot as plt
import os
import numpy

class GUI:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def read_show_save_img(file_pathname, outfile):
        """
        读取、显示、存储图像
        """
        for filename in os.listdir(file_pathname):
            print(filename)
            # 【1】读入图像
            img = cv2.imread(file_pathname+'/'+filename, 1)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道
            #print(img[0][0])
            #BGR=numpy.array([255,218,170])
            #upper =BGR+10
            #lower =BGR-10
            #mask =cv2.inRange(img,lower,upper)
            # 【2】显示图像+窗口操作
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
            cv2.imshow('image', img)  # 窗口会自动调整为图像大小。第一个参数是窗口的名字
            k = cv2.waitKey(0)
            # 等待特定的几毫秒，看是否有键盘输入。有输入，返回键ascll码，无输入，返回-1.
            # 若参数为0，则无限期等待键盘输入。
            if k == 27:  # esc键
                cv2.destroyWindow('image')  # 清除某个窗口
                cv2.destroyAllWindows()  # 清除所有窗口
            elif k == ord('s'):
                # 【3】保存图像
                cv2.imwrite('./outfile'+"/"+filename, img)
                cv2.destroyAllWindows()  # 清除所有窗口

    read_show_save_img("./images",'./outfile')
