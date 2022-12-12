import cv2
import matplotlib.pyplot as plt
import os
import numpy
import tkinter as tk
from PIL import Image
BgrList=[]


class GUI:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)
    def read_show_save_img(file_pathname, outfile,color):
        #
        # def __init__(self):
        #     # 使用pillow读取图片，获取图片的宽和高
        #     img_pillow = Image.open(path)
        #     img_width = img_pillow.width  # 图片宽度
        #     img_height = img_pillow.height  # 图片高度
        #     self.root = tk.Tk() #创建窗口
        #     self.root.title("标准建筑提取") #窗口标题
        #     self.root.geometry("1100x700") #自定义窗口长宽
        #     self.root.resizable(width=False, height=True)
        #     # 创建画板
        #     self.canvas = tk.Canvas(self.root, width=img_width, height=img_height, bg="white")
        #     self.canvas.pack()
        #     #filename = PhotoImage(file="sunshine.gif")
        #     whiteImage = self.canvas.create_image(img_width, img_height, anchor=NE, image=path)

        """
        读取、显示、存储图像
        """
        for filename in os.listdir(file_pathname):
            if filename.endswith('.png'):
                print(filename)
                print("文件路径=" + file_pathname + '/' + filename)
                path = file_pathname + '/' + filename
                # 【1】读入图像
                img = cv2.imread(path, 0)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道
                # print(img[0][0])
                BGR = numpy.array(color)
                upper = BGR + 50
                lower = BGR - 50
                mask = cv2.inRange(img, lower, upper)

                # 先说输入！！！
                # 1.contour：带有轮廓信息的图像；2. cv2.RETR_TREE：提取轮廓后，输出轮廓信息的组织形式
                # 3.cv2.CHAIN_APPROX_SIMPLE：指定轮廓的近似办法（压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标）；

                # 再说输出
                # 1._  是跟输入contour类似的一张二值图
                # 2.contours：list结构，列表中每个元素代表一个边沿信息。
                # 每个元素是(x,1,2)的三维向量，x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标；
                # 3.hierarchy：返回类型是(x,4)的二维ndarray。x和contours里的x是一样的意思。
                # 如果输入选择cv2.RETR_TREE，则以树形结构组织输出;
                # hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。

                (_, contours, hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("number of contours :%d" % (len(contours)))
                whiteImg=cv2.imread('./backgroud/1.png', 1)
                Whitebroad = whiteImg.copy()
                cv2.drawContours(Whitebroad, contours, -1, (0, 0, 255), 2)
                # 【2】显示图像+窗口操作
                cv2.namedWindow('ExtractWindow', cv2.WINDOW_NORMAL)
                # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
                print(img is None)
                #提取轮廓，将轮廓叠加在原始图像上

                cv2.imshow("ExtractWindow", Whitebroad)  # 窗口会自动调整为图像大小。第一个参数是窗口的名字
                k = cv2.waitKey(0)
                # 等待特定的几毫秒，看是否有键盘输入。有输入，返回键ascll码，无输入，返回-1.
                # 若参数为0，则无限期等待键盘输入。
                if k == 27:  # esc键
                    cv2.destroyWindow('ExtractWindow')  # 清除某个窗口
                    cv2.destroyAllWindows()  # 清除所有窗口
                elif k == ord('s'):
                    # 【3】保存图像
                    cv2.imwrite(outfile + "/" + filename, Whitebroad)
                    cv2.destroyAllWindows()  # 清除所有窗口
    # BgrList.append([98, 234, 234])

    read_show_save_img("./images", './outfile',BgrList)

