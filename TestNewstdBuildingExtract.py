import signal

import cv2
import os
import numpy as np
import image_utils as utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BgrList = []


class GUI:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
        # 水平方向上的高斯卷积核
        gaussKenrnel_x = cv2.getGaussianKernel(sigma, W, cv2.CV_64F)
        # 进行转置
        gaussKenrnel_x = np.transpose(gaussKenrnel_x)
        # 图像矩阵与水平高斯核卷积
        gaussBlur_x = signal.convolve2d(image, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
        # 构建垂直方向上的卷积核
        gaussKenrnel_y = cv2.getGaussianKernel(sigma, H, cv2.CV_64F)
        # 图像与垂直方向上的高斯核卷积核
        gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKenrnel_y, mode='same', boundary=_boundary,
                                         fillvalue=_fillvalue)
        return gaussBlur_xy
    def read_show_save_img(file_pathname, outfile):

        for filename in os.listdir(file_pathname):
            if filename.endswith('.png'):
                print(filename)
                print("文件路径=" + file_pathname + '/' + filename)
                path = file_pathname + '/' + filename
                # 【1】读入图像
                img = cv2.imread(path, 1)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道
                # print(img[0][0])
                WallBGR = np.array([255, 98, 255])
                Wallupper = WallBGR + 50
                Walllower = WallBGR - 50
                Wallmask = cv2.inRange(img, Walllower, Wallupper)

                DoorBGR = np.array([255, 225, 98])
                Doorupper = DoorBGR + 50
                Doorlower = DoorBGR - 50
                Doormask = cv2.inRange(img, Doorlower, Doorupper)
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

                (_, WallContours, hierarchy) = cv2.findContours(Wallmask.copy(), cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)
                (_, DoorContours, hierarchy) = cv2.findContours(Doormask.copy(), cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)

                plt.figure()
                ax = plt.subplot(111, projection='3d')

                ax.set_xlim(0, 600)  # X轴，横向向右方向
                ax.set_ylim(600, 0)  # Y轴,左向与X,Z轴互为垂直
                ax.set_zlim(0, 600)  # 竖向为Z轴
                # 叠层给出3D
                for z in range(120):
                    #生成3D门
                    for item in DoorContours:
                        resultArrayToDoor = []   #两个门分别画出
                        for item1 in item:
                            for item2 in item1:
                                b = np.ones(1) * z
                                c = np.insert(item2, 2, values=b, axis=0)
                                # print(c)
                                resultArrayToDoor.append(c)
                        resultArrayToDoor=np.array(resultArrayToDoor)
                        X = []
                        Y = []
                        Z = []
                        for point in resultArrayToDoor:
                            X.append(point[0])
                            Y.append(point[1])
                            Z.append(point[2])
                        ax.plot3D(X, Y, Z, c='black')
                    #生成3D墙
                    for item in WallContours:
                        resultArrayToWall = []   # 17堵墙分别画出
                        for item1 in item:
                            for item2 in item1:
                                b = np.ones(1) * z
                                c = np.insert(item2, 2, values=b, axis=0)
                                # print(c)
                                resultArrayToWall.append(c)
                        # resultArray = np.vstack((resultArray, resultArray))
                        resultArrayToWall=np.array(resultArrayToWall)
                        X = []
                        Y = []
                        Z = []
                        for point in resultArrayToWall:
                            X.append(point[0])
                            Y.append(point[1])
                            Z.append(point[2])
                        ax.plot3D(X, Y, Z, c='grey')
                    # print("k======= +++++++++++++++:%d")
                    # print(resultArrayToWall)

                    # ax.scatter3D(X, Y, Z, c='blue', s=2)  # 绘制带o折线
                plt.show()


                # 接下来的操作应该是将二维数组转化为三维
                # 因为是一个numpy数组
                # 再利用python3d实现模型复现
                # for item in WallContours:
                #     print(item.shape)
                #     #生成只有一列的拼接矩阵
                #     print(item.shape[0],item.shape[1],1)
                #     T=numpy.full((item.shape[0],item.shape[1],1),1)
                #     print(T)
                #     # 开始拼接两个矩阵，而后循环
                #     newArray = numpy.vstack((item, T))
                # print(newArray)
                # 此时已经具备三维的雏形 ，可以引入Python3d进行生成
                print("number of Wallcontours :%d" % (len(WallContours)))
                print("number of Doorcontours :%d" % (len(DoorContours)))
                whiteImg = cv2.imread('./backgroud/1.png', 1)
                Whitebroad = whiteImg.copy()
                cv2.drawContours(Whitebroad, WallContours, -1, (132,133,135), 2)
                # 在这可以做改进，可以画到不同板子上
                cv2.drawContours(Whitebroad, DoorContours, -1, (0, 49, 83), 2)
                # 【2】显示图像+窗口 操作
                cv2.namedWindow('ExtractWindow', cv2.WINDOW_NORMAL)
                # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
                # print(img is None)
                # 提取轮廓，将轮廓叠加在原始图像上

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
                    continue

    read_show_save_img("./images", './outfile')
