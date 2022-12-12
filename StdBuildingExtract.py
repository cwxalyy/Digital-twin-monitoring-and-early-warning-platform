import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from PIL import Image
import image_utils as utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BgrList=[]



class GUI:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def read_show_save_img(file_pathname, outfile):
        """
        读取、显示、存储图像
        """
        for filename in os.listdir(file_pathname):
            if filename.endswith('.png'):
                print(filename)
                print("文件路径=" + file_pathname + '/' + filename)
                path = file_pathname + '/' + filename
                # 【1】读入图像
                img = cv2.imread(path, 1)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道
                # print(img[0][0])
                WallBGR = np.array([255,98,255])
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

                (_, WallContours, hierarchy) = cv2.findContours(Wallmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                (_, DoorContours, hierarchy) = cv2.findContours(Doormask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # print(WallContours[0][0][0])
                # print(WallContours[0].shape)

                # for i in range(len(WallContours)):
                #     print("i======= :%d" % i)
                #    # print(i,WallContours[i])
                #     resultArray = []
                #     for j in range(len(WallContours[i])):
                #         print("j======= :%d" % j)
                #         print(j,WallContours[i][j])
                #         b = np.ones(1)
                #         c =np.insert(WallContours[i][j], 2, values=b, axis=1)
                #         print(c)
                #         #c = np.concatenate((c, [c]))  # 先将p_变成list形式进行拼接，注意输入为一个tuple:
                #         # print(c)
                #         # resultArray=np.concatenate(resultArray,[c])
                #         # print(resultArray)
                #         # for k in range(len(WallContours[i][j])):
                #         #     print(k, WallContours[i][j][k])
                #         #     print("k======= :%d" % k)
                #         #     b = np.ones(1)
                #         #     c = np.insert(WallContours[i][j][k], 2, values=b, axis=1)
                #         #     print(c)
                #         #     # np.append(WallContours[i][j][k][1],10);
                #         #     print(WallContours[i][j][k])
                resultArrayToDoor = [0, 0, 0]
                for item in DoorContours:
                    for item1 in item :
                        for item2 in item1:
                            b = np.ones(1) *8
                            c = np.insert(item2, 2, values=b, axis=0)
                            print(c)
                            resultArrayToDoor = np.vstack((resultArrayToDoor, c))
                    #     resultArray= np.vstack((resultArray, resultArray))
                    # resultArray = np.vstack((resultArray, resultArray))
                print("k======= +++++++++++++++:%d")
                print(resultArrayToDoor)

                resultArrayToWall = [0, 0, 0]
                for item in DoorContours:
                    for item1 in item :
                        for item2 in item1:
                            b = np.ones(1) *10
                            c = np.insert(item2, 2, values=b, axis=0)
                            print(c)
                            resultArrayToWall = np.vstack((resultArrayToWall, c))
                    #     resultArray= np.vstack((resultArray, resultArray))
                    # resultArray = np.vstack((resultArray, resultArray))
                print("k======= +++++++++++++++:%d")
                print(resultArrayToWall)
                X1=[]
                Y1=[]
                # Z=[]
                for point in resultArrayToDoor:
                    X1.append(point[0])
                    Y1.append(point[1])
                    # Z.append(point[2])
                print(X1)
                print("门的X结束打印")
                print(Y1)
                print("门的Y结束打印")
                # print(Z)
                # print("门的Z结束打印")


                for point in resultArrayToWall:
                    # print(point[0])
                    # print(point[2])
                    print("point[0]结束打印")
                    X1.append(point[0])
                    Y1.append(point[1])
                    # Z.append(point[2])
                print(X1)
                print("墙的X结束打印")
                print(Y1)
                print("墙的Y结束打印")
                # print(Z)
                # print("墙的Z结束打印")

                # 三维，两个特征
                fig = plt.figure(figsize=(8, 6))  # 设置图标的大小
                ax = fig.add_subplot(111, projection='3d')  # 111的意思是把画布分为1行1列，画在第一个方格内。其实就是整个画布。
                m=len(X1)
                X2=np.random.rand(m)*5

                # 堆叠全1数组和X1以及X2形成样本的矩阵，倒置，用以矩阵乘法
                X = np.vstack((np.full(m, 1), X1, X2)).T
                Y =Y1
                theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),
                                      np.transpose(X)), Y1)
                print(theta)
                # 构造网格 meshgrid函数可以通过延伸矩阵构造多维坐标中的网格坐标。
                M, N = np.meshgrid(X1,X2)
                # zip函数构建一个多元祖[(x1,y1),(x2,y2)...],ravel函数将一个多维数组合并成一维数组
                Z = np.array(
                    [theta[1] * d + theta[2] * p + theta[0] for d, p in zip(np.ravel(M), np.ravel(N))]).reshape(M.shape)
                # 根据网格和函数构建图形 suface是一个完整的面
                ax.plot_surface(M, N, Z)
                # scatter是散点图
                ax.scatter(X1, X2, Y, c='r')
                # 设置坐标轴的名称
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                plt.show()


                #此时已经具备三维的雏形 ，可以引入Python3d进行生成
                print("number of contours :%d" % (len(WallContours)))
                print("number of contours :%d" % (len(DoorContours)))
                whiteImg=cv2.imread('./backgroud/1.png', 1)
                Whitebroad = whiteImg.copy()
                cv2.drawContours(Whitebroad, WallContours, -1, (0, 0, 255), 2)
                #在这可以做改进，可以画到不同板子上
                cv2.drawContours(Whitebroad, DoorContours, -1, (255, 0, 0), 2)
                # 【2】显示图像+窗口 操作
                cv2.namedWindow('ExtractWindow', cv2.WINDOW_NORMAL)
                # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
                # print(img is None)
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
                    continue


    read_show_save_img("./images", './outfile')


