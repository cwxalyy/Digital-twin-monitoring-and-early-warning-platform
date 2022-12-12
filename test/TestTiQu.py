import cv2
ColorImage =cv2.imread('./images/1.png')
BlackWimg = cv2.imread('./outfile/1.png')
#先说输入！！！
#1.contour：带有轮廓信息的图像；2. cv2.RETR_TREE：提取轮廓后，输出轮廓信息的组织形式
#3.cv2.CHAIN_APPROX_SIMPLE：指定轮廓的近似办法（压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标）；

#再说输出
# 1._  是跟输入contour类似的一张二值图
#2.contours：list结构，列表中每个元素代表一个边沿信息。
# 每个元素是(x,1,2)的三维向量，x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标；
#3.hierarchy：返回类型是(x,4)的二维ndarray。x和contours里的x是一样的意思。
# 如果输入选择cv2.RETR_TREE，则以树形结构组织输出;
# hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。

#contour = BlackWimg.copy()
(_,contours,hierarchy )= cv2.findContours(BlackWimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("number of contours :%d" %(len(contours)))

TestImg=ColorImage.copy()
cv2.drawContours(TestImg,contours,-1,(0,0,255),2)
cv2.namedWindow('提取轮廓，将轮廓叠加在原始图像上', cv2.WINDOW_NORMAL)
cv2.imshow("提取轮廓，将轮廓叠加在原始图像上",TestImg)
cv2.waitKey(0)