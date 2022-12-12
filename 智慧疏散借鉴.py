import numpy as np
import time
import random
from queue import Queue
import tkinter as tk
from tkinter import *
import math
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tkinter import *

myMap = []  #地图数组
# global i   全局变量
i = 0
global a   #全局变量人数
d = 2
a = 500


class Person:                             # 生成人的类
    Normal_Speed = 1.2                    # 速度

    def __init__(self, id, pos_x, pos_y): #创建初始的人类  我能不能多创几个？ 老人，中年，孩子
        self.id = id                      #老人，中年，孩子..
        self.pos = (pos_x, pos_y)         #在原胞自动机的坐标
        self.speed = Person.Normal_Speed  #调用速度接口
        self.savety = False

    def name(self):                       # 生成ID
        return "ID_" + str(self.id)

    def __str__(self):                    #定义其坐标
        return self.name() + " (%d, %d)" % (self.pos[0], self.pos[1])


class People:  # 在定义的地图上生成人群
    def __init__(self, cnt, myMap): #cnt是多少人， myMap是某个地图
        self.list = []              #创建数组
        self.tot = cnt

        self.map = myMap
        # 某时刻 map 上站的人的个数
        # 反映人流密度
        self.rmap = np.zeros((myMap.Length + 2, myMap.Width + 2))
        # map 上总的经过人数
        # 热力图
        self.thmap = np.zeros(((myMap.Length + 2), (myMap.Width + 2)))
        for i in range(cnt):  # 把地图填满人
            pos_x, pos_y = myMap.Random_Valid_Point()   #Random_Valid_Point()底下的随机点数
            self.list.append(Person(i + 1, pos_x, pos_y))
            self.addMapValue(self.rmap, pos_x, pos_y)      #加点，顺便加上人流密度
            self.addMapValue(self.thmap, pos_x, pos_y)     #加点，顺便加上map上经过的总人数

    def setMapValue(self, mp, x, y, val=0):  # 创建地图，mp是二维数组
        x, y = int(x), int(y)
        mp[x][y] = val

    def addMapValue(self, mp, x, y, add=1):  # 往地图上加点
        if mp is self.rmap:
            x, y = int(x), int(y)
            mp[x][y] += add
        else:
            x, y = int(x), int(y)
            mp[x][y] += add

    def getMapValue(self, mp, x, y):  # 获取地图上的值
        x, y = int(x), int(y)
        return mp[x][y]

    def getSpeed(self, p):  # 根据人的密度计算行走速度，这是速度
        x, y = int(p.pos[0]), int(p.pos[1])
        tot = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if self.map.Check_Valid(nx, ny):
                    tot += self.rmap[nx][ny]  #rmap即是人流密度
        # ratio = random.uniform(math.exp(-2*tot/(5*5)), 1.5*math.exp(-2*tot/(5*5)))
        if tot < 2:  # 人越多速度越小
            ratio = random.uniform(1.1, 1.5)
        elif tot < 4:
            ratio = random.uniform(0.7, 1.1)
        elif tot < 7:
            ratio = random.uniform(0.5, 0.7)
        else:
            ratio = random.uniform(0.4, 0.6)
        return Person.Normal_Speed * ratio

    def move(self, p, dire, show=False):  # 移动 两点间
        # 移动
        if show:
            print(p, end=' ')
            print("to", end=' ')
        (now_x, now_y) = p.pos
        self.addMapValue(self.rmap, now_x, now_y, -1) #加点

        (next_x, next_y) = p.pos + MoveTO[dire]
        self.addMapValue(self.rmap, next_x, next_y, 1)
        p.pos = (next_x, next_y)

        if self.map.checkSavefy(p.pos): #是否有界
            p.savety = True
            self.setMapValue(self.rmap, next_x, next_y, 0)

        addThVal = self.getMapValue(self.rmap, next_x, next_y) #获取点
        self.addMapValue(self.thmap, next_x, next_y, addThVal) #调用方法

        if show:
            print(p)    # p是起点

    def run(self):  # 确定两点
        cnt = 0
        for p in self.list:
            if p.savety:  # 出界
                cnt = cnt + 1
                continue
            speed = self.getSpeed(p)
            # speed = p.speed #random.uniform(p.speed-0.1, p.speed+0.1)
            # (now_x, now_y) = p.pos
            choice = []  # 方向
            weigh = []  # 权重

            Dire = [0, 1, 2, 3, 4, 5, 6, 7]
            random.shuffle(Dire)

            for dire in Dire:  #应该是方向？
                dx, dy = MoveTO[dire][0] * speed, MoveTO[dire][1] * speed
                (next_x, next_y) = p.pos[0] + dx, p.pos[1] + dy
                # 下一步能走
                if self.map.Check_Valid(next_x, next_y) and self.getMapValue(self.rmap, next_x, next_y) <= 1:
                    choice.append(dire)
                    weigh.append(self.map.getDeltaP(p.pos, (next_x, next_y)))
            # 有无障碍 ，有无人流密度超过阈值（动态变化）
            if len(choice) > 0:  # 按照势能 判断权重最大的点 最优方向
                index = weigh.index(max(weigh))
                self.move(p, choice[index])
                p.speed = speed
            else:  # 如果人太多了 就不动
                self.addMapValue(self.thmap, p.pos[0], p.pos[1])
                p.speed = speed

            if p.savety:
                cnt = cnt + 1

        return cnt


Direction = {
    "RIGHT": 0, "UP": 1, "LEFT": 2, "DOWN": 3, "NONE": -1
}

MoveTO = []
MoveTO.append(np.array([1, 0]))  # RIGHT
MoveTO.append(np.array([0, -1]))  # UP
MoveTO.append(np.array([-1, 0]))  # LEFT
MoveTO.append(np.array([0, 1]))  # DOWN

MoveTO.append(np.array([1, -1]))
MoveTO.append(np.array([-1, -1]))
MoveTO.append(np.array([-1, 1]))
MoveTO.append(np.array([1, 1]))


# 输入：一条线段的两个端点
# 输出：整点集合
def Init_Exit(P1, P2):  # 设计出口两个端点
    exit = list()  #定义个链表

    if P1[0] == P2[0]:
        x = P1[0]
        for y in range(P1[1], P2[1] + d):
            exit.append((x, y))
    elif P1[1] == P2[1]:
        y = P1[1]
        for x in range(P1[0], P2[0] + d):
            exit.append((x, y))
    # 斜线
    else:
        pass

    return exit


# 两点坐标围成的矩形区域
def Init_Barrier(A, B):  # 设置障碍
    if A[0] > B[0]:
        A, B = B, A

    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]

    if y1 < y2:
        return ((x1, y1), (x2, y2))
    else:
        return ((x1, y2), (x2, y1))


# 两点坐标围成的矩形区域
def Init_Fire(A, B):  # 设置火灾
    if A[0] > B[0]:
        A, B = B, A

    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]

    if y1 < y2:
        return ((x1, y1), (x2, y2))
    else:
        return ((x1, y2), (x2, y1))


def Init_Arrow1(A, B):  # 设置箭头1
    if A[0] > B[0]:
        A, B = B, A

    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]

    if y1 < y2:
        return ((x1, y1), (x2, y2))
    else:
        return ((x1, y2), (x2, y1))


def Init_Arrow2(A, B, C):  # 设置箭头2
    #   if A[0] > B[0]:
    #       A , B = B , A

    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]
    x3, y3 = C[0], C[1]

    #    if y1 < y2:
    #        return ((x1 , y1) , (x2 , y2))
    #    else:
    return ((x1, y1), (x2, y2), (x3, y3))


#
# 外墙宽度
# Size
Outer_Size = 1


# 障碍 	-1
# LEFT 	0
# UP	1
# RIGHT 2
# DOWN  3


class Map:  # 地图存放
    def __init__(self, L, W, E, B, F, A1, A2):
        self.Length = L #地图长度
        self.Width = W  #地图宽度
        self.Exit = E   #地图出口
        self.Barrier = B #地图障碍
        self.Fire = F    #火灾触发地
        self.Arrow1 = A1
        self.Arrow2 = A2
        self.barrier_list = []  #框架链表
        self.fire_list = []      #火警区域
        self.arrow1_list = []      #箭头一链表
        self.arrow2_list = []       #箭头二链表
        # 0~L+1
        # 0~W+1
        # 势能
        # 出口为 1
        # 障碍为 inf
        self.space = np.zeros((self.Length + Outer_Size * 2, self.Width + Outer_Size * 2))  # 创建空数组
        for j in range(0, self.Width + Outer_Size * 2):  # 设计地图边缘
            self.space[0][j] = self.space[L + 1][j] = float("inf")  # 边缘 势能无穷大
            self.barrier_list.append((0, j))
            self.barrier_list.append((L + 1, j))

        for i in range(0, self.Length + Outer_Size * 2):
            self.space[i][0] = self.space[i][W + 1] = float("inf")
            self.barrier_list.append((i, 0))
            self.barrier_list.append((i, W + 1))

        for (A, B) in self.Barrier:
            for i in range(A[0], B[0] + 1):
                for j in range(A[1], B[1] + 1):
                    self.space[i][j] = float("inf")
                    self.barrier_list.append((i, j))
        for (A, B) in self.Fire:
            for i in range(A[0], B[0] + 1):
                for j in range(A[1], B[1] + 1):
                    self.space[i][j] = float("inf")
                    self.fire_list.append((i, j))
        # 出口
        for (ex, ey) in self.Exit:  # 出口势能设为1
            self.space[ex][ey] = 1
            if ex == self.Length:
                self.space[ex + 1][ey] = 1
            if ey == self.Width:
                self.space[ex][ey + 1] = 1
            # #print("%d %d"%(ex, ey))
            if (ex, ey) in self.barrier_list:
                self.barrier_list.remove((ex, ey))

        # #print(self.barrier_list)

        # #print(type(self.space))
        #
        # 显示全部
        # #print(self.space)

        self.Init_Potential()

    # self.print(self.space)

    def print(self, mat):
        for line in mat:
            for v in line:
                print(v, end=' ')
            print("")

    def Check_Valid(self, x, y):  # 判断是否为障碍
        # pass
        x, y = int(x), int(y)
        if x > self.Length + 1 or x < 0 or y > self.Width + 1 or y < 0:
            return False

        if self.space[x][y] == float("inf"):
            return False
        else:
            return True

    def checkSavefy(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if x == self.Length + 1:  # 判断是否出界  如果出界回去
            x -= 1
        elif x == -1:
            x += 1
        if y == self.Width + 1:
            y -= 1
        elif y == -1:
            y -= 0

        if (x, y) in self.Exit:
            return True
        else:
            return False

    def getDeltaP(self, P1, P2):  # 算势能差
        x1, y1 = int(P1[0]), int(P1[1])
        x2, y2 = int(P2[0]), int(P2[1])
        return self.space[x1][y1] - self.space[x2][y2]

    def Init_Potential(self):  #
        minDis = np.zeros((self.Length + Outer_Size * 2, self.Width + Outer_Size * 2))
        for i in range(self.Length + Outer_Size * 2):
            for j in range(self.Width + Outer_Size * 2):
                minDis[i][j] = float("inf")

        # #print(minDis)
        for (sx, sy) in self.Exit:
            # print(sx, sy)
            tmp = self.BFS(sx, sy)
            # self.#print(tmp)
            # print("----")
            for i in range(self.Length + Outer_Size * 2):
                for j in range(self.Width + Outer_Size * 2):
                    minDis[i][j] = min(minDis[i][j], tmp[i][j])

        self.space = minDis

    # return minDis
    # #print(minDis)

    def BFS(self, x, y):  # 支撑势能迭代计算  宽度优先搜索算法
        if not self.Check_Valid(x, y):  #如果不是障碍
            return

        tmpDis = np.zeros((self.Length + Outer_Size * 2, self.Width + Outer_Size * 2)) #准备数组
        for i in range(self.Length + Outer_Size * 2):
            for j in range(self.Width + Outer_Size * 2):
                tmpDis[i][j] = self.space[i][j] # 点迭代

        queue = Queue()
        queue.put((x, y))
        tmpDis[x][y] = 1
        while not queue.empty():
            (x, y) = queue.get() #队列录入
            dis = tmpDis[x][y]
            # if dis>0:
            # 	continue

            for i in range(8):
                move = MoveTO[i]   #移动点迹
                (nx, ny) = (x, y) + move
                if self.Check_Valid(nx, ny) and tmpDis[nx][ny] == 0:
                    queue.put((nx, ny))
                    tmpDis[nx][ny] = dis + (1.0 if i < 4 else 1.4)

        return tmpDis

    def Random_Valid_Point(self):  # 随机找点
        x = random.uniform(1, self.Length + 2)
        y = random.uniform(1, self.Width + 2)
        while not myMap[i].Check_Valid(x, y):
            x = random.uniform(1, self.Length + 2)
            y = random.uniform(1, self.Width + 2)

        return x, y


def Init_Map():
    # 房间长宽
    Length = 60
    Width = 10

    # 出口
    # 点集
    Exit = Init_Exit(P1=(30, 2), P2=(30, 4))
    Exit.extend(Init_Exit(P1=(0, 8), P2=(0, 9)))

    # 障碍 矩形区域
    Barrier = list()
    Barrier.append(Init_Barrier(A=(3, 5), B=(30, 7)))
    Barrier.append(Init_Barrier(A=(10, 6), B=(15, 8)))
    # 火灾 矩形区域
    Fire = list()
    Fire.append(Init_Fire(A=(3, 5), B=(30, 7)))
    # 箭头 矩形区域
    Arrow1 = list()
    Arrow1.append(Init_Arrow1(A=(3, 5), B=(30, 7)))
    # 箭头 三角形区域
    Arrow2 = list()
    Arrow2.append(Init_Arrow2(A=(3, 5), B=(30, 7), C=(5, 7)))
    return Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)


# 房间长宽
Length = 110
Width = 100

# 出口
# 点集
Exit = Init_Exit(P1=(18, 0), P2=(25, 0))
Exit.extend(Init_Exit(P1=(85, 100), P2=(92, 100)))

# 障碍 矩形区域
# 边框
Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(18, 1)))
Barrier.append(Init_Barrier(A=(25, 0), B=(110, 1)))
Barrier.append(Init_Barrier(A=(0, 0), B=(1, 100)))
Barrier.append(Init_Barrier(A=(109, 0), B=(110, 100)))
Barrier.append(Init_Barrier(A=(0, 99), B=(85, 100)))
Barrier.append(Init_Barrier(A=(92, 99), B=(110, 100)))

# 四周房间
Barrier.append(Init_Barrier(A=(16, 0), B=(17, 27)))
Barrier.append(Init_Barrier(A=(16, 30), B=(17, 58)))
Barrier.append(Init_Barrier(A=(16, 61), B=(17, 79)))
Barrier.append(Init_Barrier(A=(16, 82), B=(17, 84)))
Barrier.append(Init_Barrier(A=(93, 16), B=(94, 18)))
Barrier.append(Init_Barrier(A=(93, 21), B=(94, 39)))
Barrier.append(Init_Barrier(A=(93, 42), B=(94, 70)))
Barrier.append(Init_Barrier(A=(93, 73), B=(94, 100)))

Barrier.append(Init_Barrier(A=(26, 16), B=(28, 17)))
Barrier.append(Init_Barrier(A=(31, 16), B=(49, 17)))
Barrier.append(Init_Barrier(A=(52, 16), B=(80, 17)))
Barrier.append(Init_Barrier(A=(83, 16), B=(110, 17)))
Barrier.append(Init_Barrier(A=(0, 83), B=(27, 84)))
Barrier.append(Init_Barrier(A=(30, 83), B=(58, 84)))
Barrier.append(Init_Barrier(A=(61, 83), B=(79, 84)))
Barrier.append(Init_Barrier(A=(82, 83), B=(84, 84)))

Barrier.append(Init_Barrier(A=(0, 31), B=(17, 32)))
Barrier.append(Init_Barrier(A=(0, 62), B=(17, 63)))
Barrier.append(Init_Barrier(A=(31, 83), B=(32, 100)))
Barrier.append(Init_Barrier(A=(62, 83), B=(63, 100)))
Barrier.append(Init_Barrier(A=(83, 83), B=(84, 100)))
Barrier.append(Init_Barrier(A=(26, 0), B=(27, 17)))
Barrier.append(Init_Barrier(A=(47, 0), B=(48, 17)))
Barrier.append(Init_Barrier(A=(78, 0), B=(79, 17)))
Barrier.append(Init_Barrier(A=(93, 37), B=(110, 38)))
Barrier.append(Init_Barrier(A=(93, 68), B=(110, 69)))

# 中央房间
Barrier.append(Init_Barrier(A=(26, 27), B=(84, 28)))
Barrier.append(Init_Barrier(A=(26, 27), B=(27, 73)))
Barrier.append(Init_Barrier(A=(26, 72), B=(28, 73)))
Barrier.append(Init_Barrier(A=(31, 72), B=(51, 73)))
Barrier.append(Init_Barrier(A=(54, 72), B=(56, 73)))
Barrier.append(Init_Barrier(A=(59, 72), B=(79, 73)))
Barrier.append(Init_Barrier(A=(82, 72), B=(84, 73)))
Barrier.append(Init_Barrier(A=(83, 27), B=(84, 73)))
Barrier.append(Init_Barrier(A=(54, 27), B=(56, 73)))

# 火灾 矩形区域
Fire = list()
# Fire.append ( Init_Fire ( A=(17 , 61) , B=(26 , 76) ) )

# 箭头 矩形区域
Arrow1 = list()
Arrow1.append(Init_Arrow1(A=(20, 8), B=(22, 12)))
Arrow1.append(Init_Arrow1(A=(20, 48), B=(22, 52)))
Arrow1.append(Init_Arrow1(A=(88, 46), B=(90, 50)))
Arrow1.append(Init_Arrow1(A=(88, 88), B=(90, 92)))
Arrow1.append(Init_Arrow1(A=(54, 21), B=(58, 23)))
Arrow1.append(Init_Arrow1(A=(52, 77), B=(56, 79)))
# 箭头 三角形区域
Arrow2 = list()
Arrow2.append(Init_Arrow2(A=(21, 6), B=(19, 8), C=(23, 8)))
Arrow2.append(Init_Arrow2(A=(21, 46), B=(19, 48), C=(23, 48)))
Arrow2.append(Init_Arrow2(A=(89, 52), B=(87, 50), C=(91, 50)))
Arrow2.append(Init_Arrow2(A=(89, 94), B=(87, 92), C=(91, 92)))
Arrow2.append(Init_Arrow2(A=(52, 22), B=(54, 20), C=(54, 24)))
Arrow2.append(Init_Arrow2(A=(58, 78), B=(56, 80), C=(56, 76)))

myMap1 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 110
Width = 100

# 出口
# 点集
Exit = Init_Exit(P1=(18, 0), P2=(25, 0))
Exit.extend(Init_Exit(P1=(85, 100), P2=(92, 100)))

# 障碍 矩形区域
# 边框
Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(18, 1)))
Barrier.append(Init_Barrier(A=(25, 0), B=(110, 1)))
Barrier.append(Init_Barrier(A=(0, 0), B=(1, 100)))
Barrier.append(Init_Barrier(A=(109, 0), B=(110, 100)))
Barrier.append(Init_Barrier(A=(0, 99), B=(85, 100)))
Barrier.append(Init_Barrier(A=(92, 99), B=(110, 100)))

# 四周房间
Barrier.append(Init_Barrier(A=(16, 0), B=(17, 27)))
Barrier.append(Init_Barrier(A=(16, 30), B=(17, 58)))
Barrier.append(Init_Barrier(A=(16, 61), B=(17, 79)))
Barrier.append(Init_Barrier(A=(16, 82), B=(17, 84)))
Barrier.append(Init_Barrier(A=(93, 16), B=(94, 18)))
Barrier.append(Init_Barrier(A=(93, 21), B=(94, 39)))
Barrier.append(Init_Barrier(A=(93, 42), B=(94, 70)))
Barrier.append(Init_Barrier(A=(93, 73), B=(94, 100)))

Barrier.append(Init_Barrier(A=(26, 16), B=(28, 17)))
Barrier.append(Init_Barrier(A=(31, 16), B=(49, 17)))
Barrier.append(Init_Barrier(A=(52, 16), B=(80, 17)))
Barrier.append(Init_Barrier(A=(83, 16), B=(110, 17)))
Barrier.append(Init_Barrier(A=(0, 83), B=(27, 84)))
Barrier.append(Init_Barrier(A=(30, 83), B=(58, 84)))
Barrier.append(Init_Barrier(A=(61, 83), B=(79, 84)))
Barrier.append(Init_Barrier(A=(82, 83), B=(84, 84)))

Barrier.append(Init_Barrier(A=(0, 31), B=(17, 32)))
Barrier.append(Init_Barrier(A=(0, 62), B=(17, 63)))
Barrier.append(Init_Barrier(A=(31, 83), B=(32, 100)))
Barrier.append(Init_Barrier(A=(62, 83), B=(63, 100)))
Barrier.append(Init_Barrier(A=(83, 83), B=(84, 100)))
Barrier.append(Init_Barrier(A=(26, 0), B=(27, 17)))
Barrier.append(Init_Barrier(A=(47, 0), B=(48, 17)))
Barrier.append(Init_Barrier(A=(78, 0), B=(79, 17)))
Barrier.append(Init_Barrier(A=(93, 37), B=(110, 38)))
Barrier.append(Init_Barrier(A=(93, 68), B=(110, 69)))

# 中央房间
Barrier.append(Init_Barrier(A=(26, 27), B=(84, 28)))
Barrier.append(Init_Barrier(A=(26, 27), B=(27, 73)))
Barrier.append(Init_Barrier(A=(26, 72), B=(28, 73)))
Barrier.append(Init_Barrier(A=(31, 72), B=(51, 73)))
Barrier.append(Init_Barrier(A=(54, 72), B=(56, 73)))
Barrier.append(Init_Barrier(A=(59, 72), B=(79, 73)))
Barrier.append(Init_Barrier(A=(82, 72), B=(84, 73)))
Barrier.append(Init_Barrier(A=(83, 27), B=(84, 73)))
Barrier.append(Init_Barrier(A=(54, 27), B=(56, 73)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(65, 73), B=(75, 83)))

# 箭头 矩形区域
Arrow1 = list()
Arrow1.append(Init_Arrow1(A=(20, 8), B=(22, 12)))
Arrow1.append(Init_Arrow1(A=(20, 48), B=(22, 52)))
Arrow1.append(Init_Arrow1(A=(88, 46), B=(90, 50)))
Arrow1.append(Init_Arrow1(A=(88, 88), B=(90, 92)))
Arrow1.append(Init_Arrow1(A=(54, 21), B=(58, 23)))
Arrow1.append(Init_Arrow1(A=(52, 77), B=(56, 79)))
# 箭头 三角形区域
Arrow2 = list()
Arrow2.append(Init_Arrow2(A=(21, 6), B=(19, 8), C=(23, 8)))
Arrow2.append(Init_Arrow2(A=(21, 46), B=(19, 48), C=(23, 48)))
Arrow2.append(Init_Arrow2(A=(89, 52), B=(87, 50), C=(91, 50)))
Arrow2.append(Init_Arrow2(A=(89, 94), B=(87, 92), C=(91, 92)))
Arrow2.append(Init_Arrow2(A=(52, 22), B=(54, 20), C=(54, 24)))
Arrow2.append(Init_Arrow2(A=(58, 78), B=(56, 80), C=(56, 76)))

myMap2 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 110
Width = 100

# 出口
# 点集
Exit = Init_Exit(P1=(18, 0), P2=(25, 0))
Exit.extend(Init_Exit(P1=(85, 100), P2=(92, 100)))

# 障碍 矩形区域
# 边框
Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(18, 1)))
Barrier.append(Init_Barrier(A=(25, 0), B=(110, 1)))
Barrier.append(Init_Barrier(A=(0, 0), B=(1, 100)))
Barrier.append(Init_Barrier(A=(109, 0), B=(110, 100)))
Barrier.append(Init_Barrier(A=(0, 99), B=(85, 100)))
Barrier.append(Init_Barrier(A=(92, 99), B=(110, 100)))

# 四周房间
Barrier.append(Init_Barrier(A=(16, 0), B=(17, 27)))
Barrier.append(Init_Barrier(A=(16, 30), B=(17, 58)))
Barrier.append(Init_Barrier(A=(16, 61), B=(17, 79)))
Barrier.append(Init_Barrier(A=(16, 82), B=(17, 84)))
Barrier.append(Init_Barrier(A=(93, 16), B=(94, 18)))
Barrier.append(Init_Barrier(A=(93, 21), B=(94, 39)))
Barrier.append(Init_Barrier(A=(93, 42), B=(94, 70)))
Barrier.append(Init_Barrier(A=(93, 73), B=(94, 100)))

Barrier.append(Init_Barrier(A=(26, 16), B=(28, 17)))
Barrier.append(Init_Barrier(A=(31, 16), B=(49, 17)))
Barrier.append(Init_Barrier(A=(52, 16), B=(80, 17)))
Barrier.append(Init_Barrier(A=(83, 16), B=(110, 17)))
Barrier.append(Init_Barrier(A=(0, 83), B=(27, 84)))
Barrier.append(Init_Barrier(A=(30, 83), B=(58, 84)))
Barrier.append(Init_Barrier(A=(61, 83), B=(79, 84)))
Barrier.append(Init_Barrier(A=(82, 83), B=(84, 84)))

Barrier.append(Init_Barrier(A=(0, 31), B=(17, 32)))
Barrier.append(Init_Barrier(A=(0, 62), B=(17, 63)))
Barrier.append(Init_Barrier(A=(31, 83), B=(32, 100)))
Barrier.append(Init_Barrier(A=(62, 83), B=(63, 100)))
Barrier.append(Init_Barrier(A=(83, 83), B=(84, 100)))
Barrier.append(Init_Barrier(A=(26, 0), B=(27, 17)))
Barrier.append(Init_Barrier(A=(47, 0), B=(48, 17)))
Barrier.append(Init_Barrier(A=(78, 0), B=(79, 17)))
Barrier.append(Init_Barrier(A=(93, 37), B=(110, 38)))
Barrier.append(Init_Barrier(A=(93, 68), B=(110, 69)))

# 中央房间
Barrier.append(Init_Barrier(A=(26, 27), B=(84, 28)))
Barrier.append(Init_Barrier(A=(26, 27), B=(27, 73)))
Barrier.append(Init_Barrier(A=(26, 72), B=(28, 73)))
Barrier.append(Init_Barrier(A=(31, 72), B=(51, 73)))
Barrier.append(Init_Barrier(A=(54, 72), B=(56, 73)))
Barrier.append(Init_Barrier(A=(59, 72), B=(79, 73)))
Barrier.append(Init_Barrier(A=(82, 72), B=(84, 73)))
Barrier.append(Init_Barrier(A=(83, 27), B=(84, 73)))
Barrier.append(Init_Barrier(A=(54, 27), B=(56, 73)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(33, 17), B=(45, 27)))

# 箭头 矩形区域
Arrow1 = list()
Arrow1.append(Init_Arrow1(A=(20, 8), B=(22, 12)))
Arrow1.append(Init_Arrow1(A=(20, 48), B=(22, 52)))
Arrow1.append(Init_Arrow1(A=(88, 46), B=(90, 50)))
Arrow1.append(Init_Arrow1(A=(88, 88), B=(90, 92)))
Arrow1.append(Init_Arrow1(A=(52, 21), B=(56, 23)))
Arrow1.append(Init_Arrow1(A=(52, 77), B=(56, 79)))
# 箭头 三角形区域
Arrow2 = list()
Arrow2.append(Init_Arrow2(A=(21, 6), B=(19, 8), C=(23, 8)))
Arrow2.append(Init_Arrow2(A=(21, 46), B=(19, 48), C=(23, 48)))
Arrow2.append(Init_Arrow2(A=(89, 52), B=(87, 50), C=(91, 50)))
Arrow2.append(Init_Arrow2(A=(89, 94), B=(87, 92), C=(91, 92)))
Arrow2.append(Init_Arrow2(A=(58, 22), B=(56, 24), C=(56, 20)))
Arrow2.append(Init_Arrow2(A=(58, 78), B=(56, 80), C=(56, 76)))

myMap3 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 110
Width = 100

# 出口
# 点集
Exit = Init_Exit(P1=(18, 0), P2=(25, 0))
Exit.extend(Init_Exit(P1=(85, 100), P2=(92, 100)))

# 障碍 矩形区域
# 边框
Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(18, 1)))
Barrier.append(Init_Barrier(A=(25, 0), B=(110, 1)))
Barrier.append(Init_Barrier(A=(0, 0), B=(1, 100)))
Barrier.append(Init_Barrier(A=(109, 0), B=(110, 100)))
Barrier.append(Init_Barrier(A=(0, 99), B=(85, 100)))
Barrier.append(Init_Barrier(A=(92, 99), B=(110, 100)))

# 四周房间
Barrier.append(Init_Barrier(A=(16, 0), B=(17, 27)))
Barrier.append(Init_Barrier(A=(16, 30), B=(17, 58)))
Barrier.append(Init_Barrier(A=(16, 61), B=(17, 79)))
Barrier.append(Init_Barrier(A=(16, 82), B=(17, 84)))
Barrier.append(Init_Barrier(A=(93, 16), B=(94, 18)))
Barrier.append(Init_Barrier(A=(93, 21), B=(94, 39)))
Barrier.append(Init_Barrier(A=(93, 42), B=(94, 70)))
Barrier.append(Init_Barrier(A=(93, 73), B=(94, 100)))

Barrier.append(Init_Barrier(A=(26, 16), B=(28, 17)))
Barrier.append(Init_Barrier(A=(31, 16), B=(49, 17)))
Barrier.append(Init_Barrier(A=(52, 16), B=(80, 17)))
Barrier.append(Init_Barrier(A=(83, 16), B=(110, 17)))
Barrier.append(Init_Barrier(A=(0, 83), B=(27, 84)))
Barrier.append(Init_Barrier(A=(30, 83), B=(58, 84)))
Barrier.append(Init_Barrier(A=(61, 83), B=(79, 84)))
Barrier.append(Init_Barrier(A=(82, 83), B=(84, 84)))

Barrier.append(Init_Barrier(A=(0, 31), B=(17, 32)))
Barrier.append(Init_Barrier(A=(0, 62), B=(17, 63)))
Barrier.append(Init_Barrier(A=(31, 83), B=(32, 100)))
Barrier.append(Init_Barrier(A=(62, 83), B=(63, 100)))
Barrier.append(Init_Barrier(A=(83, 83), B=(84, 100)))
Barrier.append(Init_Barrier(A=(26, 0), B=(27, 17)))
Barrier.append(Init_Barrier(A=(47, 0), B=(48, 17)))
Barrier.append(Init_Barrier(A=(78, 0), B=(79, 17)))
Barrier.append(Init_Barrier(A=(93, 37), B=(110, 38)))
Barrier.append(Init_Barrier(A=(93, 68), B=(110, 69)))

# 中央房间
Barrier.append(Init_Barrier(A=(26, 27), B=(84, 28)))
Barrier.append(Init_Barrier(A=(26, 27), B=(27, 73)))
Barrier.append(Init_Barrier(A=(26, 72), B=(28, 73)))
Barrier.append(Init_Barrier(A=(31, 72), B=(51, 73)))
Barrier.append(Init_Barrier(A=(54, 72), B=(56, 73)))
Barrier.append(Init_Barrier(A=(59, 72), B=(79, 73)))
Barrier.append(Init_Barrier(A=(82, 72), B=(84, 73)))
Barrier.append(Init_Barrier(A=(83, 27), B=(84, 73)))
Barrier.append(Init_Barrier(A=(54, 27), B=(56, 73)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(17, 64), B=(26, 70)))

# 箭头 矩形区域
Arrow1 = list()
Arrow1.append(Init_Arrow1(A=(20, 8), B=(22, 12)))
Arrow1.append(Init_Arrow1(A=(20, 48), B=(22, 52)))
Arrow1.append(Init_Arrow1(A=(88, 46), B=(90, 50)))
Arrow1.append(Init_Arrow1(A=(88, 88), B=(90, 92)))
Arrow1.append(Init_Arrow1(A=(54, 21), B=(58, 23)))
Arrow1.append(Init_Arrow1(A=(52, 77), B=(56, 79)))
# 箭头 三角形区域
Arrow2 = list()
Arrow2.append(Init_Arrow2(A=(21, 6), B=(19, 8), C=(23, 8)))
Arrow2.append(Init_Arrow2(A=(21, 46), B=(19, 48), C=(23, 48)))
Arrow2.append(Init_Arrow2(A=(89, 52), B=(87, 50), C=(91, 50)))
Arrow2.append(Init_Arrow2(A=(89, 94), B=(87, 92), C=(91, 92)))
Arrow2.append(Init_Arrow2(A=(52, 22), B=(54, 20), C=(54, 24)))
Arrow2.append(Init_Arrow2(A=(58, 78), B=(56, 80), C=(56, 76)))

myMap4 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 220
Width = 110

# 出口
# 点集
Exit = Init_Exit(P1=(220, 70), P2=(220, 80))
Exit.extend(Init_Exit(P1=(100, 0), P2=(108, 0)))
Exit.extend(Init_Exit(P1=(29, 12), P2=(29, 17)))
# 障碍 矩形区域
# 边框

Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(2, 110)))
Barrier.append(Init_Barrier(A=(219, 0), B=(220, 70)))
Barrier.append(Init_Barrier(A=(219, 80), B=(220, 110)))
Barrier.append(Init_Barrier(A=(0, 0), B=(100, 1)))
Barrier.append(Init_Barrier(A=(108, 0), B=(220, 1)))
Barrier.append(Init_Barrier(A=(0, 109), B=(220, 110)))

# 下面部分
Barrier.append(Init_Barrier(A=(2, 80), B=(16, 81)))
Barrier.append(Init_Barrier(A=(20, 80), B=(38, 81)))
Barrier.append(Init_Barrier(A=(42, 80), B=(58, 81)))
Barrier.append(Init_Barrier(A=(62, 80), B=(78, 81)))
Barrier.append(Init_Barrier(A=(82, 80), B=(98, 81)))
Barrier.append(Init_Barrier(A=(102, 80), B=(118, 81)))
Barrier.append(Init_Barrier(A=(122, 80), B=(138, 81)))
Barrier.append(Init_Barrier(A=(142, 80), B=(159, 81)))
Barrier.append(Init_Barrier(A=(163, 80), B=(166, 81)))

# 竖框
Barrier.append(Init_Barrier(A=(21, 81), B=(22, 109)))
Barrier.append(Init_Barrier(A=(43, 81), B=(44, 97)))
Barrier.append(Init_Barrier(A=(43, 100), B=(44, 109)))
Barrier.append(Init_Barrier(A=(63, 81), B=(64, 84)))
Barrier.append(Init_Barrier(A=(63, 89), B=(64, 109)))
Barrier.append(Init_Barrier(A=(83, 81), B=(84, 84)))
Barrier.append(Init_Barrier(A=(83, 89), B=(84, 97)))
Barrier.append(Init_Barrier(A=(83, 100), B=(84, 109)))
Barrier.append(Init_Barrier(A=(103, 81), B=(104, 109)))
Barrier.append(Init_Barrier(A=(123, 81), B=(124, 84)))
Barrier.append(Init_Barrier(A=(123, 89), B=(124, 97)))
Barrier.append(Init_Barrier(A=(123, 100), B=(124, 109)))
Barrier.append(Init_Barrier(A=(143, 81), B=(144, 84)))
Barrier.append(Init_Barrier(A=(143, 89), B=(144, 109)))
Barrier.append(Init_Barrier(A=(165, 81), B=(166, 97)))
Barrier.append(Init_Barrier(A=(165, 100), B=(166, 109)))

# 边框 障碍柱子
Barrier.append(Init_Barrier(A=(2, 97), B=(3, 100)))
Barrier.append(Init_Barrier(A=(42, 97), B=(45, 100)))
Barrier.append(Init_Barrier(A=(82, 97), B=(85, 100)))
Barrier.append(Init_Barrier(A=(122, 97), B=(125, 100)))
Barrier.append(Init_Barrier(A=(163, 97), B=(166, 100)))
Barrier.append(Init_Barrier(A=(203, 97), B=(206, 100)))

# 中间部分
Barrier.append(Init_Barrier(A=(0, 56), B=(6, 57)))
Barrier.append(Init_Barrier(A=(0, 64), B=(3, 66)))
Barrier.append(Init_Barrier(A=(0, 66), B=(17, 67)))
Barrier.append(Init_Barrier(A=(9, 56), B=(17, 57)))
Barrier.append(Init_Barrier(A=(16, 56), B=(17, 67)))

Barrier.append(Init_Barrier(A=(25, 56), B=(27, 57)))
Barrier.append(Init_Barrier(A=(25, 66), B=(91, 67)))
Barrier.append(Init_Barrier(A=(25, 56), B=(26, 66)))
Barrier.append(Init_Barrier(A=(30, 56), B=(47, 57)))
Barrier.append(Init_Barrier(A=(42, 64), B=(45, 66)))
Barrier.append(Init_Barrier(A=(45, 56), B=(46, 66)))
Barrier.append(Init_Barrier(A=(50, 56), B=(65, 57)))
Barrier.append(Init_Barrier(A=(63, 56), B=(64, 66)))
Barrier.append(Init_Barrier(A=(68, 56), B=(95, 57)))
Barrier.append(Init_Barrier(A=(82, 56), B=(83, 66)))
Barrier.append(Init_Barrier(A=(82, 64), B=(85, 66)))

Barrier.append(Init_Barrier(A=(95, 45), B=(96, 66)))
Barrier.append(Init_Barrier(A=(95, 45), B=(107, 46)))
Barrier.append(Init_Barrier(A=(94, 66), B=(220, 67)))
Barrier.append(Init_Barrier(A=(111, 45), B=(114, 46)))
Barrier.append(Init_Barrier(A=(112, 45), B=(113, 66)))
Barrier.append(Init_Barrier(A=(122, 64), B=(125, 66)))
Barrier.append(Init_Barrier(A=(118, 45), B=(141, 46)))
Barrier.append(Init_Barrier(A=(129, 45), B=(130, 66)))
Barrier.append(Init_Barrier(A=(145, 45), B=(148, 46)))
Barrier.append(Init_Barrier(A=(146, 45), B=(147, 66)))

Barrier.append(Init_Barrier(A=(152, 45), B=(175, 46)))
Barrier.append(Init_Barrier(A=(163, 45), B=(164, 66)))
Barrier.append(Init_Barrier(A=(163, 64), B=(166, 66)))
Barrier.append(Init_Barrier(A=(179, 45), B=(182, 46)))
Barrier.append(Init_Barrier(A=(180, 45), B=(181, 66)))
Barrier.append(Init_Barrier(A=(186, 45), B=(199, 46)))
Barrier.append(Init_Barrier(A=(197, 45), B=(198, 66)))
Barrier.append(Init_Barrier(A=(203, 64), B=(206, 66)))
Barrier.append(Init_Barrier(A=(203, 45), B=(220, 46)))

# 上面部分
Barrier.append(Init_Barrier(A=(10, 10), B=(30, 11)))
Barrier.append(Init_Barrier(A=(10, 18), B=(43, 20)))
Barrier.append(Init_Barrier(A=(25, 20), B=(27, 22)))
Barrier.append(Init_Barrier(A=(43, 0), B=(45, 35)))
Barrier.append(Init_Barrier(A=(0, 33), B=(17, 35)))
Barrier.append(Init_Barrier(A=(25, 30), B=(27, 35)))
Barrier.append(Init_Barrier(A=(25, 33), B=(45, 35)))
Barrier.append(Init_Barrier(A=(16, 33), B=(17, 39)))
Barrier.append(Init_Barrier(A=(16, 43), B=(17, 48)))
Barrier.append(Init_Barrier(A=(0, 43), B=(7, 44)))
Barrier.append(Init_Barrier(A=(6, 43), B=(7, 48)))
Barrier.append(Init_Barrier(A=(6, 47), B=(17, 48)))
Barrier.append(Init_Barrier(A=(25, 35), B=(26, 48)))
Barrier.append(Init_Barrier(A=(25, 47), B=(27, 48)))
Barrier.append(Init_Barrier(A=(38, 35), B=(39, 48)))
Barrier.append(Init_Barrier(A=(34, 47), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 43), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 35), B=(45, 36)))
# Barrier.append ( Init_Barrier ( A=(29 , 11) , B=(30 , 18) ) )

Barrier.append(Init_Barrier(A=(49, 0), B=(50, 14)))
Barrier.append(Init_Barrier(A=(68, 0), B=(69, 37)))
Barrier.append(Init_Barrier(A=(68, 45), B=(69, 48)))
Barrier.append(Init_Barrier(A=(53, 34), B=(69, 35)))
Barrier.append(Init_Barrier(A=(53, 34), B=(54, 48)))
Barrier.append(Init_Barrier(A=(53, 47), B=(55, 48)))
Barrier.append(Init_Barrier(A=(62, 47), B=(69, 48)))
Barrier.append(Init_Barrier(A=(63, 34), B=(64, 48)))

Barrier.append(Init_Barrier(A=(77, 11), B=(166, 12)))
Barrier.append(Init_Barrier(A=(77, 11), B=(78, 37)))
Barrier.append(Init_Barrier(A=(77, 45), B=(78, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(83, 48)))
Barrier.append(Init_Barrier(A=(77, 34), B=(82, 35)))
Barrier.append(Init_Barrier(A=(77, 47), B=(83, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(90, 36)))
Barrier.append(Init_Barrier(A=(95, 10), B=(96, 36)))
Barrier.append(Init_Barrier(A=(94, 35), B=(107, 36)))
Barrier.append(Init_Barrier(A=(112, 10), B=(113, 36)))
Barrier.append(Init_Barrier(A=(129, 10), B=(130, 36)))
Barrier.append(Init_Barrier(A=(146, 10), B=(147, 36)))
Barrier.append(Init_Barrier(A=(164, 10), B=(166, 36)))
Barrier.append(Init_Barrier(A=(111, 35), B=(114, 36)))
Barrier.append(Init_Barrier(A=(118, 35), B=(141, 36)))
Barrier.append(Init_Barrier(A=(145, 35), B=(148, 36)))
Barrier.append(Init_Barrier(A=(152, 35), B=(166, 36)))

# 柱子
Barrier.append(Init_Barrier(A=(82, 0), B=(85, 3)))
Barrier.append(Init_Barrier(A=(122, 0), B=(125, 3)))
Barrier.append(Init_Barrier(A=(163, 0), B=(166, 3)))
Barrier.append(Init_Barrier(A=(203, 0), B=(206, 3)))
Barrier.append(Init_Barrier(A=(82, 32), B=(85, 35)))
Barrier.append(Init_Barrier(A=(122, 32), B=(125, 35)))
Barrier.append(Init_Barrier(A=(163, 32), B=(166, 35)))
Barrier.append(Init_Barrier(A=(203, 32), B=(206, 35)))

# 火灾 矩形区域
Fire = list()

# 箭头 矩形区域
Arrow1 = list()
# 1
Arrow1.append(Init_Arrow1(A=(20, 51), B=(22, 55)))
# 1+
Arrow1.append(Init_Arrow1(A=(17, 30), B=(13, 28)))
# 2
Arrow1.append(Init_Arrow1(A=(45, 51), B=(49, 53)))
# 3
Arrow1.append(Init_Arrow1(A=(45, 72), B=(49, 74)))
# 4
Arrow1.append(Init_Arrow1(A=(127, 72), B=(131, 74)))
# 5+
Arrow1.append(Init_Arrow1(A=(130, 39), B=(134, 41)))
# 5
Arrow1.append(Init_Arrow1(A=(183, 90), B=(185, 94)))
# 6
Arrow1.append(Init_Arrow1(A=(104, 39), B=(108, 41)))
# 7
Arrow1.append(Init_Arrow1(A=(145, 39), B=(149, 41)))
# 8
Arrow1.append(Init_Arrow1(A=(183, 22), B=(185, 26)))
# 9
Arrow1.append(Init_Arrow1(A=(136, 5), B=(140, 7)))
# 10
Arrow1.append(Init_Arrow1(A=(84, 5), B=(88, 7)))
# 10+
Arrow1.append(Init_Arrow1(A=(81, 78), B=(84, 76)))
# 11
Arrow1.append(Init_Arrow1(A=(48, 30), B=(50, 34)))
# 12
Arrow1.append(Init_Arrow1(A=(190, 72), B=(194, 74)))
# 13
Arrow1.append(Init_Arrow1(A=(72, 23), B=(74, 27)))

# 箭头 三角形区域
Arrow2 = list()
# 1
Arrow2.append(Init_Arrow2(A=(21, 49), B=(19, 51), C=(23, 51)))
# 1+
Arrow2.append(Init_Arrow2(A=(11, 29), B=(13, 26), C=(13, 30)))
# 2
Arrow2.append(Init_Arrow2(A=(43, 52), B=(45, 50), C=(45, 54)))
# 3
Arrow2.append(Init_Arrow2(A=(43, 73), B=(45, 71), C=(45, 75)))
# 4
Arrow2.append(Init_Arrow2(A=(133, 73), B=(131, 75), C=(131, 71)))
# 5
Arrow2.append(Init_Arrow2(A=(184, 88), B=(186, 90), C=(182, 90)))
# 5+
Arrow2.append(Init_Arrow2(A=(129, 38), B=(131, 36), C=(131, 38)))
# 6
Arrow2.append(Init_Arrow2(A=(102, 40), B=(104, 38), C=(104, 42)))
# 7
Arrow2.append(Init_Arrow2(A=(151, 40), B=(149, 42), C=(149, 38)))
# 8
Arrow2.append(Init_Arrow2(A=(184, 20), B=(186, 22), C=(182, 22)))
# 9
Arrow2.append(Init_Arrow2(A=(134, 6), B=(136, 4), C=(136, 8)))
# 10
Arrow2.append(Init_Arrow2(A=(90, 6), B=(88, 8), C=(88, 4)))
# 10+
Arrow2.append(Init_Arrow2(A=(89, 77), B=(86, 75), C=(86, 79)))
# 11
Arrow2.append(Init_Arrow2(A=(49, 36), B=(51, 34), C=(47, 34)))
# 12
Arrow2.append(Init_Arrow2(A=(196, 73), B=(194, 75), C=(194, 71)))
# 13
Arrow2.append(Init_Arrow2(A=(73, 21), B=(75, 23), C=(71, 23)))

myMap5 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 220
Width = 110

# 出口
# 点集
Exit = Init_Exit(P1=(220, 70), P2=(220, 80))
# Exit.extend ( Init_Exit ( P1=(100 , 0) , P2=(108 , 0) ) )
Exit.extend(Init_Exit(P1=(29, 12), P2=(29, 17)))
# 障碍 矩形区域
# 边框

Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(2, 110)))
Barrier.append(Init_Barrier(A=(219, 0), B=(220, 70)))
Barrier.append(Init_Barrier(A=(219, 80), B=(220, 110)))
Barrier.append(Init_Barrier(A=(0, 0), B=(100, 1)))
Barrier.append(Init_Barrier(A=(108, 0), B=(220, 1)))
Barrier.append(Init_Barrier(A=(0, 109), B=(220, 110)))

# 下面部分
Barrier.append(Init_Barrier(A=(2, 80), B=(16, 81)))
Barrier.append(Init_Barrier(A=(20, 80), B=(38, 81)))
Barrier.append(Init_Barrier(A=(42, 80), B=(58, 81)))
Barrier.append(Init_Barrier(A=(62, 80), B=(78, 81)))
Barrier.append(Init_Barrier(A=(82, 80), B=(98, 81)))
Barrier.append(Init_Barrier(A=(102, 80), B=(118, 81)))
Barrier.append(Init_Barrier(A=(122, 80), B=(138, 81)))
Barrier.append(Init_Barrier(A=(142, 80), B=(159, 81)))
Barrier.append(Init_Barrier(A=(163, 80), B=(166, 81)))

# 竖框
Barrier.append(Init_Barrier(A=(21, 81), B=(22, 109)))
Barrier.append(Init_Barrier(A=(43, 81), B=(44, 97)))
Barrier.append(Init_Barrier(A=(43, 100), B=(44, 109)))
Barrier.append(Init_Barrier(A=(63, 81), B=(64, 84)))
Barrier.append(Init_Barrier(A=(63, 89), B=(64, 109)))
Barrier.append(Init_Barrier(A=(83, 81), B=(84, 84)))
Barrier.append(Init_Barrier(A=(83, 89), B=(84, 97)))
Barrier.append(Init_Barrier(A=(83, 100), B=(84, 109)))
Barrier.append(Init_Barrier(A=(103, 81), B=(104, 109)))
Barrier.append(Init_Barrier(A=(123, 81), B=(124, 84)))
Barrier.append(Init_Barrier(A=(123, 89), B=(124, 97)))
Barrier.append(Init_Barrier(A=(123, 100), B=(124, 109)))
Barrier.append(Init_Barrier(A=(143, 81), B=(144, 84)))
Barrier.append(Init_Barrier(A=(143, 89), B=(144, 109)))
Barrier.append(Init_Barrier(A=(165, 81), B=(166, 97)))
Barrier.append(Init_Barrier(A=(165, 100), B=(166, 109)))

# 边框 障碍柱子
Barrier.append(Init_Barrier(A=(2, 97), B=(3, 100)))
Barrier.append(Init_Barrier(A=(42, 97), B=(45, 100)))
Barrier.append(Init_Barrier(A=(82, 97), B=(85, 100)))
Barrier.append(Init_Barrier(A=(122, 97), B=(125, 100)))
Barrier.append(Init_Barrier(A=(163, 97), B=(166, 100)))
Barrier.append(Init_Barrier(A=(203, 97), B=(206, 100)))

# 中间部分
Barrier.append(Init_Barrier(A=(0, 56), B=(6, 57)))
Barrier.append(Init_Barrier(A=(0, 64), B=(3, 66)))
Barrier.append(Init_Barrier(A=(0, 66), B=(17, 67)))
Barrier.append(Init_Barrier(A=(9, 56), B=(17, 57)))
Barrier.append(Init_Barrier(A=(16, 56), B=(17, 67)))

Barrier.append(Init_Barrier(A=(25, 56), B=(27, 57)))
Barrier.append(Init_Barrier(A=(25, 66), B=(91, 67)))
Barrier.append(Init_Barrier(A=(25, 56), B=(26, 66)))
Barrier.append(Init_Barrier(A=(30, 56), B=(47, 57)))
Barrier.append(Init_Barrier(A=(42, 64), B=(45, 66)))
Barrier.append(Init_Barrier(A=(45, 56), B=(46, 66)))
Barrier.append(Init_Barrier(A=(50, 56), B=(65, 57)))
Barrier.append(Init_Barrier(A=(63, 56), B=(64, 66)))
Barrier.append(Init_Barrier(A=(68, 56), B=(95, 57)))
Barrier.append(Init_Barrier(A=(82, 56), B=(83, 66)))
Barrier.append(Init_Barrier(A=(82, 64), B=(85, 66)))

Barrier.append(Init_Barrier(A=(95, 45), B=(96, 66)))
Barrier.append(Init_Barrier(A=(95, 45), B=(107, 46)))
Barrier.append(Init_Barrier(A=(94, 66), B=(220, 67)))
Barrier.append(Init_Barrier(A=(111, 45), B=(114, 46)))
Barrier.append(Init_Barrier(A=(112, 45), B=(113, 66)))
Barrier.append(Init_Barrier(A=(122, 64), B=(125, 66)))
Barrier.append(Init_Barrier(A=(118, 45), B=(141, 46)))
Barrier.append(Init_Barrier(A=(129, 45), B=(130, 66)))
Barrier.append(Init_Barrier(A=(145, 45), B=(148, 46)))
Barrier.append(Init_Barrier(A=(146, 45), B=(147, 66)))

Barrier.append(Init_Barrier(A=(152, 45), B=(175, 46)))
Barrier.append(Init_Barrier(A=(163, 45), B=(164, 66)))
Barrier.append(Init_Barrier(A=(163, 64), B=(166, 66)))
Barrier.append(Init_Barrier(A=(179, 45), B=(182, 46)))
Barrier.append(Init_Barrier(A=(180, 45), B=(181, 66)))
Barrier.append(Init_Barrier(A=(186, 45), B=(199, 46)))
Barrier.append(Init_Barrier(A=(197, 45), B=(198, 66)))
Barrier.append(Init_Barrier(A=(203, 64), B=(206, 66)))
Barrier.append(Init_Barrier(A=(203, 45), B=(220, 46)))

# 上面部分
Barrier.append(Init_Barrier(A=(10, 10), B=(30, 11)))
Barrier.append(Init_Barrier(A=(10, 18), B=(43, 20)))
Barrier.append(Init_Barrier(A=(25, 20), B=(27, 22)))
Barrier.append(Init_Barrier(A=(43, 0), B=(45, 35)))
Barrier.append(Init_Barrier(A=(0, 33), B=(17, 35)))
Barrier.append(Init_Barrier(A=(25, 30), B=(27, 35)))
Barrier.append(Init_Barrier(A=(25, 33), B=(45, 35)))
Barrier.append(Init_Barrier(A=(16, 33), B=(17, 39)))
Barrier.append(Init_Barrier(A=(16, 43), B=(17, 48)))
Barrier.append(Init_Barrier(A=(0, 43), B=(7, 44)))
Barrier.append(Init_Barrier(A=(6, 43), B=(7, 48)))
Barrier.append(Init_Barrier(A=(6, 47), B=(17, 48)))
Barrier.append(Init_Barrier(A=(25, 35), B=(26, 48)))
Barrier.append(Init_Barrier(A=(25, 47), B=(27, 48)))
Barrier.append(Init_Barrier(A=(38, 35), B=(39, 48)))
Barrier.append(Init_Barrier(A=(34, 47), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 43), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 35), B=(45, 36)))
# Barrier.append ( Init_Barrier ( A=(29 , 11) , B=(30 , 18) ) )

Barrier.append(Init_Barrier(A=(49, 0), B=(50, 14)))
Barrier.append(Init_Barrier(A=(68, 0), B=(69, 37)))
Barrier.append(Init_Barrier(A=(68, 45), B=(69, 48)))
Barrier.append(Init_Barrier(A=(53, 34), B=(69, 35)))
Barrier.append(Init_Barrier(A=(53, 34), B=(54, 48)))
Barrier.append(Init_Barrier(A=(53, 47), B=(55, 48)))
Barrier.append(Init_Barrier(A=(62, 47), B=(69, 48)))
Barrier.append(Init_Barrier(A=(63, 34), B=(64, 48)))

Barrier.append(Init_Barrier(A=(77, 11), B=(166, 12)))
Barrier.append(Init_Barrier(A=(77, 11), B=(78, 37)))
Barrier.append(Init_Barrier(A=(77, 45), B=(78, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(83, 48)))
Barrier.append(Init_Barrier(A=(77, 34), B=(82, 35)))
Barrier.append(Init_Barrier(A=(77, 47), B=(83, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(90, 36)))
Barrier.append(Init_Barrier(A=(95, 11), B=(96, 36)))
Barrier.append(Init_Barrier(A=(94, 35), B=(107, 36)))
Barrier.append(Init_Barrier(A=(112, 11), B=(113, 36)))
Barrier.append(Init_Barrier(A=(129, 11), B=(130, 36)))
Barrier.append(Init_Barrier(A=(146, 11), B=(147, 36)))
Barrier.append(Init_Barrier(A=(164, 11), B=(166, 36)))
Barrier.append(Init_Barrier(A=(111, 35), B=(114, 36)))
Barrier.append(Init_Barrier(A=(118, 35), B=(141, 36)))
Barrier.append(Init_Barrier(A=(145, 35), B=(148, 36)))
Barrier.append(Init_Barrier(A=(152, 35), B=(166, 36)))

# 柱子
Barrier.append(Init_Barrier(A=(82, 0), B=(85, 3)))
Barrier.append(Init_Barrier(A=(122, 0), B=(125, 3)))
Barrier.append(Init_Barrier(A=(163, 0), B=(166, 3)))
Barrier.append(Init_Barrier(A=(203, 0), B=(206, 3)))
Barrier.append(Init_Barrier(A=(82, 32), B=(85, 35)))
Barrier.append(Init_Barrier(A=(122, 32), B=(125, 35)))
Barrier.append(Init_Barrier(A=(163, 32), B=(166, 35)))
Barrier.append(Init_Barrier(A=(203, 32), B=(206, 35)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(98, 1), B=(110, 11)))

# 箭头 矩形区域
Arrow1 = list()
# 1
Arrow1.append(Init_Arrow1(A=(20, 51), B=(22, 55)))
# 1+
Arrow1.append(Init_Arrow1(A=(17, 30), B=(13, 28)))
# 2
Arrow1.append(Init_Arrow1(A=(45, 51), B=(49, 53)))
# 3
Arrow1.append(Init_Arrow1(A=(45, 72), B=(49, 74)))
# 4
Arrow1.append(Init_Arrow1(A=(127, 72), B=(131, 74)))
# 5
Arrow1.append(Init_Arrow1(A=(183, 90), B=(185, 94)))
# 5+
Arrow1.append(Init_Arrow1(A=(130, 39), B=(134, 41)))
# 6
Arrow1.append(Init_Arrow1(A=(104, 39), B=(108, 41)))
# 7
Arrow1.append(Init_Arrow1(A=(151, 39), B=(147, 41)))
# 8
Arrow1.append(Init_Arrow1(A=(183, 20), B=(185, 24)))
# 9
Arrow1.append(Init_Arrow1(A=(134, 5), B=(138, 7)))
# 10
Arrow1.append(Init_Arrow1(A=(86, 5), B=(90, 7)))
# 10+
Arrow1.append(Init_Arrow1(A=(81, 78), B=(84, 76)))
# 11
Arrow1.append(Init_Arrow1(A=(48, 30), B=(50, 34)))
# 12
Arrow1.append(Init_Arrow1(A=(190, 72), B=(194, 74)))
# 13
Arrow1.append(Init_Arrow1(A=(72, 21), B=(74, 25)))

# 箭头 三角形区域
Arrow2 = list()
# 1
Arrow2.append(Init_Arrow2(A=(21, 49), B=(19, 51), C=(23, 51)))
# 1+
Arrow2.append(Init_Arrow2(A=(11, 29), B=(13, 27), C=(13, 31)))
# 2
Arrow2.append(Init_Arrow2(A=(43, 52), B=(45, 50), C=(45, 54)))
# 3
Arrow2.append(Init_Arrow2(A=(43, 73), B=(45, 71), C=(45, 75)))
# 4
Arrow2.append(Init_Arrow2(A=(133, 73), B=(131, 75), C=(131, 71)))
# 5
Arrow2.append(Init_Arrow2(A=(184, 88), B=(186, 90), C=(182, 90)))
# 5+
Arrow2.append(Init_Arrow2(A=(128, 40), B=(130, 38), C=(130, 42)))
# 6
Arrow2.append(Init_Arrow2(A=(102, 40), B=(104, 38), C=(104, 42)))
# 7
Arrow2.append(Init_Arrow2(A=(145, 40), B=(147, 38), C=(147, 42)))
# 8
Arrow2.append(Init_Arrow2(A=(184, 26), B=(186, 24), C=(182, 24)))
# 9
Arrow2.append(Init_Arrow2(A=(140, 6), B=(138, 8), C=(138, 4)))
# 10
Arrow2.append(Init_Arrow2(A=(84, 6), B=(86, 8), C=(86, 4)))
# 10+
Arrow2.append(Init_Arrow2(A=(79, 77), B=(81, 75), C=(81, 79)))
# 11
Arrow2.append(Init_Arrow2(A=(49, 36), B=(51, 34), C=(47, 34)))
# 12
Arrow2.append(Init_Arrow2(A=(196, 73), B=(194, 75), C=(194, 71)))
# 13
Arrow2.append(Init_Arrow2(A=(73, 27), B=(75, 25), C=(71, 25)))

myMap6 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 220
Width = 110

# 出口
# 点集
Exit = Init_Exit(P1=(220, 70), P2=(220, 80))
Exit.extend(Init_Exit(P1=(100, 0), P2=(108, 0)))
Exit.extend(Init_Exit(P1=(29, 12), P2=(29, 17)))
# 障碍 矩形区域
# 边框

Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(2, 110)))
Barrier.append(Init_Barrier(A=(219, 0), B=(220, 70)))
Barrier.append(Init_Barrier(A=(219, 80), B=(220, 110)))
Barrier.append(Init_Barrier(A=(0, 0), B=(100, 1)))
Barrier.append(Init_Barrier(A=(108, 0), B=(220, 1)))
Barrier.append(Init_Barrier(A=(0, 109), B=(220, 110)))

# 下面部分
Barrier.append(Init_Barrier(A=(2, 80), B=(16, 81)))
Barrier.append(Init_Barrier(A=(20, 80), B=(38, 81)))
Barrier.append(Init_Barrier(A=(42, 80), B=(58, 81)))
Barrier.append(Init_Barrier(A=(62, 80), B=(78, 81)))
Barrier.append(Init_Barrier(A=(82, 80), B=(98, 81)))
Barrier.append(Init_Barrier(A=(102, 80), B=(118, 81)))
Barrier.append(Init_Barrier(A=(122, 80), B=(138, 81)))
Barrier.append(Init_Barrier(A=(142, 80), B=(159, 81)))
Barrier.append(Init_Barrier(A=(163, 80), B=(166, 81)))

# 竖框
Barrier.append(Init_Barrier(A=(21, 81), B=(22, 109)))
Barrier.append(Init_Barrier(A=(43, 81), B=(44, 97)))
Barrier.append(Init_Barrier(A=(43, 100), B=(44, 109)))
Barrier.append(Init_Barrier(A=(63, 81), B=(64, 84)))
Barrier.append(Init_Barrier(A=(63, 89), B=(64, 109)))
Barrier.append(Init_Barrier(A=(83, 81), B=(84, 84)))
Barrier.append(Init_Barrier(A=(83, 89), B=(84, 97)))
Barrier.append(Init_Barrier(A=(83, 100), B=(84, 109)))
Barrier.append(Init_Barrier(A=(103, 81), B=(104, 109)))
Barrier.append(Init_Barrier(A=(123, 81), B=(124, 84)))
Barrier.append(Init_Barrier(A=(123, 89), B=(124, 97)))
Barrier.append(Init_Barrier(A=(123, 100), B=(124, 109)))
Barrier.append(Init_Barrier(A=(143, 81), B=(144, 84)))
Barrier.append(Init_Barrier(A=(143, 89), B=(144, 109)))
Barrier.append(Init_Barrier(A=(165, 81), B=(166, 97)))
Barrier.append(Init_Barrier(A=(165, 100), B=(166, 109)))

# 边框 障碍柱子
Barrier.append(Init_Barrier(A=(2, 97), B=(3, 100)))
Barrier.append(Init_Barrier(A=(42, 97), B=(45, 100)))
Barrier.append(Init_Barrier(A=(82, 97), B=(85, 100)))
Barrier.append(Init_Barrier(A=(122, 97), B=(125, 100)))
Barrier.append(Init_Barrier(A=(163, 97), B=(166, 100)))
Barrier.append(Init_Barrier(A=(203, 97), B=(206, 100)))

# 中间部分
Barrier.append(Init_Barrier(A=(0, 56), B=(6, 57)))
Barrier.append(Init_Barrier(A=(0, 64), B=(3, 66)))
Barrier.append(Init_Barrier(A=(0, 66), B=(17, 67)))
Barrier.append(Init_Barrier(A=(9, 56), B=(17, 57)))
Barrier.append(Init_Barrier(A=(16, 56), B=(17, 67)))

Barrier.append(Init_Barrier(A=(25, 56), B=(27, 57)))
Barrier.append(Init_Barrier(A=(25, 66), B=(91, 67)))
Barrier.append(Init_Barrier(A=(25, 56), B=(26, 66)))
Barrier.append(Init_Barrier(A=(30, 56), B=(47, 57)))
Barrier.append(Init_Barrier(A=(42, 64), B=(45, 66)))
Barrier.append(Init_Barrier(A=(45, 56), B=(46, 66)))
Barrier.append(Init_Barrier(A=(50, 56), B=(65, 57)))
Barrier.append(Init_Barrier(A=(63, 56), B=(64, 66)))
Barrier.append(Init_Barrier(A=(68, 56), B=(95, 57)))
Barrier.append(Init_Barrier(A=(82, 56), B=(83, 66)))
Barrier.append(Init_Barrier(A=(82, 64), B=(85, 66)))

Barrier.append(Init_Barrier(A=(95, 45), B=(96, 66)))
Barrier.append(Init_Barrier(A=(95, 45), B=(107, 46)))
Barrier.append(Init_Barrier(A=(94, 66), B=(220, 67)))
Barrier.append(Init_Barrier(A=(111, 45), B=(114, 46)))
Barrier.append(Init_Barrier(A=(112, 45), B=(113, 66)))
Barrier.append(Init_Barrier(A=(122, 64), B=(125, 66)))
Barrier.append(Init_Barrier(A=(118, 45), B=(141, 46)))
Barrier.append(Init_Barrier(A=(129, 45), B=(130, 66)))
Barrier.append(Init_Barrier(A=(145, 45), B=(148, 46)))
Barrier.append(Init_Barrier(A=(146, 45), B=(147, 66)))

Barrier.append(Init_Barrier(A=(152, 45), B=(175, 46)))
Barrier.append(Init_Barrier(A=(163, 45), B=(164, 66)))
Barrier.append(Init_Barrier(A=(163, 64), B=(166, 66)))
Barrier.append(Init_Barrier(A=(179, 45), B=(182, 46)))
Barrier.append(Init_Barrier(A=(180, 45), B=(181, 66)))
Barrier.append(Init_Barrier(A=(186, 45), B=(199, 46)))
Barrier.append(Init_Barrier(A=(197, 45), B=(198, 66)))
Barrier.append(Init_Barrier(A=(203, 64), B=(206, 66)))
Barrier.append(Init_Barrier(A=(203, 45), B=(220, 46)))

# 上面部分
Barrier.append(Init_Barrier(A=(10, 10), B=(30, 11)))
Barrier.append(Init_Barrier(A=(10, 18), B=(43, 20)))
Barrier.append(Init_Barrier(A=(25, 20), B=(27, 22)))
Barrier.append(Init_Barrier(A=(43, 0), B=(45, 35)))
Barrier.append(Init_Barrier(A=(0, 33), B=(17, 35)))
Barrier.append(Init_Barrier(A=(25, 30), B=(27, 35)))
Barrier.append(Init_Barrier(A=(25, 33), B=(45, 35)))
Barrier.append(Init_Barrier(A=(16, 33), B=(17, 39)))
Barrier.append(Init_Barrier(A=(16, 43), B=(17, 48)))
Barrier.append(Init_Barrier(A=(0, 43), B=(7, 44)))
Barrier.append(Init_Barrier(A=(6, 43), B=(7, 48)))
Barrier.append(Init_Barrier(A=(6, 47), B=(17, 48)))
Barrier.append(Init_Barrier(A=(25, 35), B=(26, 48)))
Barrier.append(Init_Barrier(A=(25, 47), B=(27, 48)))
Barrier.append(Init_Barrier(A=(38, 35), B=(39, 48)))
Barrier.append(Init_Barrier(A=(34, 47), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 43), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 35), B=(45, 36)))
# Barrier.append ( Init_Barrier ( A=(29 , 11) , B=(30 , 18) ) )

Barrier.append(Init_Barrier(A=(49, 0), B=(50, 14)))
Barrier.append(Init_Barrier(A=(68, 0), B=(69, 37)))
Barrier.append(Init_Barrier(A=(68, 45), B=(69, 48)))
Barrier.append(Init_Barrier(A=(53, 34), B=(69, 35)))
Barrier.append(Init_Barrier(A=(53, 34), B=(54, 48)))
Barrier.append(Init_Barrier(A=(53, 47), B=(55, 48)))
Barrier.append(Init_Barrier(A=(62, 47), B=(69, 48)))
Barrier.append(Init_Barrier(A=(63, 34), B=(64, 48)))

Barrier.append(Init_Barrier(A=(77, 11), B=(166, 12)))
Barrier.append(Init_Barrier(A=(77, 11), B=(78, 37)))
Barrier.append(Init_Barrier(A=(77, 45), B=(78, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(83, 48)))
Barrier.append(Init_Barrier(A=(77, 34), B=(82, 35)))
Barrier.append(Init_Barrier(A=(77, 47), B=(83, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(90, 36)))
Barrier.append(Init_Barrier(A=(95, 10), B=(96, 36)))
Barrier.append(Init_Barrier(A=(94, 35), B=(107, 36)))
Barrier.append(Init_Barrier(A=(112, 10), B=(113, 36)))
Barrier.append(Init_Barrier(A=(129, 10), B=(130, 36)))
Barrier.append(Init_Barrier(A=(146, 10), B=(147, 36)))
Barrier.append(Init_Barrier(A=(164, 10), B=(166, 36)))
Barrier.append(Init_Barrier(A=(111, 35), B=(114, 36)))
Barrier.append(Init_Barrier(A=(118, 35), B=(141, 36)))
Barrier.append(Init_Barrier(A=(145, 35), B=(148, 36)))
Barrier.append(Init_Barrier(A=(152, 35), B=(166, 36)))

# 柱子
Barrier.append(Init_Barrier(A=(82, 0), B=(85, 3)))
Barrier.append(Init_Barrier(A=(122, 0), B=(125, 3)))
Barrier.append(Init_Barrier(A=(163, 0), B=(166, 3)))
Barrier.append(Init_Barrier(A=(203, 0), B=(206, 3)))
Barrier.append(Init_Barrier(A=(82, 32), B=(85, 35)))
Barrier.append(Init_Barrier(A=(122, 32), B=(125, 35)))
Barrier.append(Init_Barrier(A=(163, 32), B=(166, 35)))
Barrier.append(Init_Barrier(A=(203, 32), B=(206, 35)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(122, 36), B=(135, 45)))

# 箭头 矩形区域
Arrow1 = list()
# 1
Arrow1.append(Init_Arrow1(A=(20, 51), B=(22, 55)))
# 1+
Arrow1.append(Init_Arrow1(A=(17, 30), B=(13, 28)))
# 2
Arrow1.append(Init_Arrow1(A=(45, 51), B=(49, 53)))
# 3
Arrow1.append(Init_Arrow1(A=(45, 72), B=(49, 74)))
# 4
Arrow1.append(Init_Arrow1(A=(127, 72), B=(131, 74)))
# 5
Arrow1.append(Init_Arrow1(A=(183, 90), B=(185, 94)))
# 5+
Arrow1.append(Init_Arrow1(A=(130, 39), B=(134, 41)))
# 6
Arrow1.append(Init_Arrow1(A=(104, 39), B=(108, 41)))
# 7
Arrow1.append(Init_Arrow1(A=(145, 39), B=(149, 41)))
# 8
Arrow1.append(Init_Arrow1(A=(183, 22), B=(185, 26)))
# 9
Arrow1.append(Init_Arrow1(A=(136, 5), B=(140, 7)))
# 10
Arrow1.append(Init_Arrow1(A=(84, 5), B=(88, 7)))
# 10+
Arrow1.append(Init_Arrow1(A=(81, 78), B=(84, 76)))
# 11
Arrow1.append(Init_Arrow1(A=(48, 30), B=(50, 34)))
# 12
Arrow1.append(Init_Arrow1(A=(190, 72), B=(194, 74)))
# 13
Arrow1.append(Init_Arrow1(A=(72, 23), B=(74, 27)))

# 箭头 三角形区域
Arrow2 = list()
# 1
Arrow2.append(Init_Arrow2(A=(21, 49), B=(19, 51), C=(23, 51)))
# 1+
Arrow2.append(Init_Arrow2(A=(11, 29), B=(13, 27), C=(13, 31)))
# 2
Arrow2.append(Init_Arrow2(A=(43, 52), B=(45, 50), C=(45, 54)))
# 3
Arrow2.append(Init_Arrow2(A=(43, 73), B=(45, 71), C=(45, 75)))
# 4
Arrow2.append(Init_Arrow2(A=(133, 73), B=(131, 75), C=(131, 71)))
# 5
Arrow2.append(Init_Arrow2(A=(184, 88), B=(186, 90), C=(182, 90)))
# 5+
Arrow2.append(Init_Arrow2(A=(128, 40), B=(130, 38), C=(130, 42)))
# 6
Arrow2.append(Init_Arrow2(A=(102, 40), B=(104, 38), C=(104, 42)))
# 7
Arrow2.append(Init_Arrow2(A=(151, 40), B=(149, 42), C=(149, 38)))
# 8
Arrow2.append(Init_Arrow2(A=(184, 20), B=(186, 22), C=(182, 22)))
# 9
Arrow2.append(Init_Arrow2(A=(134, 6), B=(136, 4), C=(136, 8)))
# 10
Arrow2.append(Init_Arrow2(A=(90, 6), B=(88, 8), C=(88, 4)))
# 10+
Arrow2.append(Init_Arrow2(A=(79, 77), B=(81, 75), C=(81, 79)))
# 11
Arrow2.append(Init_Arrow2(A=(49, 36), B=(51, 34), C=(47, 34)))
# 12
Arrow2.append(Init_Arrow2(A=(196, 73), B=(194, 75), C=(194, 71)))
# 13
Arrow2.append(Init_Arrow2(A=(73, 21), B=(75, 23), C=(71, 23)))

myMap7 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

# 房间长宽
Length = 220
Width = 110

# 出口
# 点集
Exit = Init_Exit(P1=(220, 70), P2=(220, 80))
Exit.extend(Init_Exit(P1=(100, 0), P2=(108, 0)))
Exit.extend(Init_Exit(P1=(29, 12), P2=(29, 17)))
# 障碍 矩形区域
# 边框

Barrier = list()
Barrier.append(Init_Barrier(A=(0, 0), B=(2, 110)))
Barrier.append(Init_Barrier(A=(219, 0), B=(220, 70)))
Barrier.append(Init_Barrier(A=(219, 80), B=(220, 110)))
Barrier.append(Init_Barrier(A=(0, 0), B=(100, 1)))
Barrier.append(Init_Barrier(A=(108, 0), B=(220, 1)))
Barrier.append(Init_Barrier(A=(0, 109), B=(220, 110)))

# 下面部分
Barrier.append(Init_Barrier(A=(2, 80), B=(16, 81)))
Barrier.append(Init_Barrier(A=(20, 80), B=(38, 81)))
Barrier.append(Init_Barrier(A=(42, 80), B=(58, 81)))
Barrier.append(Init_Barrier(A=(62, 80), B=(78, 81)))
Barrier.append(Init_Barrier(A=(82, 80), B=(98, 81)))
Barrier.append(Init_Barrier(A=(102, 80), B=(118, 81)))
Barrier.append(Init_Barrier(A=(122, 80), B=(138, 81)))
Barrier.append(Init_Barrier(A=(142, 80), B=(159, 81)))
Barrier.append(Init_Barrier(A=(163, 80), B=(166, 81)))

# 竖框
Barrier.append(Init_Barrier(A=(21, 81), B=(22, 109)))
Barrier.append(Init_Barrier(A=(43, 81), B=(44, 97)))
Barrier.append(Init_Barrier(A=(43, 100), B=(44, 109)))
Barrier.append(Init_Barrier(A=(63, 81), B=(64, 84)))
Barrier.append(Init_Barrier(A=(63, 89), B=(64, 109)))
Barrier.append(Init_Barrier(A=(83, 81), B=(84, 84)))
Barrier.append(Init_Barrier(A=(83, 89), B=(84, 97)))
Barrier.append(Init_Barrier(A=(83, 100), B=(84, 109)))
Barrier.append(Init_Barrier(A=(103, 81), B=(104, 109)))
Barrier.append(Init_Barrier(A=(123, 81), B=(124, 84)))
Barrier.append(Init_Barrier(A=(123, 89), B=(124, 97)))
Barrier.append(Init_Barrier(A=(123, 100), B=(124, 109)))
Barrier.append(Init_Barrier(A=(143, 81), B=(144, 84)))
Barrier.append(Init_Barrier(A=(143, 89), B=(144, 109)))
Barrier.append(Init_Barrier(A=(165, 81), B=(166, 97)))
Barrier.append(Init_Barrier(A=(165, 100), B=(166, 109)))

# 边框 障碍柱子
Barrier.append(Init_Barrier(A=(2, 97), B=(3, 100)))
Barrier.append(Init_Barrier(A=(42, 97), B=(45, 100)))
Barrier.append(Init_Barrier(A=(82, 97), B=(85, 100)))
Barrier.append(Init_Barrier(A=(122, 97), B=(125, 100)))
Barrier.append(Init_Barrier(A=(163, 97), B=(166, 100)))
Barrier.append(Init_Barrier(A=(203, 97), B=(206, 100)))

# 中间部分
Barrier.append(Init_Barrier(A=(0, 56), B=(6, 57)))
Barrier.append(Init_Barrier(A=(0, 64), B=(3, 66)))
Barrier.append(Init_Barrier(A=(0, 66), B=(17, 67)))
Barrier.append(Init_Barrier(A=(9, 56), B=(17, 57)))
Barrier.append(Init_Barrier(A=(16, 56), B=(17, 67)))

Barrier.append(Init_Barrier(A=(25, 56), B=(27, 57)))
Barrier.append(Init_Barrier(A=(25, 66), B=(91, 67)))
Barrier.append(Init_Barrier(A=(25, 56), B=(26, 66)))
Barrier.append(Init_Barrier(A=(30, 56), B=(47, 57)))
Barrier.append(Init_Barrier(A=(42, 64), B=(45, 66)))
Barrier.append(Init_Barrier(A=(45, 56), B=(46, 66)))
Barrier.append(Init_Barrier(A=(50, 56), B=(65, 57)))
Barrier.append(Init_Barrier(A=(63, 56), B=(64, 66)))
Barrier.append(Init_Barrier(A=(68, 56), B=(95, 57)))
Barrier.append(Init_Barrier(A=(82, 56), B=(83, 66)))
Barrier.append(Init_Barrier(A=(82, 64), B=(85, 66)))

Barrier.append(Init_Barrier(A=(95, 45), B=(96, 66)))
Barrier.append(Init_Barrier(A=(95, 45), B=(107, 46)))
Barrier.append(Init_Barrier(A=(94, 66), B=(220, 67)))
Barrier.append(Init_Barrier(A=(111, 45), B=(114, 46)))
Barrier.append(Init_Barrier(A=(112, 45), B=(113, 66)))
Barrier.append(Init_Barrier(A=(122, 64), B=(125, 66)))
Barrier.append(Init_Barrier(A=(118, 45), B=(141, 46)))
Barrier.append(Init_Barrier(A=(129, 45), B=(130, 66)))
Barrier.append(Init_Barrier(A=(145, 45), B=(148, 46)))
Barrier.append(Init_Barrier(A=(146, 45), B=(147, 66)))

Barrier.append(Init_Barrier(A=(152, 45), B=(175, 46)))
Barrier.append(Init_Barrier(A=(163, 45), B=(164, 66)))
Barrier.append(Init_Barrier(A=(163, 64), B=(166, 66)))
Barrier.append(Init_Barrier(A=(179, 45), B=(182, 46)))
Barrier.append(Init_Barrier(A=(180, 45), B=(181, 66)))
Barrier.append(Init_Barrier(A=(186, 45), B=(199, 46)))
Barrier.append(Init_Barrier(A=(197, 45), B=(198, 66)))
Barrier.append(Init_Barrier(A=(203, 64), B=(206, 66)))
Barrier.append(Init_Barrier(A=(203, 45), B=(220, 46)))

# 上面部分
Barrier.append(Init_Barrier(A=(10, 10), B=(30, 11)))
Barrier.append(Init_Barrier(A=(10, 18), B=(43, 20)))
Barrier.append(Init_Barrier(A=(25, 20), B=(27, 22)))
Barrier.append(Init_Barrier(A=(43, 0), B=(45, 35)))
Barrier.append(Init_Barrier(A=(0, 33), B=(17, 35)))
Barrier.append(Init_Barrier(A=(25, 30), B=(27, 35)))
Barrier.append(Init_Barrier(A=(25, 33), B=(45, 35)))
Barrier.append(Init_Barrier(A=(16, 33), B=(17, 39)))
Barrier.append(Init_Barrier(A=(16, 43), B=(17, 48)))
Barrier.append(Init_Barrier(A=(0, 43), B=(7, 44)))
Barrier.append(Init_Barrier(A=(6, 43), B=(7, 48)))
Barrier.append(Init_Barrier(A=(6, 47), B=(17, 48)))
Barrier.append(Init_Barrier(A=(25, 35), B=(26, 48)))
Barrier.append(Init_Barrier(A=(25, 47), B=(27, 48)))
Barrier.append(Init_Barrier(A=(38, 35), B=(39, 48)))
Barrier.append(Init_Barrier(A=(34, 47), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 43), B=(45, 48)))
Barrier.append(Init_Barrier(A=(44, 35), B=(45, 36)))
# Barrier.append ( Init_Barrier ( A=(29 , 11) , B=(30 , 18) ) )

Barrier.append(Init_Barrier(A=(49, 0), B=(50, 14)))
Barrier.append(Init_Barrier(A=(68, 0), B=(69, 37)))
Barrier.append(Init_Barrier(A=(68, 45), B=(69, 48)))
Barrier.append(Init_Barrier(A=(53, 34), B=(69, 35)))
Barrier.append(Init_Barrier(A=(53, 34), B=(54, 48)))
Barrier.append(Init_Barrier(A=(53, 47), B=(55, 48)))
Barrier.append(Init_Barrier(A=(62, 47), B=(69, 48)))
Barrier.append(Init_Barrier(A=(63, 34), B=(64, 48)))

Barrier.append(Init_Barrier(A=(77, 11), B=(166, 12)))
Barrier.append(Init_Barrier(A=(77, 11), B=(78, 37)))
Barrier.append(Init_Barrier(A=(77, 45), B=(78, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(83, 48)))
Barrier.append(Init_Barrier(A=(77, 34), B=(82, 35)))
Barrier.append(Init_Barrier(A=(77, 47), B=(83, 48)))
Barrier.append(Init_Barrier(A=(82, 35), B=(90, 36)))
Barrier.append(Init_Barrier(A=(95, 10), B=(96, 36)))
Barrier.append(Init_Barrier(A=(94, 35), B=(107, 36)))
Barrier.append(Init_Barrier(A=(112, 10), B=(113, 36)))
Barrier.append(Init_Barrier(A=(129, 10), B=(130, 36)))
Barrier.append(Init_Barrier(A=(146, 10), B=(147, 36)))
Barrier.append(Init_Barrier(A=(164, 10), B=(166, 36)))
Barrier.append(Init_Barrier(A=(111, 35), B=(114, 36)))
Barrier.append(Init_Barrier(A=(118, 35), B=(141, 36)))
Barrier.append(Init_Barrier(A=(145, 35), B=(148, 36)))
Barrier.append(Init_Barrier(A=(152, 35), B=(166, 36)))

# 柱子
Barrier.append(Init_Barrier(A=(82, 0), B=(85, 3)))
Barrier.append(Init_Barrier(A=(122, 0), B=(125, 3)))
Barrier.append(Init_Barrier(A=(163, 0), B=(166, 3)))
Barrier.append(Init_Barrier(A=(203, 0), B=(206, 3)))
Barrier.append(Init_Barrier(A=(82, 32), B=(85, 35)))
Barrier.append(Init_Barrier(A=(122, 32), B=(125, 35)))
Barrier.append(Init_Barrier(A=(163, 32), B=(166, 35)))
Barrier.append(Init_Barrier(A=(203, 32), B=(206, 35)))

# 火灾 矩形区域
Fire = list()
Fire.append(Init_Fire(A=(83, 67), B=(90, 80)))

# 箭头 矩形区域
Arrow1 = list()
# 1
Arrow1.append(Init_Arrow1(A=(20, 51), B=(22, 55)))
# 1+
Arrow1.append(Init_Arrow1(A=(13, 28), B=(17, 30)))
# 2
Arrow1.append(Init_Arrow1(A=(45, 51), B=(49, 53)))
# 3
Arrow1.append(Init_Arrow1(A=(45, 72), B=(49, 74)))
# 4
Arrow1.append(Init_Arrow1(A=(127, 72), B=(131, 74)))
# 5
Arrow1.append(Init_Arrow1(A=(183, 90), B=(185, 94)))
# 5+
Arrow1.append(Init_Arrow1(A=(130, 39), B=(134, 41)))
# 6
Arrow1.append(Init_Arrow1(A=(104, 39), B=(108, 41)))
# 7
Arrow1.append(Init_Arrow1(A=(145, 39), B=(149, 41)))
# 8
Arrow1.append(Init_Arrow1(A=(183, 22), B=(185, 26)))
# 9
Arrow1.append(Init_Arrow1(A=(136, 5), B=(140, 7)))
# 10
Arrow1.append(Init_Arrow1(A=(84, 5), B=(88, 7)))

# 10+
Arrow1.append(Init_Arrow1(A=(81, 78), B=(84, 76)))
# 11
Arrow1.append(Init_Arrow1(A=(48, 30), B=(50, 34)))
# 12
Arrow1.append(Init_Arrow1(A=(190, 72), B=(194, 74)))
# 13
Arrow1.append(Init_Arrow1(A=(72, 23), B=(74, 27)))

# 箭头 三角形区域
Arrow2 = list()
# 1
Arrow2.append(Init_Arrow2(A=(21, 49), B=(19, 51), C=(23, 51)))
# 1+
Arrow2.append(Init_Arrow2(A=(11, 29), B=(13, 27), C=(13, 31)))
# 2
Arrow2.append(Init_Arrow2(A=(43, 52), B=(45, 50), C=(45, 54)))
# 3
Arrow2.append(Init_Arrow2(A=(43, 73), B=(45, 71), C=(45, 75)))
# 4
Arrow2.append(Init_Arrow2(A=(133, 73), B=(131, 75), C=(131, 71)))
# 5
Arrow2.append(Init_Arrow2(A=(184, 88), B=(186, 90), C=(182, 90)))
# 5+
Arrow2.append(Init_Arrow2(A=(128, 40), B=(130, 38), C=(130, 42)))
# 6
Arrow2.append(Init_Arrow2(A=(102, 40), B=(104, 38), C=(104, 42)))
# 7
Arrow2.append(Init_Arrow2(A=(151, 40), B=(149, 42), C=(149, 38)))
# 8
Arrow2.append(Init_Arrow2(A=(184, 20), B=(186, 22), C=(182, 22)))
# 9
Arrow2.append(Init_Arrow2(A=(134, 6), B=(136, 4), C=(136, 8)))
# 10
Arrow2.append(Init_Arrow2(A=(90, 6), B=(88, 8), C=(88, 4)))
# 10+
Arrow2.append(Init_Arrow2(A=(79, 77), B=(81, 75), C=(81, 79)))
# 11
Arrow2.append(Init_Arrow2(A=(49, 36), B=(51, 34), C=(47, 34)))
# 12
Arrow2.append(Init_Arrow2(A=(196, 73), B=(194, 75), C=(194, 71)))
# 13
Arrow2.append(Init_Arrow2(A=(73, 21), B=(75, 23), C=(71, 23)))

myMap8 = Map(L=Length, W=Width, E=Exit, B=Barrier, F=Fire, A1=Arrow1, A2=Arrow2)

myMap.append(myMap1)
myMap.append(myMap2)
myMap.append(myMap3)
myMap.append(myMap4)
myMap.append(myMap5)
myMap.append(myMap6)
myMap.append(myMap7)
myMap.append(myMap8)


class GUI:
    # GUI
    # 图像比例
    Pic_Ratio = 5

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("疏散模拟")
        self.root.geometry("1100x700")
        self.root.resizable(width=False, height=False)

        width = myMap[i].Length * GUI.Pic_Ratio
        height = myMap[i].Width * GUI.Pic_Ratio
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.pack()

        self.label_time = tk.Label(self.root, text="Time = 0.00s", font='Arial -27 bold')
        self.label_evac = tk.Label(self.root, text="Evacution People: 0", font='Arial -27 bold')
        self.label_scale = tk.Label(self.root, text="比例尺：0", font='Arial -27 bold')
        self.label_time.pack()
        self.label_evac.pack()
        self.label_scale.pack()

        self.setBarrier()
        self.setExit()
        self.setFire()
        self.setArrow()

    # 障碍
    def setBarrier(self):
        for (A, B) in myMap[i].Barrier:
            x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
            [x1, y1, x2, y2] = map(lambda x: x * GUI.Pic_Ratio, [x1, y1, x2, y2])
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

    # 出口
    def setExit(self):
        for (x, y) in myMap[i].Exit:
            sx, sy = x - 1, y - 1
            ex, ey = x + 1, y + 1
            [sx, sy, ex, ey] = map(lambda x: x * GUI.Pic_Ratio, [sx, sy, ex, ey])
            self.canvas.create_rectangle(sx, sy, ex, ey, fill="green", outline="green")

    # 火灾
    def setFire(self):
        for (A, B) in myMap[i].Fire:
            x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
            [x1, y1, x2, y2] = map(lambda x: x * GUI.Pic_Ratio, [x1, y1, x2, y2])
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline="yellow")

    def setArrow(self):
        for (A, B) in myMap[i].Arrow1:
            x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
            [x1, y1, x2, y2] = map(lambda x: x * GUI.Pic_Ratio, [x1, y1, x2, y2])

            # Map.space[x1][y1] - Map.space[x2][y2]
            if myMap[i].space[A[0]][A[1]] - myMap[i].space[B[0]][B[1]] > 0:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", outline="red")  # 两个点变成矩形
            else:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="green")
        for (A, B, C) in myMap[i].Arrow2:
            x1, y1, x2, y2, x3, y3 = A[0], A[1], B[0], B[1], C[0], C[1]
            [x1, y1, x2, y2, x3, y3] = map(lambda x: x * GUI.Pic_Ratio, [x1, y1, x2, y2, x3, y3])
            if myMap[i].space[A[0]][A[1]] - myMap[i].space[B[0]][B[1]] >0 or myMap[i].space[A[0]][A[1]] -  myMap[i].space[C[0]][C[1]] >0:
                self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, fill="red", outline="red")
            else:
                self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, fill="green", outline="green")

    def Update_People(self, People_List):
        for p in People_List:
            # print(p.id)
            self.canvas.delete(p.name())

        self.Show_People(People_List)
    def Show_People(self, People_List):
        for p in People_List:
            if p.savety:
                continue
            ox, oy = p.pos[0], p.pos[1]
            print(p.pos[0])
            print(p.pos[1])
            x1, y1 = ox - 1.2, oy - 1.2
            x2, y2 = ox + 1.2, oy + 1.2
            [x1, y1, x2, y2] = map(lambda x: x * GUI.Pic_Ratio, [x1, y1, x2, y2])
            self.canvas.create_oval(x1, y1, x2, y2, fill="red", outline="red", tag=p.name())


def Cellular_Automata():
    UI = GUI()
    # Total_People=500
    # Total_People = 200
    P = People(a, myMap[i])

    UI.Show_People(P.list)


    Time_Start = time.time()  # 计时
    Eva_Number = 0  # 逃离人数
    scale = 10000;
    while Eva_Number < a:
        Eva_Number = P.run()  # 检测是否有人逃出去

        UI.Update_People(P.list1)  # 刷新
        UI.Update_People(P.list2)
        UI.Update_People(P.list3)
        time.sleep(random.uniform(0.15, 0.25))  # 控制刷新率

        UI.canvas.update()
        UI.root.update()

        Time_Pass = time.time() - Time_Start
        UI.label_time['text'] = "Time = " + "%.2f" % Time_Pass + "s"
        UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)
    # print("%.2fs" % (Time_Pass) + " 已疏散人数:" +str(Eva_Number))

    Time_Pass = time.time() - Time_Start
    UI.label_time['text'] = "Time = " + "%.2f" % Time_Pass + "s"
    UI.label_evac['text'] = "Evacution People: " + str(Eva_Number)
    UI.label_scale['text'] = "比例尺" + str(scale)

    # 热力图
    #    sns.heatmap ( P.thmap.T , cmap='Reds' )
    #    plt.axis ( 'equal' )
    #    plt.show ()

    UI.root.mainloop()


def choice1():
    global a
    a = 500


def choice2():
    global a
    a = 1000


def choice3():
    global j
    j = 0.9


def choice4():
    global j
    j = 1.1


def choice5():
    global i
    if j < 1:
        i = 1
    else:
        i = 5


def choice6():
    global i
    if j < 1:
        i = 2
    else:
        i = 6


def choice7():
    global i
    if j < 1:
        i = 3
    else:
        i = 7


def choice8():
    global i
    if j < 1:
        i = 0
    else:
        i = 4


def window():
    win = Tk()  # 构造窗体
    win.title("火灾模拟仿真交互式界面")
    Label(win, text='逃生者人数：', width=10) \
        .grid(row=2, column=0, sticky=W, padx=10, pady=5)
    Button(win, text='500人', width=10, command=choice1) \
        .grid(row=2, column=1, sticky=E, padx=10, pady=5)
    Button(win, text='1000人', width=10, command=choice2) \
        .grid(row=2, column=2, sticky=E, padx=10, pady=5)
    Label(win, text='选择地图：', width=10) \
        .grid(row=3, column=0, sticky=W, padx=10, pady=5)
    Button(win, text='地图1', width=10, command=choice3) \
        .grid(row=3, column=1, sticky=E, padx=10, pady=5)
    Button(win, text='地图2', width=10, command=choice4) \
        .grid(row=3, column=2, sticky=W, padx=10, pady=5)
    Label(win, text='火情位置：', width=10) \
        .grid(row=4, column=0, sticky=W, padx=10, pady=5)
    Button(win, text='位置1', width=10, command=choice5) \
        .grid(row=4, column=1, sticky=E, padx=10, pady=5)
    Button(win, text='位置2', width=10, command=choice6) \
        .grid(row=4, column=2, sticky=W, padx=10, pady=5)
    Button(win, text='位置3', width=10, command=choice7) \
        .grid(row=4, column=3, sticky=W, padx=10, pady=5)
    Button(win, text='无火情', width=10, command=choice8) \
        .grid(row=4, column=4, sticky=W, padx=10, pady=5)
    Button(win, text='开始仿真', width=10, command=Cellular_Automata) \
        .grid(row=5, column=0, sticky=E, padx=10, pady=5)
    Button(win, text='退出仿真', width=10, command=win.quit) \
        .grid(row=5, column=4, sticky=W, padx=10, pady=5)
    v1 = StringVar()
    e1 = Entry(win, textvariable=v1)
    win.mainloop()  # 进入消息循环机制


window()
