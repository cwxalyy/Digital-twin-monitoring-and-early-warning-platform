from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.set_xlim(0, 20) # X轴，横向向右方向
ax.set_ylim(20, 0) # Y轴,左向与X,Z轴互为垂直
ax.set_zlim(0, 20) # 竖向为Z轴
z = np.linspace(0, 4*np.pi, 500)
x = 10*np.sin(z)
y = 10*np.cos(z)
ax.plot3D(x, y, z, 'black') # 绘制黑色空间曲线
# ----------------------------------------------------------
z1 = np.linspace(0, 4*np.pi, 500)
x1 = 5*np.sin(z1)
y1 = 5*np.cos(z1)
ax.plot3D(x1,y1,z1,'g--')   #绘制绿色空间虚曲线
#------------------------------------------------------------
ax.plot3D([0,18,0],[5,18,10],[0,5,0],'om-')  #绘制带o折线
plt.show()