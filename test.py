import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[11], [14]])

# 方法1
c = np.r_[a, b]  # 沿着矩阵行拼接
print('c=', c)
d = np.c_[a, b]  # 沿着矩阵列拼接
print('d=', d)

# 方法2
# e = np.vstack((a, b))  # 沿着矩阵行拼接
# print('e=', e)
# f = np.hstack((a, b))  # 沿着矩阵列拼接
# print('f=', f)

