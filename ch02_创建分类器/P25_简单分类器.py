# 如果输入点(a,b)的a大于或等于b, 那么它属于类型class_0, 反之属于class_1.

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

X = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]
])

y = [0, 1, 1, 0, 0, 1, 1, 0]   # 为X数据集分配的标记

# 按照y分的两类标记, 将X数据集分为两类
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color="black", marker="s")
plt.scatter(class_1[:,0], class_1[:,1], color="black", marker="x")

line_x = range(10)
line_y = line_x

plt.plot(line_x, line_y, color="black", linewidth=2)
plt.show()
