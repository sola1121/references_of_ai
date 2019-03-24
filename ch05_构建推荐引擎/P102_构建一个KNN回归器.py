import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


# 生成样本数据
amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - .5 * amplitude

# 计算目标并添加噪声
y = np.sinc(X).ravel()
y += .2 * (.5 - np.random.rand(y.size))

# 画出输入的图形
plt.figure(figsize=(6, 8))
plt.subplot(211)
plt.subplots_adjust(hspace=.5)
plt.scatter(X, y, s=40, c="k", facecolors="none")
plt.title("Input data")

# 用输入数据10倍的密度创建一维网络
x_values = np.linspace(-0.5*amplitude, 0.5*amplitude, 10*num_points)[:, np.newaxis]

# 定义最近邻的个数
num_neighbors = 8

# 定义并训练回归器
knn_regressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
knn_regressor.fit(X, y)
y_values = knn_regressor.predict(x_values)


# 作图
plt.subplot(212)
plt.scatter(X, y, s=40, edgecolors="black", facecolors="none", label="input data")
plt.plot(x_values, y_values, c="blue", linestyle="--", label="predicted values")
plt.xlim(min(X)-1, max(X)+1)
plt.ylim(min(y)-1, max(y)+1)
plt.legend()
plt.title("K Nearset Neighbors Regressor")

plt.show()

