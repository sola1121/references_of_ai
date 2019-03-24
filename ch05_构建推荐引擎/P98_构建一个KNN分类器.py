import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


file_dir = "./dat/data_nn_classifier.txt"
X, y = list(), list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(",")[:-1])
        y.append(line[:-1].split(",")[-1])

X, y = np.array(X, dtype=np.float), np.array(y, dtype=np.int)
test_datapoint = [4.5, 3.6]


### KNN分类模型 ###
# 设置最近邻的点的个数
num_numbers = 10

# 创建KNN分类器模型并训练
classifier = KNeighborsClassifier(n_neighbors=num_numbers, weights="distance")
classifier.fit(X, y)


### 原始点 ###
# 画出输入的数据
plt.figure(figsize=(6, 6))

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=.5)
markers = "^+dshp<>"
mappers = np.array([markers[i] for i in y])
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mappers[i], c="black")
plt.scatter(test_datapoint[0], test_datapoint[1], marker=".", c="red", s=150)
plt.title("Input datapoints")


### 作画边界 ###
# 建立网格来画出边界
x_min, x_max = min(X[:, 0])-1, max(X[:, 0])+1
y_min, y_max = min(X[:, 1])-1, max(X[:, 1])+1

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))

# 计算网格中所有点的预测输出
mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)

# 作画
plt.subplot(2,1,2)
plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.get_cmap("Pastel2"))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


### 提取模型, 使用虚拟点进行计算 ###
# 提取KNN分类结果
dist, indices = classifier.kneighbors([test_datapoint])

for point in zip(dist[0], X[indices][0]):
    print("相距距离", round(point[0], 4), ", 点", point[1])

plt.subplot(2,1,1)
for i in indices:
    plt.scatter(X[i, 0], X[i, 1], marker="x", s=120, c="orange")
plt.title("k nearest neighbors")

# plt.show()

