import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# 输入数据
X = np.array([
    [1, 1],
    [1, 3],
    [2, 2],
    [2.5, 5],
    [3, 1],
    [4, 2],
    [2, 3.5],
    [3, 3],
    [3.5, 4]
])

# 寻找最近邻的数量
num_neighbors = 3

# 输入数据点
input_point = [2.6, 1.7]

# 画出数据点
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c="black")


# 建立最近邻模型
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm="ball_tree")
knn.fit(X)

# 计算输入点与输入数据的所有点的距离
distances, indices = knn.kneighbors([input_point])

# 打印出k个最近邻点
print("\nk nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank+1) + " --",  round(distances[0][rank], 3), "-->", X[index])


# 画出最近邻点
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:,1], marker="x", s=150)
plt.scatter(input_point[0], input_point[1], marker="+", color="red")
plt.show()

