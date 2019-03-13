import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


file_dir = "./dat/data_multivar.txt"
X = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(","))

X = np.array(X).astype(np.float)

# 设置带宽参数bandwidth
bandwidth = estimate_bandwidth(X, quantile=.1, n_samples=len(X))

# 使用MeanShift计算聚类
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# 训练模型
meanshift_estimator.fit(X)

# 提取标记, 中心点
labels = meanshift_estimator.labels_
centroids = meanshift_estimator.cluster_centers_
print("The number of clusters in input data =", len(centroids))

# 作画数据点和聚类中心
markers = ".*xv"
plt.figure()
for i, marker in zip(range(len(centroids)), markers):
    # 画出属于某个集群中心的数据点
    plt.scatter(X[labels==i, 0], X[labels==i, 1], c="k", marker=marker)
    # 画出集群中心
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker="o", markerfacecolor="k", markeredgecolor="k", markersize=10)
plt.show()

