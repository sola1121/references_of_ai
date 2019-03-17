import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth, MeanShift

file_dir = "../ch04_无监督学习_聚类/dat/data_multivar.txt"

X = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(","))

X = np.array(X).astype(np.float)

# 指定输入参数创建一个均值漂移模型
# 设置带宽参数
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X),n_jobs=2)
# 用MeanShift计算聚类
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# 训练模型
meanshift_estimator.fit(X)

# 提取标记
labels = meanshift_estimator.labels_
# 提取模型中的中心点
centroids = meanshift_estimator.cluster_centers_

# 作图
markers = ".*vx"
plt.figure()

for i, marker in zip(range(len(centroids)), markers):
    # 作画每个集群的中心点
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='k')

    # 画出集群中心
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker="o", markerfacecolor="k", markeredgecolor="k", markersize=10)

plt.title("Clusters and their centroids")
plt.show()

