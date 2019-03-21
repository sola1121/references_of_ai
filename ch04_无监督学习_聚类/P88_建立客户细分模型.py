import csv

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn import cluster, covariance, manifold
from sklearn.cluster import MeanShift, estimate_bandwidth


file_dir = "./dat/wholesale.csv"
X = list()
names = None   # 记录数据名称

with open(file_dir, 'r') as file:
    csv_reader = csv.reader(file.readlines(), delimiter=",")
    temp_data = list(csv_reader)
    names = temp_data[0]
    X.append(temp_data[1:])

X = np.array(X).astype(np.int32)[0][:, 2:]

# 估计带宽参数
bandwidth = estimate_bandwidth(X, quantile=.8, n_samples=len(X))
# 用MeanShift函数计算聚类
meanshift_estimateor = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimateor.fit(X)

# 中心质点
centroids = meanshift_estimateor.cluster_centers_
# 标签
labels = meanshift_estimateor.labels_

num_clusters = len(np.unique(labels))
print("共有分类:", num_clusters)
print('\t'.join([name[:3] for name in names[2:]]))
for centroid in centroids:
    print('\t'.join([str(int(x)) for x in centroid]))


# 数据可视化
centroids_milk_groceries = centroids[:, 1:3]   # 提取milk, grocery列的数据

# 用centroids_milk_grocerise中的坐标画出中西点
plt.figure()
plt.scatter(centroids_milk_groceries[:, 0], centroids_milk_groceries[:, 1], s=100, edgecolors='k', facecolors="none")
offset = .2
plt.xlim(centroids_milk_groceries[:, 0].min() - offset * centroids_milk_groceries[:, 0].ptp(), 
         centroids_milk_groceries[:, 0].max() + offset * centroids_milk_groceries[:, 0].ptp())
plt.ylim(centroids_milk_groceries[:, 1].min() - offset * centroids_milk_groceries[:, 1].ptp(),
         centroids_milk_groceries[:, 1].max() + offset * centroids_milk_groceries[:, 1].ptp())
plt.title("Centroids of clusters for milk and groceries")
plt.show()

