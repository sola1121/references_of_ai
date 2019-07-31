import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering


file_dir = "../ch04_无监督学习_聚类/dat/data_multivar.txt"
X = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(","))
    
X = np.array(X).astype(np.float)

# 不考虑数据具有连接性
agg1_estimator = AgglomerativeClustering(n_clusters=4)
agg1_estimator.fit(X)

agg1_labels = agg1_estimator.labels_
agg1_num_clusters = len(np.unique(agg1_labels))
agg1_leaves_num = agg1_estimator.n_leaves_   # 分类器叶数

# 考虑数据的连接性
# 使用临近算法计算数据点的权重
connectivity = kneighbors_graph(X, n_neighbors=5)
agg2_estimator = AgglomerativeClustering(n_clusters=4, connectivity=connectivity)
agg2_estimator.fit(X)

agg2_labels = agg2_estimator.labels_
agg2_num_clusters = len(np.unique(agg2_labels))
agg2_leaves_num = agg2_estimator.n_leaves_

# 作图
markers = "dv+p*s"

plt.figure(figsize=[5, 6])

plt.subplot(211)
plt.subplots_adjust(hspace=.8)
for i, marker in zip(range(agg1_num_clusters), markers[:agg1_num_clusters]):
    # 聚合的数据点
    plt.scatter(X[agg1_labels==i, 0], X[agg1_labels==i, 1], marker=marker)
plt.title("No connectiveity, leaves %s, cluster number %s" % (agg1_leaves_num, agg1_num_clusters))

plt.subplot(212)
for i, marker in zip(range(agg2_num_clusters), markers[:agg2_num_clusters]):
    # 聚合的数据点
    plt.scatter(X[agg2_labels==i, 0], X[agg2_labels==i, 1], marker=marker)
plt.title("K-Neighbors connectivity, leaves %s, cluster number %s" % (agg2_leaves_num, agg2_num_clusters))

plt.show()

