# 在监督学习中, 可以使用预测值和真实值进行比较来计算模型的准确性, 但是在无监督学习中, 数据没有标记.
# 在无监督学习中通过观察集群分离的离散程度, 来确定聚合的效果. 观察集群是不是被分离得很合理, 一个集群中的所有数据点是不是足够紧密.
# 轮廓系数(Solhouette Coefficient) 得分指标
#            得分 = (x-y)/max(x, y)
# x表示在同一个集群中某个数据点与其他数据点的平均距离(与集群内点距离), y表示某个数据点与最近的另一个集群的所有点的平均距离(与集群外点距)

import numpy as np
from sklearn.metrics import silhouette_score

# silhouette_score(X, labels, metric = 'euclidean', sample_size = None, random_state = None, **kwds)
# 定义不同n_clusters的模型, 如: KMeans(n_clusters=1), KMeans(n_clusters=3), KMeans(n_clusters=4)来计算分数的多少
# socre = silhouette_score(X, labels=k_means.labels_, sample_size=len(X))

