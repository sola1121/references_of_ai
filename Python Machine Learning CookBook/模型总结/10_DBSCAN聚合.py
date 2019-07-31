import itertools

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


file_dir = "../ch04_无监督学习_聚类/dat/data_perf.txt"
data = list()   # 数据集
scores = list()   # 对应epsilon的silhouette score得分集
best_labels = None   # 用于记录最好的分类结果
best_model = None   # 用于记录最好的模型

with open(file_dir, 'r') as file:
    for line in file.readlines():
        data.append(line[:-1].split(","))

data = np.array(data).astype(np.float)

# eps参数取值范围
eps_grid = np.linspace(.3, 1.2, num=10)

# 通过不同的eps参数来获取轮廓系数最高的eps配置
for epsilon in eps_grid:
    # 训练DBSCAN模型
    dbscan_model = DBSCAN(eps=epsilon, min_samples=5, n_jobs=2)
    dbscan_model.fit(data)
    # 获取labels_分类点
    labels = dbscan_model.labels_
    # 打印得分
    score = silhouette_score(X=data, labels=labels, sample_size=len(data))
    scores.append((epsilon, score))
    print("epsilon取值 %.2f 时, 轮廓系数 %.4f" % (epsilon, score))
    # 记录最好的分类和模型
    if score == max([x[1] for x in scores]):
        best_labels = labels
        best_model = dbscan_model

unique_labels = np.unique(best_labels)
print("\n最好的分类种类", str(unique_labels)[1:-1], ", 共", len(unique_labels)-1, "类")

# 提取核心样本. 从训练模型中提取核心样本的数据点索引
mask_core = np.zeros(best_labels.shape, dtype=np.bool)   # 生成和labels_形状相同的全为False的矩阵
mask_core[best_model.core_sample_indices_] = True   # 用于判断聚合的点将会为True, 没有使用的即为噪声点为False

# 作图
markers = itertools.cycle("+xdsp^*<")
plt.figure()
for cur_label, marker in zip(unique_labels, markers):
    # 对-1分类点, 即噪声点进行处理
    if cur_label == -1:
        marker = "."
    # 对分类点进行作图
    cur_mask = (cur_label == best_labels)
    cur_data = data[cur_mask & mask_core]
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker)

plt.show()

