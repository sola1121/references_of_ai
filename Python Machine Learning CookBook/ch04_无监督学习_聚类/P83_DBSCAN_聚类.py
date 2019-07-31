from itertools import cycle

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


file_dir = "./dat/data_perf.txt"
X = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(","))

X = np.array(X).astype(np.float)

# 寻找最佳集群数量参数, 声明一些预先变量
eps_grid = np.linspace(.3, 1.2, num=10)   # 用于寻找最优epsilon参数
silhouette_scores = list()
eps_best = eps_grid[0]
silhouette_score_max = -1
model_best = None
labels_best = None

# 搜索参数空间
for eps in eps_grid:
    # 训练DBSCAN模型
    model = DBSCAN(eps=eps, min_samples=5, n_jobs=2)
    model.fit(X)

    # 提取标记
    labels = model.labels_

    # 提取性能指标
    score = round(silhouette_score(X, labels=labels), 4)
    silhouette_scores.append(score)
    print("Epsilon:", round(eps, 1), " --> silhouette score:", score)

    # 保存指标的最佳得分和最佳的epsilon参数
    if score > silhouette_score_max:
        silhouette_score_max = score
        eps_best = round(eps, 1)
        model_best = model
        labels_best = labels

# 作图分数条形图
plt.figure(figsize=[6, 7])
plt.subplot(211)
plt.subplots_adjust(hspace=.8)
plt.bar(eps_grid, silhouette_scores, width=.05, color='k', align="center")
plt.xticks(ticks=eps_grid)
plt.title("Silhouette score vs epsilon")
# 打印最优参数
print("\nBest epsilon =", eps_best)


# 检查标记中没有分配集群的数据点
offset = 0
if -1 in labels_best:
    offset = 1

# 数据中集群的数量
num_clusters = len(set(labels_best)) - offset
print("\nEstimated number of clusters =", num_clusters)

# 提取核心样本. 从训练模型中提取核心样本的数据点索引
mask_core = np.zeros(labels.shape, dtype=np.bool)   # 生成和labels_形状相同的全为False的矩阵
mask_core[model_best.core_sample_indices_] = True   # 用于判断聚合的点将会为True, 没有使用的即为噪声点为False
print(mask_core, "\n")

# 作画集群结果
labels_unique = set(labels_best)
markers = cycle("o^sdp*")
plt.subplot(212)
for cur_label, marker, in zip(labels_unique, markers):
    # 用黑点表示为分批恩的数据点
    if cur_label == -1:
        marker = "."

    # 为当前标记添加符号
    cur_mask = (labels_best == cur_label)
    print(cur_label, " ------- ", cur_mask)
    cur_data = X[cur_mask & mask_core]   # 满足cur_mask的, 描边
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, edgecolors="black", s=96, facecolors="none")

    cur_data = X[cur_mask & mask_core]   # 满足cur_mask的, 填色
    plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker, edgecolors="black", s=32)
else:
    print(labels_best)

plt.title("Data separated into clusters")
# plt.show()

