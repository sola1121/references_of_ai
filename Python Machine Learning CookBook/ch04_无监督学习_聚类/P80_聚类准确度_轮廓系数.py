import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

file_dir = "./dat/data_perf.txt"
data = list()
scores = list()
range_values = np.arange(2, 10)   # 用于确认集群的最佳数量

with open(file_dir, 'r') as file:
    for line in file.readlines():
        data.append(line[:-1].split(","))

data = np.array(data).astype(np.float)


for i in range_values:
    # 训练模型
    k_means = KMeans(n_clusters=i, init='k-means++', n_init=10)
    k_means.fit(data)
    score = metrics.silhouette_score(data, labels=k_means.labels_, sample_size=len(data))
    print("\nNumber of clusters =", i)
    print("Silhouette score =", score)
    scores.append(score)


# 作画并找出峰值
plt.figure(figsize=[6, 7])

plt.subplot(211)   # 画出得分条形图
plt.subplots_adjust(hspace=.8)
plt.bar(range_values, scores, width=.6, color='k', align="center")
plt.title("Silhouette score vs number of clusters")

plt.subplot(212)   # 画出原始数据
plt.scatter(data[:, 0], data[:, 1], color='k', s=30, marker="o", facecolors="none")
x_min, x_max = min(data[:, 0])-1, max(data[:, 0])+1
y_min, y_max = min(data[:, 1])-1, max(data[:, 1])+1
plt.title("Input data")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])

plt.show()

