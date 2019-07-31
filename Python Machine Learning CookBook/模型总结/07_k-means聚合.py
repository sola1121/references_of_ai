import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_dir = "../ch04_无监督学习_聚类/dat/data_multivar.txt"
data = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        data.append(line[:-1].split(","))

data = np.array(data).astype("float64")

# 定义一个k-means聚合, 并训练
k_means = KMeans(n_clusters=10, init="k-means++", n_init=10, n_jobs=2)
k_means.fit(data)

# 使用网格点进行预测, 画出图形范围
x_min, x_max = min(data[:, 0])-1.0, max(data[:, 0])+1.0
y_min, y_max = min(data[:, 1])-1.0, max(data[:, 1])+1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
predict_labels = k_means.predict(np.c_[x_values.ravel(), y_values.ravel()])
predict_labels = predict_labels.reshape(x_values.shape)

plt.figure()
# 作出通过网格点标出的边界
plt.imshow(predict_labels, interpolation="nearest", 
           extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect="auto",
           origin="lower"
)
# 作出原始点
plt.scatter(data[:, 0], data[:, 1], c="blue")
# 获取聚合中心质点, 和聚合分类标签
centroids, labels = k_means.cluster_centers_, k_means.labels_
print(centroids, labels, sep="\n\n")
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", facecolor="none", edgecolors="black", linewidths=3)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

