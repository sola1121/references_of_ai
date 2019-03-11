import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans   # 从聚合中导入kmeans

file_dir = "./dat/data_multivar.txt"
data = list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        data.append(line[:-1].split(","))

data = np.array(data).astype("float64")

# 初始化一个k-means对象, 然后对其进行训练
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=10)
kmeans.fit(data)

# 作图
x_min, x_max = min(data[:, 0])-1.0, max(data[:, 0])+1.0
y_min, y_max = min(data[:, 1])-1.0, max(data[:, 1])+1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
predict_labels = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])
print(predict_labels)
predict_labels = predict_labels.reshape(x_values.shape)


plt.figure()
# 画出边界
plt.imshow(predict_labels, interpolation="nearest", 
           extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect="auto",
           origin="lower"
)

# 作画中心点
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=150, linewidths=3, color="k", zorder=10, facecolors="blue")

# 作画原始数据点
plt.scatter(data[:, 0], data[:, 1], marker="o", facecolors="none", edgecolors="black")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

