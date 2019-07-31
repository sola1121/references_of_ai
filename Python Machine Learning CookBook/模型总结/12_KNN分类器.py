import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


file_dir = "../ch05_构建推荐引擎/dat/data_nn_classifier.txt"
X, y = list(), list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        X.append(line[:-1].split(",")[:-1])
        y.append(line[:-1].split(",")[-1])

X, y = np.array(X, dtype=np.float), np.array(y, dtype=np.int)

# 建立KNN分类器, 并训练, n_neighbors=1, 使用一个最临近, 那么这将会是一个传统的分类器, 而这里使用是找出最近5个点
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=2)
knn_classifier.fit(X, y)

# 使用评估点进行分析
test_point = [4.9, 3.8]
test_result = knn_classifier.predict([test_point])   # 2
print("使用距离的分类, 这个点的类别是", test_result[0])

# 使用评估点, 列出最近邻的点, n_neighbors已经指定了将会列出来的点个数, 获取统计的最近邻点
distances, indices = knn_classifier.kneighbors([test_point])
print("\n和其最近的5个点")
for distance, index in zip(distances[0], indices[0]):
    print("点", X[index], ", 距离", round(distance, 4))

# 作图
plt.figure()
# 作画原始数据点
count = [False] * len(np.unique(y))
markers = "dsp^+"
mappers = [markers[i] for i in y]   # 根据y的值是0,1,2的特征, 替换0,1,2为d,s,p
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mappers[i], c="black")
    if not count[y[i]]:
        plt.annotate(y[i], xy=X[i], color="red", size=24)
        count[y[i]] = True
# 作画评估点
plt.scatter(test_point[0], test_point[1], marker='.', c="blue", s=120)
# 与评估点最近的5个点
for index in indices[0]:
    plt.scatter(X[index][0], X[index][1], marker='x', c="blue", s=150, linewidths=1, alpha=.5)

plt.show()

