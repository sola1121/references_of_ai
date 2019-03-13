import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def perform_clustering(X, connectivity, title, num_clusters=3, linkage="ward"):
    plt.figure()
    model = AgglomerativeClustering(n_clusters=num_clusters, connectivity=connectivity, linkage=linkage)
    model.fit(X)
    # 提取标记
    labels = model.labels_
    markers = ".vx"

    for i, marker in zip(range(num_clusters), markers):
        plt.title(title)
        # 画出属于某个集群中心的数据点
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=50, marker=marker, color="k", facecolors="none")


def get_spiral(t, noise_amplitude=.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x, y, noise_amplitude)


def add_noise(x, y, amplitude):
    X = np.concatenate((x, y))
    X += amplitude * np.random.randn(2, X.shape[1])
    return X.T


def get_rose(t, noise_amplitude=.02):
    # 设置玫瑰曲线方程; 如果变量k是奇数, 那么曲线有k朵花瓣; 如果k是偶数, 那么有2k朵花瓣
    k = 5
    r = np.cos(k*t) + .25
    x = r * np.cos(t)
    y = r * np.sin(t)

    return add_noise(x, y, noise_amplitude)


def get_hypotrochoid(t, noise_amplitude):
    a, b, h = 10.0, 2.0, 4.0
    x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t)

    return add_noise(x, y, 0)


if __name__ == "__main__":

    # 生成数据样本
    n_samples = 500
    np.random.seed(2)
    t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    X = get_spiral(t)

    # 不考虑螺旋形的数据连接性
    connectivity = None
    perform_clustering(X, connectivity, "No connectivity")

    # 根据数据连接线创建K个临近点的图形
    connectivity = kneighbors_graph(X, 10, include_self=False)
    perform_clustering(X, connectivity, "K-Neighbors connectivity")

    plt.show()

