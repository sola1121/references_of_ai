import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def plot_feature_importances(feature_importances, title, feature_names):
    """以bar画出特征的重要性"""
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到底排序
    index_sorted = np.flipud(np.argsort(feature_importances))   # 使用argsort是由小到大返回标, 使用flipud翻转顺序
    # 让X坐标上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5   # shape这里是array的形状, 取第一个就是列的个数, 即数据多少.
    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], rotation=45)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


def plot_classifier(classifier, X, y):
    """使用分类器画出分类边界"""
    #定义图形的取值范围, X点坐标的(x, y)的最大最小值, 还增加了1余量.
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    # 设置网格数据的步长
    step_size = 0.01
    # 定义网格, 使用X集的x坐标和y坐标的取值范围生成网格数据
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # 计算分类器输出结果, 对网格中的数据点进行预测, ravel将矩阵平面展开, c_将平面矩阵以列拼接
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)
    # 用色彩区域画出分类结果
    plt.figure()
    # 选择配色方案, x坐标网格值, y坐标网格值, 在对应的(x, y)上对应的颜色值
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)   # 在网格中对应的点(x, y)与其对应的颜色值C
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors="black", linewidths=1, cmap=plt.cm.Paired)  # 单独画出训练使用的点
    # 设置图形的取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # 设置X轴与Y轴
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))
    plt.show()


def plot_cluster(model, X):
    """使用聚合算法做出不同种类的点"""
    labels = model.labels_   # 提取标记点
    centroids = model.cluster_center_   # 提取中心点
    num_clusters = len(np.unique(labels))   # 中心点的个数
    markers = ".*xv^+d<>spD1234|_"   # 用于区分不同的数据集的点的样式, 可以处理18种分类样式
    plt.figure()
    for i, marker in zip(range(num_clusters), markers[:num_clusters]):
        # 画出属于某个集群中心点的数据点
        plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='k')   # 使用labels轻松将不同的聚合类点进行分类
        # 画出集群中心点
        centroid = centroids[i]
        plt.plot(centroid[0], centroid[1], marker="o", markerfacecolor="none", markeredgecolor="black", markersize=10)
    plt.title("Clusters and their  centorids")
    plt.show()

