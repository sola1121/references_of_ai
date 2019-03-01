import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression   # 逻辑回归分类器

def plot_classifier(classifier, X, y):
    #定义图形的取值范围, X点坐标的(x, y)的最大最小值, 还增加了1余量.
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    # 设置网格数据的步长
    step_size = 0.01
    # 定义网格, 使用X集的x坐标和y坐标的取值范围生成网格数据
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    print("\ngrid values\nx\n", x_values, "\ny\n", y_values)
    # 计算分类器输出结果, 对网格中的数据点进行预测, ravel将矩阵平面展开, c_将平面矩阵以列拼接
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    print("\n\noriginal mesh_out\n", mesh_output, "\n\n")
    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)
    print("reshaped mesh_out\n", mesh_output)
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


X = np.array([
    [4, 7],
    [3.5, 8],
    [3.1, 6.2],
    [0.5, 1],
    [1, 2],
    [1.2, 1.9],
    [6, 2],
    [5.7, 1.5],
    [5.4, 2.2],
])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])   # 将X数据集分为了3类

# 初始化一个逻辑回归分类器
classifier = LogisticRegression(solver="liblinear", C=10000)   # C表示dui分类错误(misclassification)的惩罚值. 
# 训练分类器
classifier.fit(X, y)
# 作画
plot_classifier(classifier, X, y)
