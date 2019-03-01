import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB   # 高斯朴素贝叶斯
from sklearn.model_selection import train_test_split


def plot_classifier(classifier, X, y):
    #定义图形的取值范围, X点坐标的(x, y)的最大最小值, 还增加了1余量.
    x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
    y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0
    # 设置网格数据的步长
    step_size = 0.01
    # 定义网格, 使用X集的x坐标和y坐标的取值范围生成网格数据
    x_values, y_values =  np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # 计算分类器输出结果, 对网格中的数据点进行预测, ravel将矩阵平面展开, c_将平面矩阵以列拼接
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)
    # 用色彩区域画出分类结果
    plt.figure()
    # 选择配色方案, x坐标网格值, y坐标网格值, 在对应的(x, y)上对应的颜色值
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)   # 在网格中对应的点(x, y)与其对应的颜色值
    plt.scatter(X[..., 0], X[..., 1], s=80, edgecolors="black", linewidths=1, cmap=plt.cm.Paired)  # 单独画出训练使用的点
    # 设置图形的取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # 设置X轴与Y轴
    plt.xticks(np.arange(x_min, x_max, 1.0))
    plt.yticks(np.arange(y_min, y_max, 1.0))
    plt.show()

file_dir = "./dat/data_multivar.txt"
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# 建立一个朴素贝叶斯分类器
gaussiannb_classifier = GaussianNB()
# 使用数据集进行训练
gaussiannb_classifier.fit(X_train, y_train)
# 使用模型对原数据集X进行预测
y_test_pred = gaussiannb_classifier.predict(X_test)

# 计算分类器的准确性
accuracy = 100.0 * sum(y_test==y_test_pred) / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

# 作画
plot_classifier(gaussiannb_classifier, X_test, y_test)
