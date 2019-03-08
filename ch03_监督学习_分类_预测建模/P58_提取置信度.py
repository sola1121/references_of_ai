import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


file_dir = "./dat/data_multivar.txt"
X, y = list(), list()

input_datapoints = np.array([
    [2, 1.5],
    [8, 9],
    [4.8, 5.2],
    [4, 4],
    [2.5, 7],
    [7.6, 2],
    [5.4, 5.9]
])

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 定义一个径向基函数向量机, 并进行训练
rbf_classifier = SVC(kernel="rbf")
rbf_classifier.fit(X_train, y_train)

# 测量数据点与边界的距离
print("\nDistance from the boundary.")
for i in input_datapoints:
    print(i, "-->", rbf_classifier.decision_function([i]))

# 定义一个径向基函数向量机, 并要求在训练时算出概率, 并进行训练
rbf_classifier = SVC(kernel="rbf", probability=True)
rbf_classifier.fit(X_train, y_train)

# 查看点的置信度
print("\nConfidence measure.")
for i in input_datapoints:
    print(i, "-->",rbf_classifier.predict_proba([i]))

# 使用网格数据进行预测, 可视化分类区间
x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

mesh_output = rbf_classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)

# 作图
plt.figure()
plt.pcolormesh(x_values, y_values, mesh_output)

plt.scatter(input_datapoints[:, 0], input_datapoints[:, 1], color="black")

for point in input_datapoints:
    text = "({}, {})".format(point[0], point[1])
    plt.annotate(text, point)

plt.show()

