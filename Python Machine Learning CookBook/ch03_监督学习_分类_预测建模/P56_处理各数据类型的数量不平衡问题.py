import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report

file_dir = "./dat/data_multivar_imbalance.txt"
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)

# 建立多项式SVM向量机, 使用class_weight="auto"参数让向量机根据不同类型点的数量自动调整权重, 解决不平衡问题
poly_clssifier = SVC(kernel="poly", degree=3, class_weight="balanced")
poly_clssifier.fit(X, y)

# 使用网格数据预测, 并显示
x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

mesh_ouput = poly_clssifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
mesh_ouput = mesh_ouput.reshape(x_values.shape)

plt.figure()
plt.pcolormesh(x_values, y_values, mesh_ouput)

# 对原始数据的分类, 并展示
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.scatter(class_0[:,0], class_0[:,1])
plt.scatter(class_1[:,0], class_1[:,1])
plt.title("class-0 %s, class-1 %s." % (class_0.shape[0], class_1.shape[0]))

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

