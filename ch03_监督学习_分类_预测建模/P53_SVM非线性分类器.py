import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

file_dir = "./dat/data_multivar.txt"
X, y = list(), list()

with open(file_dir, 'r') as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)

# 创建SVM非线性分类器, 并训练
poly_classiffier = SVC(kernel="poly", degree=3)   # 使用一个3次多项式
poly_classiffier.fit(X, y)

rbf_classiffier = SVC(kernel="rbf")   # 径向基函数(Radial Basis Function) 默认的
rbf_classiffier.fit(X, y)

# 作图比较, 使用新的网格数据进行预测
x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
mesh_X_test = np.c_[x_values.ravel(), y_values.ravel()]

poly_mesh_output = poly_classiffier.predict(mesh_X_test)
poly_mesh_output = poly_mesh_output.reshape(x_values.shape)

rbf_mesh_output = rbf_classiffier.predict(mesh_X_test)
rbf_mesh_output = rbf_mesh_output.reshape(x_values.shape)

# 原始数据的分类
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

# 作图
plt.figure(figsize=(5, 7))
plt.subplot(211)
plt.pcolormesh(x_values, y_values, poly_mesh_output)
plt.scatter(class_0[:,0], class_0[:,1], facecolors="black", edgecolors="black", marker="s")
plt.scatter(class_1[:,0], class_1[:,1], facecolors="None", edgecolors="black", marker="s")

plt.subplot(212)
plt.pcolormesh(x_values, y_values, rbf_mesh_output)
plt.scatter(class_0[:,0], class_0[:,1], facecolors="black", edgecolors="black", marker="s")
plt.scatter(class_1[:,0], class_1[:,1], facecolors="None", edgecolors="black", marker="s")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
