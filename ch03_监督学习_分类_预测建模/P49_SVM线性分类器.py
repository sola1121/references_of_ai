import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC   # 引入支持向量机


file_dir = "./dat/data_multivar.txt"
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(x) for x in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)

# 生成预测和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# 使用线性核函数(linear kernel)初始化一个SVM对象, 并对其进行训练.
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)

# 使用测试数据测试
y_test_pred = classifier.predict(X_test)
diff_accuracy = sum(y_test_pred == y_test) / y_test.shape[0]
print("X_test的预测和实际值对比相识度 %.4f" % diff_accuracy)

# 交叉验证
accuracy = cross_val_score(classifier, X, y, scoring="accuracy", cv=5)
f1_score = cross_val_score(classifier, X, y, scoring="f1_weighted", cv=5)
print("将所有集合X, y带入模型的交叉验证", "accuracy:", round(accuracy.mean(), 4), "   F1 score:", round(f1_score.mean(), 4))
# 输出性能报告
target_names = ["Class-" + str(n) for n in set(y)]
print("\n", classification_report(y_true=y_test, y_pred=y_test_pred, target_names=target_names))


# 可视化分类区域, 使用网格数据显示的预测显示分类区域.
x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])   # 使用网格数据带入模型进行验证
mesh_output = mesh_output.reshape(x_values.shape)

plt.figure()
plt.pcolormesh(x_values, y_values, mesh_output)
# 将原始的数据分类
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
# 原始数据作图
plt.scatter(class_0[:, 0], class_0[:, 1], facecolor="black", edgecolors="black", marker="s")
plt.scatter(class_1[:, 0], class_1[:, 1], facecolor="None", edgecolors="black", marker="s")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.show()
