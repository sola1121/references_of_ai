import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

file_dir = "../ch02_监督学习_分类_创建分类器/dat/data_multivar.txt"
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 定义一个高斯朴素贝叶斯分类器
gaussiannb_classifier = GaussianNB()
# 训练模型
gaussiannb_classifier.fit(X_train, y_train)
# 使用测试集进行预测
y_test_pred = gaussiannb_classifier.predict(X_test)

# 精确度对比
accuracy = sum(y_test_pred==y_test) / y_test_pred.shape[0] * 100
print("Accuracy is %.2f" % accuracy, "%.")

# 使用所有数据集进行交叉验证
num_validations = 5
# 准确率
accuracy = cross_val_score(gaussiannb_classifier, X, y=y, scoring="accuracy", cv=num_validations)
print("Accuracy2 of the classifier: ", round(100*accuracy.mean(), 2), "%.", "     origin: ", accuracy)
# 精度
precision = cross_val_score(gaussiannb_classifier, X, y=y, scoring="precision_weighted", cv=num_validations)
print("Precision of the classifier: ", round(100*precision.mean(), 2), "%.", "     origin: ", precision)
# 召回率
recall = cross_val_score(gaussiannb_classifier, X, y=y, scoring="recall_weighted", cv=num_validations)
print("Precision of the classifier: ", round(100*recall.mean(), 2), "%.", "     origin: ", recall)
# F1得分
f1 = cross_val_score(gaussiannb_classifier, X, y=y, scoring="f1_weighted", cv=num_validations)
print("Precision of the classifier: ", round(100*f1.mean(), 2), "%.", "     origin: ", f1)


# 作图可视化
plt.figure()
# 使用网格区分区域
x_max, x_min = max(X_test[:, 0])+1.0, min(X_test[:, 0])-1.0
y_max, y_min = max(X_test[:, 1])+1.0, min(X_test[:, 1])-1.0

grid_step = 0.01
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))

mesh_output = gaussiannb_classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
mesh_output = mesh_output.reshape(x_values.shape)

plt.pcolormesh(x_values, y_values, mesh_output)
plt.scatter(X_test[:, 0], X_test[:, 1])
plt.show()
