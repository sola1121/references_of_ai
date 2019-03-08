import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

file_dir = "../ch03_监督学习_分类_预测建模/dat/data_multivar.txt"
X, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        X.append(temp_data[:-1])
        y.append(temp_data[-1])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# MARK: 使用不同的核定义不同的SVM向量机, 并训练
# SVM线性分类器
linear_classifier = SVC(kernel="linear", class_weight="balanced")
linear_classifier.fit(X_train, y_train)

# SVM多项式分类器(非线性分类)
poly_classifier = SVC(kernel="poly", degree=4, class_weight="balanced")
poly_classifier.fit(X_train, y_train)

# SVM径向基函数分类器(非线性分类)
rbf_classifier = SVC(kernel="rbf", class_weight="balanced")
rbf_classifier.fit(X_train, y_train)

# 使用测试数据进行验证
linear_y_test_pred = linear_classifier.predict(X_test)
poly_y_test_pred = poly_classifier.predict(X_test)
rbf_y_test_pred =rbf_classifier.predict(X_test)

# 交叉验证, 准确度, 和生成性能报告
linear_accuracy = cross_val_score(linear_classifier, X, y=y, scoring="accuracy", cv=5)
poly_accuracy = cross_val_score(poly_classifier, X, y=y, scoring="accuracy", cv=5)
rbf_accuracy = cross_val_score(rbf_classifier, X, y=y, scoring="accuracy", cv=5)

print("Linear classifier model accuracy: %.2f" % linear_accuracy.mean())
print("Poly classifier model accuracy: %.2f" % poly_accuracy.mean())
print("RBF classifier model accuracy: %.2f" % rbf_accuracy.mean())

linear_report = classification_report(y_true=y_test, y_pred=linear_y_test_pred)
poly_report = classification_report(y_true=y_test, y_pred=poly_y_test_pred)
rbf_report = classification_report(y_true=y_test, y_pred=rbf_y_test_pred)

print("Linear classifier model report:\n", linear_report, "\n")
print("Poly classifier model report:\n", poly_report, "\n")
print("RBF classifier model report:\n", rbf_report, "\n")

