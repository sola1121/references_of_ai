import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

# 定义一个逻辑回归分类器
logistic_classifier = LogisticRegression(solver="liblinear", C=1000)
# 进行训练
logistic_classifier.fit(X, y)


# 生成平面中的点
accuracy_step = 0.1
x_max, x_min = max(X[:, 0])+1.0, min(X[:, 0])-1.0
y_max, y_min = max(X[:, 1])+1.0, min(X[:, 1])-1.0

x_values, y_values = np.meshgrid(np.arange(x_min, x_max, accuracy_step), np.arange(y_min, y_max, accuracy_step))
X_test = np.c_[x_values.ravel(), y_values.ravel()]

# 使用训练模型进行预测
y_test = logistic_classifier.predict(X_test)
y_test = y_test.reshape(x_values.shape)   # 同化shape, 好作图
print(y_test)

# 作图
plt.figure()
plt.pcolormesh(x_values, y_values, y_test, cmap="tab10")
# plt.show()
