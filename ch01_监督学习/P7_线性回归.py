import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


file_dir = "./dat/data_singlevar.txt"

x, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        xt, yt = [float(i) for i in line.split(",")]
        x.append(xt)
        y.append(yt)

num_training = int(0.8*len(x))
num_test = len(x) - num_training

# 训练数据
x_train = np.array(x[:num_training]).reshape((num_training, 1))   # 变成有num_training行, 1列的空间向量
y_train = np.array(y[:num_training])
# 测试数据
x_test = np.array(x[num_training:]).reshape((num_test, 1))   # 变成有num_test行, 1列的空间向量
y_test = np.array(y[num_training:])

# 创建线性回归器
linear_regressior = LinearRegression()
# 用训练数据集训练模型
linear_regressior.fit(x_train, y_train)

# 使用模型对训练数据x_train的y_train进行预测
y_train_pred = linear_regressior.predict(x_train)

# 使用模型对测试数据x_test的y_test进行预测
y_test_pred = linear_regressior.predict(x_test)


plt.figure()
plt.subplot(311)
plt.subplots_adjust(hspace=.5)
plt.scatter(x, y, color="blue")
plt.title("origin data")


plt.subplot(312)
plt.subplots_adjust(hspace=.5)
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, y_train_pred, color="black", linewidth=4)
plt.title("training data")


plt.subplot(313)
plt.subplots_adjust(hspace=.5)
plt.scatter(x_test, y_test, color="green")
plt.plot(x_test, y_test_pred, color="black", linewidth=4)
plt.title("test data")

plt.show()