import pickle

import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, 
                             explained_variance_score, r2_score )


data_dir = "../ch01_监督学习_回归_线性回归/dat/data_singlevar.txt"
x, y = list(), list()

with open(data_dir, "r") as file:
    for data in file.readlines():
        temp_x, temp_y = data.split(",")
        x.append(float(temp_x))
        y.append(float(temp_y))

x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)   # 转化为2D array的形式

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)

# 创建回归器
linear_regressor = LinearRegression()
# 训练回归器
linear_regressor.fit(x_train, y_train)

# 利用训练好的回归器预测
y_train_pred = linear_regressor.predict(x_train)
y_test_pred = linear_regressor.predict(x_test)

# 判定误差
print("Mean absolute error =", mean_absolute_error(y_true=y_test, y_pred=y_test_pred))  # 平均绝对误差
print("Mean squared error =", mean_squared_error(y_true=y_test, y_pred=y_test_pred))   # 均方误差
print("Median absolute error =", median_absolute_error(y_true=y_test, y_pred=y_test_pred))   # 中位数绝对误差
print("Explained variance score =", explained_variance_score(y_true=y_test, y_pred=y_test_pred))   # 解释方差分
print("R2 score =", r2_score(y_true=y_test, y_pred=y_test_pred))   # R方得分

# 可以通过误差来判定, 如果觉得可以, 可以保存
with open("../tmp/saved_model.pkl", "wb") as file:
    pickle.dump(linear_regressor, file)

# 作图判断
plt.figure()
plt.subplot(211)
plt.subplots_adjust(hspace=0.8)
plt.scatter(x_train, y_train, color="black")
plt.scatter(x_train, y_train_pred, color="blue")
plt.plot(x_train, y_train_pred, color="yellow")
plt.title("x_trian data and x_train_pred data compare")

plt.subplot(212)
plt.subplots_adjust(hspace=0.8)
plt.scatter(x_test, y_test, color="black")
plt.scatter(x_test, y_test_pred, color="blue")
plt.plot(x_test, y_test_pred, color="yellow")
plt.title("x_test data and y_test_pred data compare")

plt.show()
