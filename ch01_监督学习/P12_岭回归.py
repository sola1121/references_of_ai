# 线性回归对异常值很敏感, 所以引入正则化项的系数作为阀值来消除异常值对线性回归的影响, 这个方法就被称为岭回归.
import numpy as np
from sklearn.linear_model import Ridge   # 使用岭回归
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, 
                             explained_variance_score, r2_score)


file_dir = "./dat/data_multivar.txt"

x, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        x.append(temp_data[:-1])
        y.append(temp_data[-1])
        

num_training = int(0.8*len(x))
num_test = len(x) - num_training

# 因为这里的x数据是一个三维的, 就不用reshape转换了
# 训练数据
x_train = np.array(x[:num_training])
y_train = np.array(y[:num_training])
# 测试数据
x_test = np.array(x[num_training:])
y_test = np.array(y[num_training:])

# 创建岭回归器
# alpha参数控制回归器的复杂度. 当alpha趋于0时, 岭回归器就是普通最小二乘法的线性回归器.
ridge_regressor = Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
# 用训练数据集训练模型
ridge_regressor.fit(x_train, y_train)

# 使用模型对训练数据x_train的y_train进行预测
y_train_pred = ridge_regressor.predict(x_train)

# 使用模型对测试数据x_test的y_test进行预测
y_test_pred = ridge_regressor.predict(x_test)


# 对结果准确度进行评估
print("平均绝对误差:", mean_absolute_error(y_true=y_test, y_pred=y_test_pred))
print("均方误差:", mean_squared_error(y_true=y_test, y_pred=y_test_pred))
print("中位数绝对误差:", median_absolute_error(y_true=y_test, y_pred=y_test_pred))
print("解释方差分:", explained_variance_score(y_true=y_test, y_pred=y_test_pred))
print("R2得分:", r2_score(y_true=y_test, y_pred=y_test_pred))
