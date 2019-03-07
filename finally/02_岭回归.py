from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, 
                             explained_variance_score, r2_score)

file_dir = "../ch01_监督学习_回归_线性回归/dat/data_multivar.txt"

x, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        x.append(temp_data[:-1])
        y.append(temp_data[-1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=10)

# 建立岭回归模型
ridge_regressor = Ridge(alpha=2.0, fit_intercept=True, max_iter=10000)
# 进行学习
ridge_regressor.fit(x_train, y_train)

# 进行预测
y_test_pred = ridge_regressor.predict(x_test)

# 对结果准确度进行评估
print("平均绝对误差:", mean_absolute_error(y_true=y_test, y_pred=y_test_pred))
print("均方误差:", mean_squared_error(y_true=y_test, y_pred=y_test_pred))
print("中位数绝对误差:", median_absolute_error(y_true=y_test, y_pred=y_test_pred))
print("解释方差分:", explained_variance_score(y_true=y_test, y_pred=y_test_pred))
print("R2得分:", r2_score(y_true=y_test, y_pred=y_test_pred))
