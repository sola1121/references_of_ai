import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


file_dir = "../ch01_监督学习/dat/data_multivar.txt"

x, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        x.append(temp_data[:-1])
        y.append(temp_data[-1])

# 训练数据
x_train = np.array(x)
y_train = np.array(y)

# 定义多项式预处理器
polynomial = PolynomialFeatures(degree=10)
# 定义线性回归器
linear_regressor = LinearRegression()
# 对训练数据进行多项式处理
x_train_transformed = polynomial.fit_transform(x_train)
# 进行学习
linear_regressor.fit(x_train_transformed, y_train)

# 进行预测
the_point = x_train[0].reshape(1, -1)   # 取训练集中的第一个数
print("取值", the_point, "  对应的值", y_train[0])
# 对预测数据进行多项式处理
poly_point = polynomial.fit_transform(the_point)
# 获取预测值
poly_point_pred = linear_regressor.predict(poly_point)
print("通过预测获得的值", poly_point_pred)
