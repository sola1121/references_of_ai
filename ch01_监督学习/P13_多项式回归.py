# 线性回归模型有一个主要的局限性, 那就是它只能把输入的数据拟合成直线, 而多项式回归模型通过拟合多项式方程来克服这类问题, 从而提高模型的准确性.
import numpy as np
from sklearn.preprocessing import PolynomialFeatures   # 映入多项式特性
from sklearn.linear_model import LinearRegression

file_dir = "./dat/data_multivar.txt"

x, y = list(), list()

with open(file_dir, "r") as file:
    for line in file.readlines():
        temp_data = [float(i) for i in line.split(",")]
        x.append(temp_data[:-1])
        y.append(temp_data[-1])

# 训练数据
x_train = np.array(x)
y_train = np.array(y)


# 使用多项式特性对数据进行处理
# 这个模型的曲率是由多项式的次数决定的. 随着模型曲率的增加, 模型变得更加准确. 
# 但是, 增加曲率的同时也增加了模型的复杂性, 因此拟合速度会变慢
polynomial = PolynomialFeatures(degree=12)   # 更改次数可以看到越来越接近实际值3, 6, 9, 12
# 多项式预处理数据, 数据将会变的多元
x_train_transformed = polynomial.fit_transform(x_train)

# 创建线性回归器
linear_regressor = LinearRegression()
# 用多项式预处理后的训练数据集训练模型
linear_regressor.fit(x_train_transformed, y_train)

# 进行取值预测
the_point = x_train[0].reshape(1, -1)   # 这里取文件中的第一值进行预测
the_point = [[0.39, 2.78, 7.11]]

poly_point = polynomial.fit_transform(the_point)

poly_point_pred = linear_regressor.predict(poly_point)

print("使用多项式回归模型预测的结果", poly_point_pred)
print("实际的结果:", y_train[0])
