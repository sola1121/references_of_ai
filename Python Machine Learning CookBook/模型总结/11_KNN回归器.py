import numpy as np
import numpy as np
import matplotlib; matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 生成数据
x = 100 * np.random.rand(1000, 1)  # 将会构成一个0-99的一千行一列的x轴数据集
y = np.sinc(x).ravel()   # 基本正弦函数
y += 0.02 * (0.5 - np.random.rand(y.shape[0]))   # 添加噪声

# 创建KNN回归器, 并训练. KNN回归器对有规律的点具有较好的预测
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=2)
knn_regressor.fit(x, y)

# 生成测试点
x_test = np.linspace(0, 100, num=10000).reshape(-1, 1)   # 10000个点在1-99之间, 是原数据集的10倍密度
# 使用模型进行预测
y_pred = knn_regressor.predict(x_test)

# 作画
plt.figure()
# 原数据点
plt.scatter(x, y, s=5, c="black")
# 通过预测点作画预测线
plt.plot(x_test, y_pred, color="blue", alpha=.5)
plt.show()

