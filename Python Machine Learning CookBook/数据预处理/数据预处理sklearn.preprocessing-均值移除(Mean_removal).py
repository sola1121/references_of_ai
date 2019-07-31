import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, 5.4], 
    [0, 4, -0.3, 2.1], 
    [1, 3.3, -1.9, -4.3],
])

# 均值移除(Mean removal), 将每个特征的平均值移除, 以保证特征均值为0(即标准化处理).
# 可以消除特征彼此之间的偏差(bias).
data_standardized = preprocessing.scale(data)
print(data_standardized)   # 均值移除后的数据
print("Mean =", data_standardized.mean(axis=0))   # 平均值
print("Std deviation = ", data_standardized.std(axis=0))   # 标准差

# 处理后 特征均值几乎是0, 标准差为1