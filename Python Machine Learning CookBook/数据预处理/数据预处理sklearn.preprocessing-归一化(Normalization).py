import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, 5.4], 
    [0, 4, -0.3, 2.1], 
    [1, 3.3, -1.9, -4.3],
])

# 归一化(Normalization)
# 数据归一化用于需要对特征向量的值进行调整时, 以保证每个特征向量的值都缩放到相同的数值范围. 
# 机器学习中最常用的归一化形式就是将特征向量调整为L1泛数, 使特征向量的数值之和为1.
data_normalized = preprocessing.normalize(data, norm="l1")
print("L1 normalized data\n", data_normalized)

# 处理后, 数据点没有因为特征的基本性质而产生较大差异, 即确保数据处于同一数量级, 提高不同特征数据的可比性.