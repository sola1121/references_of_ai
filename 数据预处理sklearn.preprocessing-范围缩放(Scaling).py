import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, 5.4], 
    [0, 4, -0.3, 2.1], 
    [1, 3.3, -1.9, -4.3],
])

# 范围缩放(Scaling)
# 数据点中每个特征的数值范围可能变化很大, 因此有时将特征的数值方位缩放到合理的大小是必要的.
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)   # 使用定义好的范围预处理器处理数据
print("Min max scaled data\n", data_scaled)

# 处理后, 所有数据点的特征数值都位于指定的数值范围内.