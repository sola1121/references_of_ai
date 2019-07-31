import numpy as np
from sklearn import preprocessing

data = np.array([
    [3, -1.5, 2, 5.4], 
    [0, 4, -0.3, 2.1], 
    [1, 3.3, -1.9, -4.3],
])

# 二值化(Binarization), 用于将数值特征向量转换为布尔类型向量.
data_binarizer = preprocessing.Binarizer(threshold=1.4)   # 定义二值化处理器
data_binarized = data_binarizer.transform(data)   # 将数据放入二值化处理器中得到返回值
print("Binarized data\n", data_binarized)
