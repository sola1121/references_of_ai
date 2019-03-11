import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC

file_dir = "./dat/traffic_data.txt"
X = list()

# 读取数据
with open(file_dir, 'r') as file:
    for line in file.readlines():
        data = line[:-1].split(",")
        X.append(data)

X = np.array(X)

# 标记编码
label_encoders = list()
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoders.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoders[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)   # 所有行, 排除左后一列
y = X_encoded[:, -1].astype(int)   # 所有行, 取最后一列

# 用径向基函数创建并训练回归器
param = {"kernel": "rbf", "C": 10.0}
rbf_regressor = SVC(**param)
rbf_regressor.fit(X, y)


# 交叉验证
y_pred = rbf_regressor.predict(X)
print("绝对平均误差", round(mean_absolute_error(y_true=y, y_pred=y_pred)))


# 对单一数据示例进行测试
input_data = ["Tuesday", "13:35", "San Francisco", "yes"]
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data)
    else:
        input_data_encoded[i] = int(label_encoders[count].transform([input_data[i]]))
        count += 1

output_class = rbf_regressor.predict([input_data_encoded])
print(input_data, "is predicted", output_class)

