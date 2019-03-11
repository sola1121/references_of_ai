import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

file_dir = "./dat/building_event_binary.txt"
X = list()

# 读取数据
with open(file_dir, 'r') as file:
    for line in file.readlines():
        data = line[:-1].split(",")
        X.append([data[0]] + data[2:])   # 去掉具体的日期列, 取星期, 事件, 离开大楼人数, 进入大楼人数, 活动类型.

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

# 用径向基函数, 概率输出和类型平衡方法训练SVM分类器
params = {"kernel": "rbf", "probability": True, "class_weight": "balanced"}
rbf_classifier = SVC(**params)
rbf_classifier.fit(X, y)


# 交叉验证
accuracy = cross_val_score(rbf_classifier, X, y=y, scoring="accuracy", cv=5)
print("Accuracy of the classifier %.2f" % np.mean(accuracy))


# 使用值进行推测
input_data = ["Tuesday", "12:30:00", "21", "23"]
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoders[count].transform([input_data[i]]))   # 满足2D-array
        count += 1

output_class = rbf_classifier.predict([input_data_encoded])   # 满足2D-array
print(input_data, "to predict is", label_encoders[-1].inverse_transform(output_class))

